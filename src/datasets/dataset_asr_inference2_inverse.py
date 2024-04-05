import os
import glob
import json
import wave
import struct
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from tqdm import tqdm


# 直前の発話のみ
# 出力: CNN-AE feature, VAD出力ラベル, 最後のIPU=1ラベル
class ATRDataset(Dataset):
    def __init__(self, config, speaker_list=None):
        self.config = config
        self.data_dir = self.config.data_dir
        
#         name_path = "/mnt/aoni04/jsakuma/data/ATR2022/asr/names/{}.txt".format(split)
#         with open(name_path) as f:
#             lines = f.readlines()
    
#         self.file_names = [line.replace('\n', '') for line in lines]
        self.file_names = [file_path.split('/')[-1].replace('.csv', '') for file_path in sorted(glob.glob(os.path.join(self.data_dir, 'csv', '*.csv')))]
#         if speaker_list is not None:
#             self.file_names = [name for name in self.file_names if spk_dict[name+'.wav'] in speaker_list]
        
        self.offset = 300  # VADのhang over
        self.frame_length = 50  # 1frame=50ms
        self.sample_rate = 16000
        self.max_positive_length = 40 # システム発話のターゲットの最大長(0/1の1の最大長) [frame]
        self.N = 10  # 現時刻含めたNフレーム先の発話状態を予測
        
        self.data = self.get_data()#[::-1]
        
    def read_wav(self, wavpath):
        wf = wave.open(wavpath, 'r')

        # waveファイルが持つ性質を取得
        ch = wf.getnchannels()
        width = wf.getsampwidth()
        fr = wf.getframerate()
        fn = wf.getnframes()

        x = wf.readframes(wf.getnframes()) #frameの読み込み
        x = np.frombuffer(x, dtype= "int16") #numpy.arrayに変換

        return x
    
    def get_turn_info(self, file_name):
        # 各種ファイルの読み込み
        df_turns_path = os.path.join(self.data_dir, 'csv/{}.csv'.format(file_name))
        df_vad_path = os.path.join(self.data_dir,'vad/{}.csv'.format(file_name))       
        #feat_list = os.path.join(self.data_dir, 'cnn_ae/{}/*_spec.npy'.format(file_name))
        wav_list = os.path.join(self.data_dir, 'wav/{}/*.wav'.format(file_name))
        #feat_list = sorted(glob.glob(feat_list))
        wav_list = sorted(glob.glob(wav_list))
        
        df = pd.read_csv(df_turns_path)
        df_vad = pd.read_csv(df_vad_path)

        # 対話全体の長さ確認
        if len(df) > 0: N = (max(df['nxt_end'].iloc[0], df['end'].iloc[0]+self.max_positive_length*self.frame_length)-df['start'].iloc[0])//self.sample_rate*1000
        else: N = 0

        # vadの結果
        uttr_user = np.zeros(N//self.frame_length)
        uttr_agent = np.zeros(N//self.frame_length)      
        for i in range(len(df_vad)):
            spk = df_vad['spk'].iloc[i]
            start = (df_vad['start'].iloc[i]) // self.frame_length
            end = (df_vad['end'].iloc[i]) // self.frame_length

            if spk==1:
                uttr_user[start:end]=1
            else:
                uttr_agent[start:end]=1

        batch_list = []
        num_turn = len(df)
        for t in range(num_turn): 
            #feat_path = feat_list[t]
            wav_path = wav_list[t]
            ch = df['spk'].iloc[t]
            offset = df['offset'].iloc[t]
            start=df['start'].iloc[t]//self.frame_length
            cur_end = df['end'].iloc[t]//self.frame_length
            nxt_start = df['nxt_start'].iloc[t]//self.frame_length
            nxt_end = df['nxt_end'].iloc[t]//self.frame_length

            end = max(nxt_start + self.max_positive_length, nxt_end)
            
            vad_user = uttr_user[start:end]
            vad_agent = uttr_agent[start:end]
            
            turn_label = np.zeros(N//self.frame_length)
            turn_label[start:cur_end] = 1
            turn_label = turn_label[start:end]

            timing_target = np.zeros(N//self.frame_length)
            timing_target[nxt_start:] = 1
            
            turn_timing_target = timing_target[start:end]

            eou = cur_end-start           
            batch = {"ch": ch,
                     "feat_path": None,
                     "wav_path": wav_path,
                     "vad": vad_user,
                     "turn": turn_label,                     
                     "target": turn_timing_target,
                     "eou": eou,
                    }

            batch_list.append(batch)
            
        return batch_list
    
    def get_data(self):
        data = []
        for file_name in tqdm(self.file_names[::-1]):
            data += self.get_turn_info(file_name)
                        
        return data            
        
    def __getitem__(self, index):
        batch = self.data[index]
        #feat = np.load(batch['feat_path'])
        wav = self.read_wav(batch['wav_path'])
        vad = batch['vad']
        turn = batch['turn']
        target = batch['target']
        eou = batch['eou']
        
        length = min(len(vad), len(turn), len(target))
        batch['vad'] = vad[:length]
        batch['turn'] = turn[:length]
        batch['target'] = target[:length]
        
        wav_len = int((eou+1) * self.sample_rate * self.frame_length / 1000)        
        batch['wav'] = wav#[:wav_len]        
        
        return list(batch.values())

    def __len__(self):
        # raise NotImplementedError
        return len(self.data)
    

def collate_fn(batch):
    chs, feat_paths, wav_paths, vad, turn, targets, eou, wavs = zip(*batch)
    
    batch_size = len(chs)
    
    max_len = max([len(v) for v in vad])
    max_wav_len = max([len(w) for w in wavs])
    
    vad_ = torch.zeros(batch_size, max_len).long()
    turn_ = torch.zeros(batch_size, max_len).long()
    target_ = torch.ones(batch_size, max_len).long()*(-100)
    wav_ = torch.zeros(batch_size, max_wav_len)
    
    input_lengths = []
    wav_lengths = []
    for i in range(batch_size):
        l1 = len(vad[i])
        input_lengths.append(l1)
        
        l2 = len(wavs[i])
        wav_lengths.append(l2)
        
        vad_[i, :l1] = torch.tensor(vad[i]).long()       
        turn_[i, :l1] = torch.tensor(turn[i]).long()       
        target_[i, :l1] = torch.tensor(targets[i]).long()       
        wav_[i, :l2] = torch.from_numpy(wavs[i].astype(np.float32)).clone()
        
    input_lengths = torch.tensor(input_lengths).long()
    wav_lengths = torch.tensor(wav_lengths).long()
        
    return chs, vad_, turn_, target_, input_lengths, wav_, wav_lengths, wav_paths
    
    
def create_dataloader(dataset, batch_size, shuffle=False, pin_memory=True, num_workers=2):
    loader = DataLoader(
        dataset, 
        batch_size=batch_size,
        shuffle=shuffle, 
        pin_memory=pin_memory,
        num_workers=num_workers,
        collate_fn= lambda x: collate_fn(x),
    )
    return loader

def get_dataset(config, speaker_list=None):
    dataset = ATRDataset(config, speaker_list)
    return dataset


def get_dataloader(dataset, config):
    dataloader = create_dataloader(dataset, config.optim_params.batch_size, shuffle=False)
    return dataloader