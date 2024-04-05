import os
import glob
import json
import wave
import struct
import torch
import itertools
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import random

from tqdm import tqdm


name_mapping = {'F1(伊藤)': 'F1',
                'F2(不明)': 'F2',
                'F3(中川)': 'F3',
                'F4(川村)': 'F4',
                'M1(黒河内)': 'M1',
                'M2(平林)': 'M2',
                'M3(浜田)': 'M3',
                'M4(不明)': 'M4'
               }

# 音声最大長
MAX_LEN = 20000000

# cv_id(1/2/3/4/5) -> data_id()
cv_to_data = {
    0: {'train': [1,2,3,4,5], 'val': [6], 'test': [6]},
    1: {'train': [2,3,4,5], 'val': [1], 'test': [6]},
    2: {'train': [1,3,4,5], 'val': [2], 'test': [6]},
    3: {'train': [1,2,4,5], 'val': [3], 'test': [6]},
    4: {'train': [1,2,3,5], 'val': [4], 'test': [6]},
    5: {'train': [1,2,3,4], 'val': [5], 'test': [6]},
}

# 物理尺度 <-> 心理尺度
M = 33.9
alpha = 0.8
alpha_ = -alpha+1
K = np.exp(-1.06)
T = 310
# numpyの累乗(numpy.powerは負の値に対して使用することができない)
def numpy_exp(x, a):
    mask = np.sign(x)
    val = np.power(np.abs(x), a)
    return mask * val
# 物理尺度 -> 心理尺度
def ms_to_ipu(ms):
    mask1 = (ms < -T).astype(np.int)
    mask2 = np.logical_and(-T <= ms, ms < 0).astype(np.int)
    mask3 = np.logical_and(0 <= ms, ms < T).astype(np.int)
    mask4 = (ms >= T).astype(np.int)
    y1 = mask1 * ((numpy_exp(-ms, alpha_) - numpy_exp(T, alpha_)) / ((alpha_) * K) * (-1) - (T / M))
    y2 = mask2 * (ms / M)
    y3 = mask3 * (ms / M)
    y4 = mask4 * ((numpy_exp(ms, alpha_) - numpy_exp(T, alpha_)) / ((alpha_) * K) + (T / M))
    y = y1 + y2 + y3 + y4
    return y * M
# 心理尺度 -> 物理尺度
def ipu_to_ms(ipu):
    mask1 = (ipu < -T).astype(np.int)
    mask2 = np.logical_and(-T <= ipu, ipu < 0).astype(np.int)
    mask3 = np.logical_and(0 <= ipu, ipu < T).astype(np.int)
    mask4 = (ipu >= T).astype(np.int)
    ipu = ipu / M
    x1 = mask1 * (-1) * numpy_exp((alpha_ * K * (ipu + T/M) * (-1) + numpy_exp(T, alpha_)), 1/alpha_)
    x2 = mask2 * (ipu * M)
    x3 = mask3 * (ipu * M)
    x4 = mask4 * numpy_exp((alpha_ * K * (ipu - T/M) + numpy_exp(T, alpha_)), 1/alpha_)
    x = x1 + x2 + x3 + x4
    return x

# loss_target 作成
frame_size = 50
ipu_range = 150
def make_loss_target(target, offset):
    
    # targetが初めて1になる場所を取得
    list_idx = np.where(target==1)[0][0]
    
    # 発話タイミングをframe_sizeに合わせて変化
    frame_timing = (offset // frame_size) * frame_size
    # これを心理尺度に落とし込む
    frame_ipu = ms_to_ipu(frame_timing)
    # y = aτ + b
    a = 1 / (2 * ipu_range)
    # a = 1 / ipu_range
    b = 0.5 - a * frame_ipu
    # b = 1 - a * frame_ipu
    x = np.array([i for i in range(len(target))])
    t = frame_size * (x - list_idx) + frame_timing
    y = a * ms_to_ipu(t) + b
    
    # yを0以下のものは0, 1以上のものは1
    mask1 = (y < 0).astype(np.int)
    mask2 = np.logical_and(y >= 0, y <= 1).astype(np.int)
    mask3 = (y > 1).astype(np.int)
    loss_target = mask1 * 0 + mask2 * y + mask3 * 1
    
    return loss_target
    # return target



# 直前の発話のみ
# 出力: CNN-AE feature, VAD出力ラベル, 最後のIPU=1ラベル
class ATRDataset(Dataset):
    def __init__(self, config, cv_id, split='train', subsets=['M1_all'], speaker_list=None):
        self.config = config
        self.data_dir = self.config.data_params.data_dir
                              
        self.file_names = []
        name_path = os.path.join(self.data_dir, "names/M1_all_shuffle.txt")
        with open(name_path) as f:
            lines = f.readlines()
        self.file_names += [line.replace('\n', '') for line in lines]
        spk_file_path = os.path.join(self.data_dir, 'speaker_ids.csv')
        df_spk = pd.read_csv(spk_file_path, encoding="shift-jis")
        df_spk['operator'] = df_spk['オペレータ'].map(lambda x: name_mapping[x])
        filenames = df_spk['ファイル名'].to_list()
        spk_ids = df_spk['operator'].to_list()
        spk_dict  = dict(zip(filenames, spk_ids))
        if speaker_list is not None:
            self.file_names = [name for name in self.file_names if spk_dict[name+'.wav'] in speaker_list]
            
        path = self.config.data_params.token_list_path
        with open(path) as f:
            lines = f.readlines()
        self.tokens = [line.split()[0] for line in lines]
        
        self.frame_length = config.data_params.frame_size  # 1frame=50ms
        self.sample_rate = config.data_params.sampling_rate
        self.max_positive_length = config.data_params.max_positive_length # システム発話のターゲットの最大長(0/1の1の最大長) [frame]
        self.asr_delay = config.data_params.asr_decoder_delay # 実際のASRの遅延 [ms]        
        self.context_num = config.data_params.n_context
        self.max_timing = config.data_params.max_timing
        self.mim_timing = config.data_params.min_timing        
        self.text_dir = config.data_params.text_dir
        self.noise_num = int(config.data_params.noise_num)
        self.noise_dir = []
        self.noise_dir.append(config.data_params.noise1_dir)
        self.noise_dir.append(config.data_params.noise2_dir)
        self.noise_dir.append(config.data_params.noise3_dir)
        self.noise_dir.append(config.data_params.noise4_dir)

        # alldata(0) or cross validation(1/2/3/4/5)      
        self.minidata_len = len(self.file_names)//6
        names, self.data = self.get_data(cv_id, split)
        with open(os.path.join(config.exp_dir, f'cv{cv_id}', 'name.txt'), 'a') as f:
            f.write('------------------------------------\n')
            f.write(f'{split}\n')
            names = sorted(list(names))
            for name in names:
                f.write(f'{name}\n')
            if split == 'test': f.write('------------------------------------\n')
                
    
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
    
    
    def token2idx(self, token, unk=1, maxlen=600): 
        if token != token or token == '':
            return [0]

        token = token.replace('<eou>', '')
        idxs = [self.tokens.index(t) if t in self.tokens else unk for t in token]

        return idxs

    
    def idx2token(self, idxs): 
        token = [self.tokens[idx] for idx in idxs]

        return token

    
    def convert_frate(self, text, fr1=50, fr2=128):
        p50 = 0
        p128 = 0
        text50 = []
        text128=['']+text
        for i in range(len(text)*fr2//fr1):
            t = fr1*(i+1)
            p128 = t // fr2
            if len(text)-1<p128:
                text50.append(text[-1])
            else:
                text50.append(text[p128])

        return text50

    
    def get_last_ipu(self, turn):
        ipu_label = np.zeros(len(turn))
        sub = turn[1:]-turn[:-1]    
        if 1 in sub:
            idx = np.where(sub==1)[0][-1]
            ipu_label[idx+1:] = 1

        return ipu_label
    
    
    def get_turn_info(self, file_name, noise_dir=None):
        # 各種ファイルの読み込み
        df_turns_path = os.path.join(self.data_dir, 'csv/{}.csv'.format(file_name))
        df_vad_path = os.path.join(self.data_dir,'vad/{}.csv'.format(file_name))
        feat_list = os.path.join(self.data_dir, 'cnn_ae/{}/*_spec.npy'.format(file_name))
        spec_list = os.path.join(self.data_dir, 'spectrogram/{}/*_spectrogram.npy'.format(file_name))
        wav_list = os.path.join(self.data_dir, 'wav/{}/*.wav'.format(file_name))
        wav_start_end_list = os.path.join(self.data_dir, 'wav_start_end/{}.csv'.format(file_name))
        # clean
        if noise_dir == None:
            feat_list = os.path.join(self.data_dir, 'cnn_ae/{}/*_spec.npy'.format(file_name))
            spec_list = os.path.join(self.data_dir, 'spectrogram/{}/*_spectrogram.npy'.format(file_name))
        # noise
        else:
            feat_list = os.path.join(noise_dir, 'cnn_ae/{}/*_spec.npy'.format(file_name))
            spec_list = os.path.join(noise_dir, 'spectrogram/{}/*_spectrogram.npy'.format(file_name))  

        feat_list = sorted(glob.glob(feat_list))
        spec_list = sorted(glob.glob(spec_list))
        wav_list = sorted(glob.glob(wav_list))
        
        df = pd.read_csv(df_turns_path)
        df_vad = pd.read_csv(df_vad_path)
        df_wav = pd.read_csv(wav_start_end_list)

        N = MAX_LEN//self.sample_rate*1000

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
        num_turn = len(df['spk'])
        
        for t in range(num_turn): 
            feat_path = feat_list[t]
            spec_path = spec_list[t]
            wav_path = wav_list[t]
            feat_file_name = feat_path.split('/')[-1].replace('.npy', '').replace('_spec', '')
            spec_file_name = spec_path.split('/')[-1].replace('.npy', '').replace('_spectrogram', '')
            wav_file_name = wav_path.split('/')[-1].replace('.wav', '')
            
            assert spec_file_name == wav_file_name, "file name mismatch! check the spec-file and wav-file!"
            
            ch = df['spk'][t]
            offset = df['offset'][t]
            next_ch = df['nxt_spk'][t]
            wav_start = df_wav['wav_start'][t]//self.frame_length
            wav_end = df_wav['wav_end'][t]//self.frame_length
            cur_usr_uttr_end = df['end'][t]//self.frame_length
            timing = df['nxt_start'][t]//self.frame_length
            
            if df['nxt_start'][t] - df['offset'][t] == df['end'][t]:
                is_barge_in = False
            else:
                is_barge_in = True
            
            if wav_end - timing > self.max_positive_length:
                wav_end = timing + self.max_positive_length

            vad_user = uttr_user[wav_start:wav_end]

            turn_label = np.zeros(N//self.frame_length)
            turn_label[wav_start:cur_usr_uttr_end] = 1
            turn_label = turn_label[wav_start:wav_end]

            timing_target = np.zeros(N//self.frame_length)
            timing_target[timing:] = 1

            turn_timing_target = timing_target[wav_start:wav_end]

            last_ipu_user = self.get_last_ipu(vad_user)

            vad_label = vad_user
            last_ipu = last_ipu_user

            context = ''
            text_path = os.path.join(self.text_dir, f'{file_name}/{wav_file_name}.csv')
            df_text = pd.read_csv(text_path)
            df_text[pd.isna(df_text['asr_recog'])] = ''
            text = df_text['asr_recog'].tolist()
            text = [txt.replace('。', '')+context for txt in text]  
            
            text = self.convert_frate(text)
            
            # Delayの調節
            n = self.asr_delay//self.frame_length#-self.asr_delay2//self.frame_length
            if n >= 0:
                text = ['']*n+text
            else:
                text = text[abs(n):]+[text[-1]]*abs(n)
            
            kana = None
            
            idx = [self.token2idx(t) for t in text]
            
            length = len(vad_label)
            m = length-len(text)
            
            if m < 0:
                text = text[:m]
                idx = idx[:m]
            else:
                text = text+[text[-1]]*m
                idx = idx+[idx[-1]]*m
            
            if len(vad_label) == 0: 
                continue
            
            # 心理尺度に基づいたtarget
            turn_timing_target2 = make_loss_target(turn_timing_target, offset) 
            # print(turn_timing_target2)     

            batch = {"ch": ch,
                     "offset": offset,
                     "is_barge_in": is_barge_in,
                     "text": text,
                     "kana": kana,
                     "idx": idx,
                     "spec_path": spec_path,
                     "feat_path": feat_path,
                     "wav_path": wav_path,
                     "vad": vad_label,
                     "turn": turn_label,
                     "last_ipu": last_ipu,
                     "target": turn_timing_target,
                     "target2": turn_timing_target2,
                    }

            batch_list.append(batch)
            
        return batch_list
    
    
    def get_data(self, cv_id, split):
        names = []
        data = []
        num_list = cv_to_data[cv_id][split]
        for num in num_list:
            names += self.file_names[self.minidata_len*(num-1):self.minidata_len*num]
        for name in tqdm(names):
            data += self.get_turn_info(name)
            if split == 'train':
                for i in range(self.noise_num):
                    data += self.get_turn_info(name, noise_dir=self.noise_dir[i])
        return names, data
    
    
    def __getitem__(self, index):
        batch = self.data[index]
        feat = np.load(batch['feat_path'])
        spec = np.load(batch['spec_path'])
        text = batch['text']
        idx = batch['idx']
        vad = batch['vad']
        turn = batch['turn']
        last_ipu = batch['last_ipu']
        target = batch['target']
        target2 = batch['target2']
        
        length = min(len(feat), len(spec), len(vad), len(turn), len(target), len(text))
        batch['text'] = text[:length]
        batch['idx'] = idx[:length]
        batch['vad'] = vad[:length]
        batch['turn'] = turn[:length]
        batch['last_ipu'] = last_ipu[:length]
        batch['target'] = target[:length]
        batch['target2'] = target2[:length]
        batch['spec'] = spec[:length]
        batch['feat'] = feat[:length]
        batch['indices'] = index
        
        wav_len = int(length * self.sample_rate * self.frame_length / 1000)
        
        assert len(batch['spec'])==len(batch['vad']), "error"
        
        return list(batch.values())

    
    def __len__(self):
        return len(self.data)
    

def collate_fn(batch):
    chs, offsets, is_barge_in, texts, kanas, idxs, spec_paths, feat_paths, wav_paths, vad, turn, last_ipu, targets, targets2, specs, feats, indices = zip(*batch)
    
    batch_size = len(chs)
    
    max_len = max([len(f) for f in specs])
    _, h, w = specs[0].shape
    _, cnnae_dim = feats[0].shape
    
    text_ = []
    # kana_ = []
    vad_ = torch.zeros(batch_size, max_len).long()
    turn_ = torch.zeros(batch_size, max_len).long()
    last_ipu_ = torch.zeros(batch_size, max_len).long()
    target_ = torch.ones(batch_size, max_len).long()*(-100)
    spec_ = torch.zeros(batch_size, max_len, h, w)
    feat_ = torch.zeros(batch_size, max_len, cnnae_dim)
    target2_ = torch.ones(batch_size, max_len)*(-100)
    
    
    input_lengths = []
    for i in range(batch_size):
        
        l1 = len(specs[i])
        input_lengths.append(l1)
                
        text_.append(texts[i]+['[PAD]']*(max_len-l1))
        vad_[i, :l1] = torch.tensor(vad[i]).long()       
        turn_[i, :l1] = torch.tensor(turn[i]).long()       
        last_ipu_[i, :l1] = torch.tensor(last_ipu[i]).long()
        target_[i, :l1] = torch.tensor(targets[i]).long() 
        feat_[i, :l1] = torch.tensor(feats[i])    
        spec_[i, :l1] = torch.tensor(specs[i])
        target2_[i, :l1] = torch.tensor(targets2[i])
       
    input_lengths = torch.tensor(input_lengths).long()
        
    return chs, text_, kanas, idxs, vad_, turn_, last_ipu_, target_, spec_, input_lengths, offsets, indices, is_barge_in, wav_paths, wav_paths, target2_, feat_
    

# torch.utils.data.DataLoader によってデータローダの作成
def create_dataloader(dataset, batch_size, shuffle=True, pin_memory=True, num_workers=2):
    loader = DataLoader(
        dataset, 
        batch_size=batch_size,
        shuffle=shuffle, 
        pin_memory=pin_memory,
        num_workers=num_workers,
        collate_fn= lambda x: collate_fn(x),
    )
    return loader


# datasetの取得 -> クラスATRDatasetの作成
def get_dataset(config, cv_id, split='train', subsets=["M1_all"], speaker_list=None):
    dataset = ATRDataset(config, cv_id, split, subsets, speaker_list)
    return dataset


# dataloaderの取得 -> create_dataloaderによって作成
def get_dataloader(dataset, config, shuffle=True):
    dataloader = create_dataloader(dataset, config.optim_params.batch_size, shuffle=shuffle)
    return dataloader