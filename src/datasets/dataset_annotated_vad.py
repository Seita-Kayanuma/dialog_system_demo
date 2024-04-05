import os
import glob
import torch
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence



MAX_LEN = 20000000


# 正規化
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])


def collate_func(batch):
    specs, feats, vads =  zip(*batch)
    batch_size = len(specs)
    max_len = max([len(spec) for spec in specs])
    _, h, w = specs[0].shape
    _, cnnae_dim = feats[0].shape
    specs_ = torch.zeros(batch_size, max_len, h, w)
    feats_ = torch.zeros(batch_size, max_len, cnnae_dim)
    vads_ = torch.zeros(batch_size, max_len)*(-100)
    lengths_ = []
    for i in range(batch_size):
        l = len(specs[i])
        specs_[i, :l] = torch.tensor(specs[i])
        feats_[i, :l] = torch.tensor(feats[i])
        vads_[i, :l] = torch.tensor(vads[i])
        lengths_.append(l)
    return specs_, feats_, vads_, lengths_


def SpecAugment(spec):
    t, h, w = spec.shape
    dh = int(h * 0.1)
    mask_h = random.randint(0, int(h * 0.9))
    mask_w = random.randint(0, int(w * 0.9))
    spec[:, mask_h:mask_h+dh] = 0
    spec[:, :, mask_w] = 0
    return spec


def min_max_scaling(spec):
    return (spec - np.min(spec)) / (np.max(spec) - np.min(spec))    



class ATRDataset(Dataset):
    
    def __init__(self, config):
        self.config = config
        self.data_dir = self.config.data_params.data_dir
        self.frame_length = config.data_params.frame_size
        self.sample_rate = config.data_params.sampling_rate
        name_path = os.path.join(self.data_dir, 'names/M1_all.txt')
        with open(name_path) as f:
            lines = f.readlines()
        self.file_names = [line.replace('\n', '') for line in lines]
        self.data = self.get_item()
    
       
    def get_item(self):
        
        batch_list = []
        
        for file_name in tqdm(self.file_names):

            # vad
            df_vad_path = os.path.join(self.data_dir,'vad/{}.csv'.format(file_name))
            df_vad = pd.read_csv(df_vad_path)
            # spec
            spec_list = os.path.join(self.data_dir, 'spectrogram/{}/*_spectrogram.npy'.format(file_name))
            # spec_list = os.path.join(self.data_dir, 'spectrogram_yaguchinoise/{}/*_spectrogram.npy'.format(file_name))
            spec_list = sorted(glob.glob(spec_list))
            # feat
            feat_list = os.path.join(self.data_dir, 'cnn_ae/{}/*_spec.npy'.format(file_name))
            feat_list = sorted(glob.glob(feat_list))
            # wav start end
            wav_start_end_list = os.path.join(self.data_dir, 'wav_start_end/{}.csv'.format(file_name))
            df_wav = pd.read_csv(wav_start_end_list)
            # f0
            f0_list = os.path.join(self.data_dir, 'f0/{}/*_f0.npy'.format(file_name))
            f0_list = sorted(glob.glob(f0_list))
            
            # vad
            N = MAX_LEN//self.sample_rate*1000
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

            num_turn = len(df_wav)
            for turn in range(num_turn):
                spec = np.load(spec_list[turn])
                feat = np.load(feat_list[turn])
                f0 = torch.tensor(np.load(f0_list[turn]))
                f0 = (f0 > 0).int()
                wav_start = df_wav['wav_start'][turn]//self.frame_length
                wav_end = df_wav['wav_end'][turn]//self.frame_length
                vad_user = uttr_user[wav_start:wav_end]
                t = min(len(spec), len(feat), len(vad_user))
                
                # normalization
                # spec = transform(spec.transpose((1, 2, 0)))
                
                batch = {
                    "spec": spec[:t],
                    "feat": feat[:t],
                    "vad": vad_user[:t]
                    # "vad": f0[:t]
                }
                batch_list.append(batch)
                
                # SpecAugment
                """
                batch2 = {
                    "spec":SpecAugment(spec),
                    "vad": vad_user[:t]
                    # "vad": f0[:t]
                }
                batch_list.append(batch2)
                """
                    
        return batch_list
    
    
    def __getitem__(self, index):
        batch = self.data[index]        
        return list(batch.values())
    
    
    def __len__(self):
        return len(self.data)



class OtherDataset(Dataset):
    
    def __init__(self, config):
        self.config = config
        self.data_path = self.config.data_params.data_dir
        names = os.listdir(os.path.join(self.data_path, 'f0'))
        file_names = [name.replace('_f0.npy', '') for name in names]
        self.file_names = file_names
        self.data = self.get_item()
    
       
    def get_item(self):
        data = []
        for i, file_name in enumerate(tqdm(self.file_names)):
            # if i >= 10: break
            spec = np.load(os.path.join(self.data_path, 'spec', f'{file_name}_spec.npy'))
            spec = torch.tensor(spec, dtype=torch.float32)
            pitch = np.load(os.path.join(self.data_path, 'f0', f'{file_name}_f0.npy'))
            vad = (torch.tensor(pitch, dtype=torch.float32) > 0).int()
            t = len(pitch)
            N = 2000
            for i in range(t//N):
                start = N * i
                end = N * (i + 1)
                batch = {'spec':spec[start:end], 'pitch':vad[start:end]}
                data.append(batch)
            # batch = {'spec':spec, 'pitch':pitch, 'file_name': file_name}
            # data.append(batch)

        return data
    
    
    def __getitem__(self, index):
        batch = self.data[index]        
        return list(batch.values())
    
    
    def __len__(self):
        return len(self.data)