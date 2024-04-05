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



def collate_func(batch):
    specs, feats, f0s = zip(*batch)
    batch_size = len(specs)
    max_len = max([len(spec) for spec in specs])
    _, h, w = specs[0].shape
    _, cnnae_dim = feats[0].shape
    specs_ = torch.zeros(batch_size, max_len, h, w)
    feats_ = torch.zeros(batch_size, max_len, cnnae_dim)
    f0s_ = torch.zeros(batch_size, max_len)*(-100)
    lengths_ = []
    for i in range(batch_size):
        l = len(specs[i])
        specs_[i, :l] = torch.tensor(specs[i])
        feats_[i, :l] = torch.tensor(feats[i])
        f0s_[i, :l] = torch.tensor(f0s[i])
        lengths_.append(l)
    return specs_, feats_, f0s_, lengths_


def SpecAugment(spec):
    t, h, w = spec.shape
    dh = int(h * 0.1)
    mask_h = random.randint(0, int(h * 0.9))
    mask_w = random.randint(0, int(w * 0.9))
    spec[:, mask_h:mask_h+dh] = 0
    spec[:, :, mask_w] = 0
    return spec   



class ATRDataset(Dataset):
    
    def __init__(self, config):
        self.config = config
        self.data_dir = self.config.data_params.data_dir
        name_path = os.path.join(self.data_dir, 'names/M1_all.txt')
        with open(name_path) as f:
            lines = f.readlines()
        self.file_names = [line.replace('\n', '') for line in lines]
        self.data = self.get_item()
       
    def get_item(self):
        batch_list = []
        for file_name in tqdm(self.file_names):
            # spec
            spec_list = os.path.join(self.data_dir, 'spectrogram/{}/*_spectrogram.npy'.format(file_name))
            spec_list = sorted(glob.glob(spec_list))
            # feat
            feat_list = os.path.join(self.data_dir, 'cnn_ae/{}/*_spec.npy'.format(file_name))
            feat_list = sorted(glob.glob(feat_list))
            # f0
            f0_list = os.path.join(self.data_dir, 'f0/{}/*_f0.npy'.format(file_name))
            f0_list = sorted(glob.glob(f0_list))
            num_turn = len(spec_list)
            for turn in range(num_turn):
                spec = np.load(spec_list[turn])
                feat = np.load(feat_list[turn])
                f0 = np.load(f0_list[turn])
                t = min(len(spec), len(feat), len(f0))
                batch = {
                    "spec": spec[:t],
                    "feat": feat[:t],
                    "f0": f0[:t]
                }
                batch_list.append(batch)
        return batch_list
    
    def __getitem__(self, index):
        batch = self.data[index]        
        return list(batch.values())
    
    def __len__(self):
        return len(self.data)



class OtherDataset(Dataset):
    
    def __init__(self, config):
        self.config = config
        self.data_dir = self.config.data_params.data_dir
        names = os.listdir(os.path.join(self.data_dir, 'f0'))
        file_names = [name.replace('_f0.npy', '') for name in names]
        self.file_names = file_names
        self.data = self.get_item()
    
    def get_item(self):
        data = []
        for i, file_name in enumerate(tqdm(self.file_names)):
            spec = np.load(os.path.join(self.data_dir, 'spec', f'{file_name}_spec.npy'))
            feat = np.load(os.path.join(self.data_dir, 'cnnae', f'{file_name}_spec.npy'))
            f0 = np.load(os.path.join(self.data_dir, 'f0', f'{file_name}_f0.npy'))
            t = len(f0)
            N = 2000
            for i in range(t//N):
                start = N * i
                end = N * (i + 1)
                batch = {
                    'spec': spec[start:end], 
                    'feat': feat[start:end],
                    'f0': f0[start:end]
                }
                data.append(batch)
        return data
    
    def __getitem__(self, index):
        batch = self.data[index]        
        return list(batch.values())
    
    def __len__(self):
        return len(self.data)