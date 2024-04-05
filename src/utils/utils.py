import os
import json
import torch
import shutil
import functools
import numpy as np
import torch.nn as nn
from dotmap import DotMap
from collections import Counter, OrderedDict


# BASE_PATH = '/mnt/aoni04/jsakuma/development/ATR-Trek'
BASE_PATH = '/Users/user/desktop/授業/lab/code/ResponseTimingEstimator_demo'

# shuffle_name
NAME_PATH = '/Users/user/desktop/授業/lab/code/ResponseTimingEstimator_DA/data/ATR_Annotated/data_-500_2000/names'
SEED = 0
TRAIN = 0.8
VALID = 0.1
TEST = 0.1

memoized = functools.lru_cache(maxsize=None)

def load_json(f_path):
    with open(f_path, 'r') as f:
        return json.load(f)


def save_json(obj, f_path):
    with open(f_path, 'w') as f:
        json.dump(obj, f, ensure_ascii=False)
        

def load_config(config_path):
    config_json = load_json(config_path)
    config = DotMap(config_json)
    
    exp_base = config.exp_base
    exp_dir = os.path.join(exp_base, config.exp_name)
    config.exp_dir = exp_dir

    exp_dir = os.path.join(BASE_PATH, exp_dir)
    os.makedirs(exp_dir, exist_ok=True)

    # save config to experiment dir
    config_out = os.path.join(exp_dir, 'config.json')
    save_json(config.toDict(), config_out)   
 
    return config


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def copy_checkpoint(folder='./', filename='checkpoint.pth.tar',
                    copyname='copy.pth.tar'):
    shutil.copyfile(os.path.join(folder, filename),
                    os.path.join(folder, copyname))


def save_checkpoint(state, is_best, folder='./', filename='checkpoint.pth.tar'):
    if not os.path.isdir(folder):
        os.mkdir(folder)
    torch.save(state, os.path.join(folder, filename))
    if is_best:
        shutil.copyfile(os.path.join(folder, filename),
                        os.path.join(folder, 'model_best.pth.tar'))


class OrderedCounter(Counter, OrderedDict):
    """Counter that remembers the order elements are first encountered"""

    def __repr__(self):
        return '%s(%r)' % (self.__class__.__name__, OrderedDict(self))

    def __reduce__(self):
        return self.__class__, (OrderedDict(self),)
    
def recursive_walk(rootdir):
    for r, dirs, files in os.walk(rootdir):
        for f in files:
            yield os.path.join(r, f)


def edit_distance(src_seq, tgt_seq):
    src_len, tgt_len = len(src_seq), len(tgt_seq)
    if src_len == 0: return tgt_len
    if tgt_len == 0: return src_len

    dist = np.zeros((src_len+1, tgt_len+1))
    for i in range(1, tgt_len+1):
        dist[0, i] = dist[0, i-1] + 1
    for i in range(1, src_len+1):
        dist[i, 0] = dist[i-1, 0] + 1
    for i in range(1, src_len+1):
        for j in range(1, tgt_len+1):
            cost = 0 if src_seq[i-1] == tgt_seq[j-1] else 1
            dist[i, j] = min(
                dist[i,j-1]+1,
                dist[i-1,j]+1,
                dist[i-1,j-1]+cost,
            )
    return dist


def get_cer(hypotheses, hypothesis_lengths, references, reference_lengths):
    assert len(hypotheses) == len(references)
    cer = []
    for i in range(len(hypotheses)):
        if len(hypotheses[i]) > 0:
            dist_i = edit_distance(
                hypotheses[i][:hypothesis_lengths[i]],
                references[i][:reference_lengths[i]],
            )
            # CER divides the edit distance by the length of the true sequence
            cer.append((dist_i[-1, -1] / float(reference_lengths[i])))
        else:
            cer.append(1)  # since we predicted empty 
    return np.mean(cer)


def l2_normalize(x, dim=1):
    return x / torch.sqrt(torch.sum(x**2, dim=dim).unsqueeze(dim))


class L2Norm(nn.Module):
    def __init__(self):
        super(L2Norm, self).__init__()
    def forward(self, x):
        y = x.reshape(x.shape[0], -1)
        y = torch.pow(y, 2.0)
        y = torch.sum(y, (1, ), keepdim=True)
        y = torch.sqrt(y)
        y = y / np.prod(x.shape[1:])
        y = y + 1e-5
        return y


class Normalize(nn.Module):
    """
    与えられた係数で正規化する
    """
    def __init__(self):
        super(Normalize, self).__init__()

    def forward(self, x, coef):
        coef = coef.reshape(-1, 1, 1, 1)
        return x / coef


def frozen_params(module):
    for p in module.parameters():
        p.requires_grad = False


@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
                      for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output


def shuffle_name():
    with open(os.path.join(NAME_PATH, 'M1_all.txt')) as f:
        lines = f.readlines()
    
    file_names = [line.replace('\n', '') for line in lines]
            
    random.seed(SEED)
    random.shuffle(file_names)
    
    LEN = len(file_names)
    
    train_names = file_names[:int(LEN*TRAIN)]
    valid_names = file_names[int(LEN*TRAIN):int(LEN*(TRAIN+VALID))]
    test_names = file_names[int(LEN*(TRAIN+VALID)):]
    
    with open(os.path.join(NAME_PATH, 'M1_train.txt'), mode='w') as f:
        for name in train_names:
            f.write(name + "\n")
    with open(os.path.join(NAME_PATH, 'M1_valid.txt'), mode='w') as f:
        for name in valid_names:
            f.write(name + "\n")
    with open(os.path.join(NAME_PATH, 'M1_test.txt'), mode='w') as f:
        for name in test_names:
            f.write(name + "\n")