# 事前に特徴量を抽出しキャッシュしておく DataSet
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import List
from os import path
import os

from ....corpus.speech.duration_v2 import DurationInfoV2Manager, DurationInfoV2
from .base import InputCalculator


def extract_ut_st_zt_ft_from_duration_info(
        info: DurationInfoV2,
        ch_ut: int, ch_st: int, framerate: int) -> torch.Tensor:
    """区間情報から，u(t)，s(t)，z(t)，f(t)を抽出する．

    Args:
      info: DurationInfo
      ch_ut: channel for u(t) calculation
      ch_st: channel for s(t) calcuration
      framerate: framerate[fps]

    output:
    u(t) ... input
    s(t) ... system utterance flag
    z(t) ... system utterance timing
    f(t) ... training flag
    """
    # --- utの計算 ---
    values = []
    for ts, te in info.vad[ch_ut]:
        # 一つ前のデータまで書き出されたデータ数から
        # 最初の時刻を計算
        prev_te = len(values) / framerate * 1000.0
        num_frames = int(np.ceil((ts - prev_te) / 1000.0 * framerate))
        values.extend([0] * num_frames)
        prev_te = len(values) / framerate * 1000.0
        num_frames = int(np.ceil((te - prev_te) / 1000.0 * framerate))
        values.extend([1] * num_frames)
    ut = - np.array(values) + 1.0
    
    #  --- 修正TURN情報の取得 ---
    turns = []
    for ts, te, tt in info.turn[ch_st]:
        if tt != 'T':
            continue
        # ts が ch_ut の VAD区間にかかっていたら，その終了区間まで修正する
        for ts_ut, te_ut in info.vad[ch_ut]:
            if ts >= ts_ut and ts <= te_ut:
                ts = te_ut
                break
        if te <= ts:
            continue
        turns.append([ts, te])
    
    # --- stの計算 ---
    values = []
    for ts, te in turns:
        # 一つ前のデータまで書き出されたデータ数から
        # 最初の時刻を計算
        prev_te = len(values) / framerate * 1000.0
        num_frames = int(np.ceil((ts - prev_te) / 1000.0 * framerate))
        values.extend([0] * num_frames)
        prev_te = len(values) / framerate * 1000.0
        num_frames = int(np.ceil((te - prev_te) / 1000.0 * framerate))
        values.extend([1] * num_frames)
    st = np.array(values)
    zt = st - np.concatenate([[0], st[:-1]]) 
    zt[zt < 0] = 0.0
    
    flag = 0.0
    ft = []
    n = min(len(ut), len(st))
    for t in range(n):
        if flag < 0.1:
            if ut[t] < 0.1 and st[t] < 0.1:
                flag = 1.0
        else:
            if st[t] > 0.9 and zt[t] < 0.1:
                flag = 0.0
        ft.append(flag)
    ft = np.array(ft, dtype=np.float32)
    
    labels = [ut, st, zt, ft]
    min_len = min([len(l) for l in labels])
    labels = [l[:min_len] for l in labels]
    labels = np.stack(labels, axis=1)
    return torch.tensor(labels, dtype=torch.float32)


class CachedLabelFeaturesDataset(Dataset):
    def __init__(self,
                 input_calculator: InputCalculator,
                 duration_info_manager: DurationInfoV2Manager,
                 dataset_name: str,
                 id_list: List[str],
                 max_offset: int = 0):
        self._input_calculator = input_calculator
        self._info_manager = duration_info_manager
        self._dataset_name = dataset_name
        self._id_list = id_list
        self._max_offset = max_offset
        
    def __len__(self):
        return len(self._id_list)

    def __getitem__(self, i):
        return self.get_data(self._id_list[i])

    def get_data(self, id: str) -> torch.Tensor:
        filename = self.get_filename(id)
        if path.exists(filename):
            data = torch.load(
                filename, map_location=self._input_calculator.device)
        else:
            dirname = path.dirname(filename)
            if not path.exists(dirname):
                os.makedirs(dirname, mode=0o755, exist_ok=True)
            data = self.build_data(id)
            torch.save(data, filename)

        if self._max_offset > 0:
            offset = np.random.randint(self._max_offset)
        else:
            offset = 0
        return data[offset:]

    def get_filename(self, id):
        from .... import config
        package_dir = config.get_package_data_dir(__package__)
        dataset_name = self._dataset_name
        filename = path.join(package_dir,
                             self._input_calculator.filename_base,
                             dataset_name,
                             '{}.torch'.format(id))
        return filename

    def build_data(self, id):
        print("building data for {}".format(id))
        info = self._info_manager.get_duration_info(id)
        batch_list = []
        for ch in (0, 1):
            ch_ut = ch
            ch_st = 1 - ch
            wav = info.wav[ch_ut]
            label = extract_ut_st_zt_ft_from_duration_info(
                info, ch_ut, ch_st,
                self._input_calculator.feature_rate)
            label = label.to(self._input_calculator.device)
            
            self._input_calculator.reset()
            ut_list = []
            feat_list = []
            for s in range(0, len(wav), 16000):
                e = s + 16000
                sub_wav = wav[s:e]
                ut, feat = self._input_calculator.calc([sub_wav])
                if ut is not None:
                    ut_list.append(ut[0])
                    feat_list.append(feat[0])
                print("\rch{}: {}/{}".format(ch, s, len(wav)),
                      end='', flush=True)
            print("")
            ut = torch.cat(ut_list, dim=0)
            feat = torch.cat(feat_list, dim=0)
            count_pad = ut.shape[0] - label.shape[0]
            if count_pad > 0:
                tensor_to_pad = torch.tensor(
                    [[1.0, 0.0, 0.0, 0.0]] * count_pad,
                    device=self._input_calculator.device)
                label = torch.cat([label, tensor_to_pad], dim=0)
            all = torch.cat([label, ut, feat], dim=1)
            all = all.unsqueeze(1)
            batch_list.append(all)
        return torch.cat(batch_list, dim=1)
