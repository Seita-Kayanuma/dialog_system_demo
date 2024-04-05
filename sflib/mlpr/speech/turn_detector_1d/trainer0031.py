# 0014 -> 0031
#   不正解タイミングで下げる量を調整（実験用） -> 0.6
# 0013 -> 0014
#   不正解タイミングで下げる量を大きくした(0.75を目指していたのを0.2にした）
# 0003 -> 0013
#   一旦ノイズ付与を諦め，データキャッシュを作成して学習を高速化した．
# 0002 -> 0003
#   y(t) -> 0.8 と y(t) -> 0.75 のそれぞれのロスを sum で reduce し，
#   バッチ数で割る（要はそれぞれの個数が反映される）ようにした
# 0001 -> 0002
#   学習フラグf(t)を下げるタイミングをz(t)が立った次のフレームになるようにした
#   y(t) -> 0.8  の学習を，f(t) & z(t) の時のみに限定した
#   y(t) -> 0.75 の学習を，y(t) が 0.8 を上方向にクロスしたタイミングに限定した
#   最大10秒のオフセットをつけるようにした
#   backprop_lenを3000（1分）にし，batch_sizeを10にした
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import tqdm
import numpy as np
import os
from os import path

from ....corpus.speech.duration_v2 import DurationInfoV2Manager, DurationInfoV2
from ....corpus.speech.waseda_soma \
    import WASEDA_SOMA, DurationInfoV2ManagerWasedaSoma

from ....ext.torch.callbacks.reporters \
    import StandardReporter, CsvWriterReporter
from ....ext.torch.callbacks.early_stopper import EarlyStopper
from ....ext.torch.callbacks.train import ClippingGrad

from .base import \
    TurnDetector1dTrainer, TorchTrainerForTurnDetector1d, TurnDetector1d, \
    InputCalculator


class DurationInfoV2Dataset(Dataset):
    def __init__(self,
                 input_calculator: InputCalculator,
                 duration_info_manager: DurationInfoV2Manager,
                 dataset_name, id_list):
        self._input_calculator = input_calculator
        self._info_manager = duration_info_manager
        self._dataset_name = dataset_name
        self._id_list = id_list

    def __len__(self):
        return len(self._id_list)

    def __getitem__(self, i):
        return self.get_data(self._id_list[i])

    def get_data(self, id):
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
        offset = np.random.randint(300)
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


def extract_ut_st_zt_ft_from_duration_info(
        info: DurationInfoV2, ch_ut, ch_st, framerate):
    """
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


class CollateDurationInfo:
    def __init__(self, frame_rate):
        self._frame_rate = frame_rate

    def __call__(self, duration_info_list):
        wav_list = []
        target_list = []
        for ch in range(2):
            wavs, targets = self._collate(duration_info_list, ch)
            wav_list.extend(wavs)
            target_list.extend(targets)

        for duration_info in duration_info_list:
            duration_info.clear_cache()

        return wav_list, target_list

    def _collate(self, duration_info_list, ch):
        wav_list = [info.wav[ch] for info in duration_info_list]

        ch_ut = ch
        ch_st = 1 - ch
        labels_list = [
            extract_ut_st_zt_ft_from_duration_info(
                info, ch_ut, ch_st, self._frame_rate)
            for info in duration_info_list
        ]
        return wav_list, labels_list

def collate_fn_none(batch):
    return batch
    
def make_data_test():
    from .base import construct_input_calculator
    autoencoder_construct_args = ([12, 13, 2], {})
    voice_activity_detector_construct_args = ([1, 1, 3, autoencoder_construct_args], {})
    input_calculator_number = 1
    input_calculator_construct_args = ([voice_activity_detector_construct_args,], {})
    input_calculator = construct_input_calculator(
        input_calculator_number, input_calculator_construct_args)
    waseda_soma = WASEDA_SOMA()
    dim = DurationInfoV2ManagerWasedaSoma(waseda_soma)
    id_list_train = waseda_soma.get_id_list()[:300]

    dataset_train = DurationInfoV2Dataset(input_calculator, dim, 'waseda_soma',
                                          id_list_train)
    train_loader = DataLoader(
        dataset_train,
        # batch_size=1,
        batch_size=3,
        collate_fn=collate_fn_none,
        # num_workers=2,
        num_workers=0,
        # shuffle=False,
        shuffle=False,
    )

    for data in train_loader:
        import ipdb; ipdb.set_trace()

    
# --- 
class TimingLoss(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._device = 'cpu'

    def to(self, device):
        super().to(device)
        self._device = device
        return self

    def forward(self, y, target, y_len):
        y_len = y.shape[0]
        t_len = target.shape[0]
        min_len = min(y_len, t_len)
        y = y[:min_len, :, :]
        target = target[:min_len, :, :]

        yt = y[:, :, 0:1]
        # at_locked = y[:, :, 1:2]
        # at_unlocked = y[:, :, 2:3]
        
        # ut = target[:, :, 0:1]
        # st = target[:, :, 1:2]
        zt = target[:, :, 2:3]
        ft = target[:, :, 3:4]

        # z(t) == 1 の時の y(t) を 0.8 に近づけるロス
        z_flag = (zt > 0.9) & (ft > 0.9)
        z_count = z_flag.sum()
        if z_count > 0:
            yt_sub = yt[z_flag]
            target = torch.ones_like(yt_sub, device=self._device) * 0.8
            z_loss = F.mse_loss(yt_sub, target, reduction='sum')
        else:
            z_loss = 0

        # z(t) == 0，f(t) == 1の時，0.8 より小さくするロス
        yt1 = torch.cat([yt[:1, :, :], yt[:-1, :, :]], dim=0)
        nz_flag = (zt < 0.1) & (ft > 0.9) & (yt >= 0.8) & (yt1 < 0.8)
        nz_count = nz_flag.sum()
        if nz_count > 0:
            yt_sub = yt[nz_flag]
            target = torch.ones_like(yt_sub, device=self._device) * 0.6
            nz_loss = F.mse_loss(yt_sub, target, reduction='sum')
        else:
            nz_loss = 0

        # print(z_count, nz_count)

        loss_count = z_count + nz_count
        if loss_count > 0:
            batch_size = y.shape[1]
            loss = (z_loss + nz_loss) / batch_size
        else:
            loss = 0
            
        # import ipdb; ipdb.set_trace()
        return loss, loss_count
    
                             
class TurnDetector1dTrainer0031(TurnDetector1dTrainer):
    def __init__(self, backprop_len=300):
        super().__init__()
        self._backprop_len = backprop_len

    def build_torch_trainer(self, detector: TurnDetector1d):
        criterion = TimingLoss()
        criterion = criterion.to(detector.device)
        optimizer = optim.Adam(detector.torch_model.parameters())
        
        waseda_soma = WASEDA_SOMA()
        dim = DurationInfoV2ManagerWasedaSoma(waseda_soma)
        
        id_list_train = waseda_soma.get_id_list()[:300]
        dataset_train = DurationInfoV2Dataset(
            detector.input_calculator,
            dim, 'waseda_soma', id_list_train)

        id_list_vali = waseda_soma.get_id_list()[300:310]
        dataset_vali = DurationInfoV2Dataset(
            detector.input_calculator,
            dim, 'waseda_soma', id_list_vali)

        train_loader = DataLoader(
            dataset_train,
            # batch_size=1,
            batch_size=30,
            # num_workers=2,
            num_workers=0,
            # shuffle=False,
            shuffle=True,
            collate_fn=collate_fn_none,
        )
        vali_loader = DataLoader(
            dataset_vali,
            batch_size=10,
            # num_workers=2,
            num_workers=0,
            shuffle=False,
            collate_fn=collate_fn_none,
        )
        
        callbacks = [
            ClippingGrad(1.),
            StandardReporter(train_report_interval=1,
                             validation_report_interval=1),
            CsvWriterReporter(detector.get_csv_log_filename()),
            # Snapshot(final_filename=turn_detector.get_model_filename()),
            # EarlyStopper(patience=100, verbose=True),
        ]

        trainer = TorchTrainerForTurnDetector1d(
            detector,
            criterion,
            optimizer,
            train_loader,
            vali_loader,
            callbacks=callbacks,
            device=detector.device,
            epoch=100,
            backprop_len=self._backprop_len
        )

        return trainer
        
