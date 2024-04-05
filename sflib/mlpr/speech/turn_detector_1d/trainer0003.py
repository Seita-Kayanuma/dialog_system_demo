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

from ....sound.sigproc.noise import NoiseAdder, IntermittentNoiseAdder
from ....corpus.speech.duration_v2 import DurationInfoV2Manager, DurationInfoV2
from ....corpus.speech.waseda_soma \
    import WASEDA_SOMA, DurationInfoV2ManagerWasedaSoma
from ....corpus.noise import JEIDA, SoundffectLab, Fujie

from ....ext.torch.callbacks.reporters \
    import StandardReporter, CsvWriterReporter
from ....ext.torch.callbacks.early_stopper import EarlyStopper
from ....ext.torch.callbacks.train import ClippingGrad

from .base import \
    TurnDetector1dTrainer, TorchTrainerForTurnDetector1d, TurnDetector1d


class DurationInfoV2Dataset(Dataset):
    def __init__(self, duration_info_manager: DurationInfoV2Manager, id_list):
        self._info_manager = duration_info_manager
        self._id_list = id_list

        self._duration_info_list = []
        for id in tqdm.tqdm(self._id_list):
            self._duration_info_list.append(
                self._info_manager.get_duration_info(id))

    def __len__(self):
        return len(self._duration_info_list)

    def __getitem__(self, i):
        return self._duration_info_list[i]


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
    def __init__(self, frame_rate,
                 noise_adder: NoiseAdder = None,
                 interm_noise_adder: IntermittentNoiseAdder = None):
        self._frame_rate = frame_rate
        self._noise_adder = noise_adder
        self._interm_noise_adder = interm_noise_adder

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
        extended = False
        if self._noise_adder is not None or \
           self._interm_noise_adder is not None:
            if self._noise_adder is not None:
                noised_wav_list = [
                    self._noise_adder.add_noise(wav) for wav in wav_list
                ]
            if self._interm_noise_adder is not None:
                noised_wav_list = [
                    self._interm_noise_adder.add_noise(wav)
                    for wav in noised_wav_list
                ]
            wav_list.extend(noised_wav_list)
            extended = True

        offset = float(np.random.rand(1) * 10)  # 最大10秒のオフセットをつける
        offset_samples = int(offset * 16000)
        for i in range(len(wav_list)):
            wav_list[i] = wav_list[i][offset_samples:]

        offset_frames = int(offset * self._frame_rate)

        ch_ut = ch
        ch_st = 1 - ch
        labels_list = [
            extract_ut_st_zt_ft_from_duration_info(
                info, ch_ut, ch_st, self._frame_rate)[offset_frames:]
            for info in duration_info_list
        ]
        if extended:
            labels_list = labels_list * 2
        return wav_list, labels_list


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
            target = (torch.ones_like(yt_sub) * 0.8).to(self._device)
            z_loss = F.mse_loss(yt_sub, target, reduction='sum')
        else:
            z_loss = 0

        # z(t) == 0，f(t) == 1の時，0.8 より小さくするロス
        yt1 = torch.cat([yt[:1, :, :], yt[:-1, :, :]], dim=0)
        nz_flag = (zt < 0.1) & (ft > 0.9) & (yt >= 0.8) & (yt1 < 0.8)
        nz_count = nz_flag.sum()
        if nz_count > 0:
            yt_sub = yt[nz_flag]
            # target = yt_sub.clone().detach() - 0.2
            target = (torch.ones_like(yt_sub) * 0.75).to(self._device)
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
    
                             
class TurnDetector1dTrainer0003(TurnDetector1dTrainer):
    def __init__(self, backprop_len=3000):
        super().__init__()
        self._backprop_len = backprop_len

    def build_torch_trainer(self, detector: TurnDetector1d):
        criterion = TimingLoss()
        criterion = criterion.to(detector.device)
        optimizer = optim.Adam(detector.torch_model.parameters())
        
        waseda_soma = WASEDA_SOMA()
        dim = DurationInfoV2ManagerWasedaSoma(waseda_soma)
        id_list_train = waseda_soma.get_id_list()[:300]
        dataset_train = DurationInfoV2Dataset(dim, id_list_train)

        id_list_vali = waseda_soma.get_id_list()[300:310]
        dataset_vali = DurationInfoV2Dataset(dim, id_list_vali)

        jeida = JEIDA()
        noise_adder = NoiseAdder(jeida.get_wav_path_list())
        
        soundffectlab = SoundffectLab()
        fujie = Fujie()
        interm_noise_adder = IntermittentNoiseAdder(
            soundffectlab.get_wav_path_list() + fujie.get_wav_path_list())

        collate_fn_trainer = CollateDurationInfo(
            detector.input_calculator.feature_rate,
            noise_adder, interm_noise_adder)
        collate_fn_valid = CollateDurationInfo(
            detector.input_calculator.feature_rate)

        train_loader = DataLoader(
            dataset_train,
            # batch_size=1,
            batch_size=10,
            collate_fn=collate_fn_trainer,
            # num_workers=2,
            num_workers=0,
            # shuffle=False,
            shuffle=True,
        )
        vali_loader = DataLoader(
            dataset_vali,
            batch_size=1,
            collate_fn=collate_fn_valid,
            # num_workers=2,
            num_workers=0,
            shuffle=False,
        )
        callbacks = [
            ClippingGrad(1.),
            StandardReporter(train_report_interval=1,
                             validation_report_interval=1),
            CsvWriterReporter(detector.get_csv_log_filename()),
            # Snapshot(final_filename=turn_detector.get_model_filename()),
            EarlyStopper(patience=100, verbose=True),
        ]

        trainer = TorchTrainerForTurnDetector1d(
            detector,
            criterion,
            optimizer,
            train_loader,
            vali_loader,
            callbacks=callbacks,
            device=detector.device,
            epoch=20,
            backprop_len=self._backprop_len
        )

        return trainer
        
