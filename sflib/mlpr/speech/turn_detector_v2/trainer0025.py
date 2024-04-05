# 25
#  24が基本で，出力側には tanh を被せず，ターゲットへの重みを 0.1〜1.0 に抑える
from .base import TorchTrainerForTurnDetector, TurnDetectorTrainer, \
    TurnDetector, TurnDetectorFeatureExtractor
from ....ext.torch.callbacks.reporters \
    import StandardReporter, CsvWriterReporter
from ....ext.torch.callbacks.early_stopper import EarlyStopper
from ....ext.torch.callbacks.snapshot import Snapshot
from ....ext.torch.callbacks.train import ClippingGrad

from ....corpus.speech.duration import DurationInfoManager, DurationInfo
from ....corpus.speech.waseda_soma import WASEDA_SOMA, DurationInfoManagerWasedaSoma
from sflib.sound.sigproc.noise import NoiseAdder, IntermittentNoiseAdder
from sflib.corpus.noise import JEIDA, SoundffectLab, Fujie

from torch.utils.data import Dataset, ConcatDataset, DataLoader
from torch.nn.utils.rnn import pack_sequence, pad_packed_sequence, pack_padded_sequence
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import tqdm
import numpy as np

torch.multiprocessing.set_start_method('spawn', force=True)


class DurationInfoDataset(Dataset):
    def __init__(self, duration_info_manager: DurationInfoManager, id_list):
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


def extract_label_from_duration_info(info: DurationInfo, ch, framerate):
    """
    chはチャネル
    framerateは，フレームレート（fps）
    """
    values_list = []
    # 1フレームあたりの時間[s]
    df_s = 1.0 / framerate
    # 総フレーム数
    total_num_frames = int((len(info.wav[ch]) / 16000.0) * framerate)
    
    # TURN
    values = []
    for ts, te, tag in info.turn[ch]:
        # 一つ前のデータまで書き出されたデータ数から
        # 最初の時刻を計算
        prev_te = len(values) / framerate * 1000.0
        num_frames = int(np.ceil((ts - prev_te) / 1000.0 * framerate))
        values.extend((-(np.arange(num_frames) + 1) * df_s).tolist())
        prev_te = len(values) / framerate * 1000.0
        num_frames = int(np.ceil((te - prev_te) / 1000.0 * framerate))
        # value = 1
        # if tag == 'S':
        #     value = 2
        # values.extend([value] * num_frames)
        values.extend(((np.arange(num_frames)[::-1]) * df_s).tolist())
    if len(values) < total_num_frames:
        num_frames = total_num_frames - len(values)
        values.extend((-(np.arange(num_frames) + 1) * df_s).tolist())
    if len(values) > total_num_frames:
        values = values[:total_num_frames]        
    values_list.append(values)

    # VAD
    values = []
    for ts, te in info.vad[ch]:
        # 一つ前のデータまで書き出されたデータ数から
        # 最初の時刻を計算
        prev_te = len(values) / framerate * 1000.0
        num_frames = int(np.ceil((ts - prev_te) / 1000.0 * framerate))
        values.extend([0] * num_frames)
        prev_te = len(values) / framerate * 1000.0
        num_frames = int(np.ceil((te - prev_te) / 1000.0 * framerate))
        values.extend([1] * num_frames)
    if len(values) < total_num_frames:
        num_frames = total_num_frames - len(values)
        values.extend([0] * num_frames)
    if len(values) > total_num_frames:
        values = values[:total_num_frames]
    values_list.append(values)

    # lens = [len(values) for values in values_list]
    # max_len = max(lens)
    # result = np.zeros((2, max_len), dtype=np.float32)
    # mask = np.arange(max_len) < np.array(lens)[:, None]
    # result[mask] = np.concatenate(values_list)

    # return torch.tensor(result.T)

    result = torch.tensor(values_list).t()
    # import ipdb; ipdb.set_trace()
    return result


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
            # import ipdb; ipdb.set_trace()
            wavs, targets = self._collate(duration_info_list, ch)
            wav_list.extend(wavs)
            target_list.extend(targets)
            
        for duration_info in duration_info_list:
            duration_info.clear_cache()

        return wav_list, target_list

    def _collate(self, duration_info_list, ch):
        wav_list = [info.wav[ch] for info in duration_info_list]
        # if self._noise_adder is not None and np.random.randint(2) > 0:
        #   wav_list = [self._noise_adder.add_noise(wav) for wav in wav_list]
        extended = False
        if self._noise_adder is not None or \
           self._interm_noise_adder is not None:
            if self._noise_adder is not None:
                noised_wav_list = [
                    self._noise_adder.add_noise(wav) for wav in wav_list
                ]
            if self._interm_noise_adder is not None:
                noised_wav_list = [
                    self._interm_noise_adder.add_noise(wav) for wav in noised_wav_list
                ]
            wav_list.extend(noised_wav_list)
            extended = True

        # DurationInfoから正解ラベルを生成する
        # 正解ラベルは，TURN, UTTERNCE, VAD の三つ組の整数
        # TURN は 0 (OFF), 1 (TURN), 2 (SHORT UTTERANCE) のいずれか
        # UTTERANCE, VAD は 0 (OFF) か 1 (ON) のいずれか
        labels_list = [
            extract_label_from_duration_info(
                info, ch, self._frame_rate)
            for info in duration_info_list
        ]
        if extended:
            labels_list = labels_list * 2
            target = pack_sequence(labels_list, enforce_sorted=False)
            # import ipdb; ipdb.set_trace()
        return wav_list, labels_list


class TurnUtteranceVadCrossEntropyLoss(nn.Module):
    def __init__(self,
                 weight_turn=1.0,
                 weight_alpha=0.1,
                 weight_vad=0.5,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._turn_loss = nn.MSELoss(reduction='sum')
        self._vad_loss = nn.BCELoss(reduction='sum')
        self._weight_turn = weight_turn
        self._weight_alpha = weight_alpha
        self._weight_vad = weight_vad

    def forward(self, y, t, y_len):
        if t.shape[0] > y.shape[0]:
            t = t[:y.shape[0]]
            
        y_dt = y[:, :, 0]
        y_alpha = y[:, :, 1]
        y_vad = y[:, :, 2]

        t_dt = t[:, :, 0]
        t_vad = t[:, :, 1]

        # y_alpha = torch.exp(-y_alpha)
        # y_dt = torch.tanh(y_dt)
        y_alpha = 0.9 * torch.sigmoid(-y_alpha) + 0.1
        t_dt = torch.tanh(y_alpha * t_dt)

        y_vad = torch.sigmoid(y_vad)
        
        total_loss = 0.0
        _, batch, _ = y.shape
        total_length = 0
        for i in range(batch):
            if y_len[i] == 0:
                continue
            total_loss += self._weight_turn * self._turn_loss(
                y_dt[:y_len[i], i], t_dt[:y_len[i], i])
            total_loss += self._weight_alpha * torch.sum(y_alpha[:y_len[i], i])
            total_loss += self._weight_vad * self._vad_loss(
                y_vad[:y_len[i], i], t_vad[:y_len[i], i])
            total_length += y_len[i]
        # import ipdb; ipdb.set_trace()
        if total_length > 0.0:
            total_loss = total_loss / total_length
        return total_loss


class TurnDetectorTrainer0025(TurnDetectorTrainer):
    def __init__(self, backprop_len=3000):
        super().__init__()
        self._backprop_len = backprop_len

    def build_torch_trainer(self, turn_detector: TurnDetector):
        criterion = TurnUtteranceVadCrossEntropyLoss()
        criterion = criterion.to(turn_detector.device)
        optimizer = optim.Adam(turn_detector.torch_model.parameters())

        waseda_soma = WASEDA_SOMA()
        dim = DurationInfoManagerWasedaSoma(waseda_soma)
        id_list_train = waseda_soma.get_id_list()[:300]
        dataset_train = DurationInfoDataset(dim, id_list_train)

        id_list_vali = waseda_soma.get_id_list()[300:310]
        dataset_vali = DurationInfoDataset(dim, id_list_vali)

        jeida = JEIDA()
        noise_adder = NoiseAdder(jeida.get_wav_path_list())

        soundffectlab = SoundffectLab()
        fujie = Fujie()
        interm_noise_adder = IntermittentNoiseAdder(
            soundffectlab.get_wav_path_list() + fujie.get_wav_path_list())

        collate_fn_trainer = CollateDurationInfo(
            turn_detector.feature_extractor.feature_rate,
            noise_adder, interm_noise_adder)
        collate_fn_valid = CollateDurationInfo(
            turn_detector.feature_extractor.feature_rate)

        train_loader = DataLoader(
            dataset_train,
            batch_size=3,
            collate_fn=collate_fn_trainer,
            num_workers=0,
            shuffle=True,
        )
        vali_loader = DataLoader(
            dataset_vali,
            batch_size=1,
            collate_fn=collate_fn_valid,
            num_workers=0,
            shuffle=False,
        )
        callbacks = [
            ClippingGrad(1.),
            StandardReporter(train_report_interval=1,
                             validation_report_interval=1),
            CsvWriterReporter(turn_detector.get_csv_log_filename()),
            EarlyStopper(patience=3, verbose=True),
        ]

        trainer = TorchTrainerForTurnDetector(
            turn_detector,
            criterion,
            optimizer,
            train_loader,
            vali_loader,
            callbacks=callbacks,
            device=turn_detector.device,
            epoch=20,
            backprop_len=self._backprop_len
        )

        return trainer
