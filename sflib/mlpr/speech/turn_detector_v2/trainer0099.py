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

    # TURN
    values = []
    for ts, te, tag in info.turn[ch]:
        # 一つ前のデータまで書き出されたデータ数から
        # 最初の時刻を計算
        prev_te = len(values) / framerate * 1000.0
        num_frames = int(np.ceil((ts - prev_te) / 1000.0 * framerate))
        values.extend([0] * num_frames)
        prev_te = len(values) / framerate * 1000.0
        num_frames = int(np.ceil((te - prev_te) / 1000.0 * framerate))
        value = 1
        if tag == 'S':
            value = 2
        values.extend([value] * num_frames)
    values_list.append(values)

    # UTTERANCE
    values = []
    for ts, te in info.utterance[ch]:
        # 一つ前のデータまで書き出されたデータ数から
        # 最初の時刻を計算
        prev_te = len(values) / framerate * 1000.0
        num_frames = int(np.ceil((ts - prev_te) / 1000.0 * framerate))
        values.extend([0] * num_frames)
        prev_te = len(values) / framerate * 1000.0
        num_frames = int(np.ceil((te - prev_te) / 1000.0 * framerate))
        values.extend([1] * num_frames)
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
    values_list.append(values)

    lens = [len(values) for values in values_list]
    max_len = max(lens)
    result = np.zeros((3, max_len), dtype=np.int64)
    mask = np.arange(max_len) < np.array(lens)[:, None]
    result[mask] = np.concatenate(values_list)
    return torch.tensor(result.T)


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

        offset = float(np.random.rand(1) * 10) # 最大10秒のオフセットをつける
        offset_samples = int(offset * 16000)
        for i in range(len(wav_list)):
            wav_list[i] = wav_list[i][offset_samples:]

        offset_frames = int(offset * self._frame_rate)
            
        # DurationInfoから正解ラベルを生成する
        # 正解ラベルは，TURN, UTTERNCE, VAD の三つ組の整数
        # TURN は 0 (OFF), 1 (TURN), 2 (SHORT UTTERANCE) のいずれか
        # UTTERANCE, VAD は 0 (OFF) か 1 (ON) のいずれか
        labels_list = [
            extract_label_from_duration_info(
                info, ch, self._frame_rate)[offset_frames:]
            for info in duration_info_list
        ]
        if extended:
            labels_list = labels_list * 2
        target = pack_sequence(labels_list, enforce_sorted=False)
        # import ipdb; ipdb.set_trace()
        return wav_list, labels_list


class TurnUtteranceVadCrossEntropyLoss(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._turn_loss = nn.CrossEntropyLoss()
        self._utterance_loss = nn.CrossEntropyLoss()
        self._vad_loss = nn.CrossEntropyLoss()

    def forward(self, y, t, y_len):
        total_loss = 0.0
        _, batch, _ = y.shape
        for i in range(batch):
            if y_len[i] == 0:
                continue
            y_turn = y[:y_len[i], i, 0:3]
            t_turn = t[:y_len[i], i, 0]
            total_loss += self._turn_loss(y_turn, t_turn)
            y_utt = y[:y_len[i], i, 3:5]
            t_utt = t[:y_len[i], i, 1]
            total_loss += self._utterance_loss(y_utt, t_utt)
            y_vad = y[:y_len[i], i, 5:7]
            t_vad = t[:y_len[i], i, 2]
            total_loss += self._vad_loss(y_vad, t_vad)
        total_loss = total_loss / batch
        return total_loss


class TurnDetectorTrainer0099(TurnDetectorTrainer):
    def __init__(self):
        super().__init__()

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
            backprop_len=5000, # 5,000 * 20 = 100,000ms = 100秒
        )

        return trainer
