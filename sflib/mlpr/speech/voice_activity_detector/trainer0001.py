from .base import TorchTrainerForVoiceActivityDetector, \
    VoiceActivityDetectorTrainer, \
    VoiceActivityDetector, VoiceActivityDetectorFeatureExtractor
from ....ext.torch.callbacks.reporters \
    import StandardReporter, CsvWriterReporter
from ....ext.torch.callbacks.early_stopper import EarlyStopper
from ....ext.torch.callbacks.snapshot import Snapshot
from ....ext.torch.callbacks.train import ClippingGrad

from ....corpus.speech.duration_v2 \
    import DurationInfoV2Manager, DurationInfoV2
from ....corpus.speech.waseda_soma \
    import WASEDA_SOMA, DurationInfoV2ManagerWasedaSoma
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


def extract_label_from_duration_info(info: DurationInfoV2, ch, framerate):
    """
    chはチャネル
    framerateは，フレームレート（fps）
    """
    values_list = []

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
    result = np.zeros((1, max_len), dtype=np.int64)
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
            # target = pack_sequence(labels_list, enforce_sorted=False)
            # import ipdb; ipdb.set_trace()
        return wav_list, labels_list


class VadCrossEntropyLoss(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._vad_loss = nn.CrossEntropyLoss()

    def forward(self, y, t, y_len):
        total_loss = 0.0
        _, batch, _ = y.shape
        for i in range(batch):
            if y_len[i] == 0:
                continue
            y_vad = y[:y_len[i], i, 0:2]
            t_vad = t[:y_len[i], i, 0]
            total_loss += self._vad_loss(y_vad, t_vad)
        total_loss = total_loss / batch
        return total_loss


class VoiceActivityDetectorTrainer0001(VoiceActivityDetectorTrainer):
    def __init__(self, backprop_len=3000):
        super().__init__()
        self._backprop_len = backprop_len

    def build_torch_trainer(self,
                            voice_activity_detector: VoiceActivityDetector):
        criterion = VadCrossEntropyLoss()
        criterion = criterion.to(voice_activity_detector.device)
        optimizer = optim.Adam(
            voice_activity_detector.torch_model.parameters())

        waseda_soma = WASEDA_SOMA()
        dim = DurationInfoV2ManagerWasedaSoma(waseda_soma)
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
            voice_activity_detector.feature_extractor.feature_rate,
            noise_adder, interm_noise_adder)
        collate_fn_valid = CollateDurationInfo(
            voice_activity_detector.feature_extractor.feature_rate)

        train_loader = DataLoader(
            dataset_train,
            # batch_size=3,
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
            CsvWriterReporter(voice_activity_detector.get_csv_log_filename()),
            # Snapshot(final_filename=turn_detector.get_model_filename()),
            EarlyStopper(patience=3, verbose=True),
        ]

        trainer = TorchTrainerForVoiceActivityDetector(
            voice_activity_detector,
            criterion,
            optimizer,
            train_loader,
            vali_loader,
            callbacks=callbacks,
            device=voice_activity_detector.device,
            epoch=20,
            backprop_len=self._backprop_len
        )

        return trainer
