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


class DummyDurationInfo:
    """Turn情報ごとに切り出したダミーの区間情報
    """

    def __init__(self, wav, vad_info, utterance_info, turn_info):
        self._wav = wav
        self._vad_info = vad_info
        self._utterance_info = utterance_info
        self._turn_info = turn_info

    @property
    def wav_filename(self):
        return ''

    @property
    def wav(self):
        return self._wav

    @property
    def vad(self):
        return self._vad_info

    @property
    def utterance(self):
        return self._utterance_info

    @property
    def turn(self):
        return self._turn_info

    def clear_cache(self):
        pass

    def to_eaf(self):
        pass

    def __repr__(self):
        return "DummyDurationInfo(VAD=(%d,), UTTERANCE=(%d,), TURN=(%d,))" \
            % (len(self.vad[0]), len(self.utterance[0]), len(self.turn[0]), )


class DurationInfoDataset(Dataset):
    def __init__(self,
                 duration_info_manager: DurationInfoManager,
                 id_list,
                 margin_ms=5000,
                 max_duration=60*1000):
        self._info_manager = duration_info_manager
        self._id_list = id_list
        self._margin_ms = margin_ms

        self._duration_info_list = []
        channel_index = []
        num_turns = []
        for id in tqdm.tqdm(self._id_list):
            di = self._info_manager.get_duration_info(id)
            self._duration_info_list.append(di)
            for ch in (0, 1):
                channel_index.append(ch)
                num_turns.append(len(di.turn[ch]))
        cumsum_index = np.cumsum(num_turns)
        self._len = cumsum_index[-1]
        self._channel_index = channel_index
        self._cumsum_index = np.concatenate([[0], cumsum_index[:-1]])
        # import ipdb; ipdb.set_trace() 

        self._last_accessed_i = 0
        self._max_duration = max_duration

    def __len__(self):
        # return len(self._duration_info_list)
        return self._len

    def __getitem__(self, i):
        # return self._duration_info_list[i]
        # info_i ... 対象とする区間情報
        # info_j ... 対象とするターンのインデクス
        info_i = np.where(self._cumsum_index <= i)[0][-1]
        ch = self._channel_index[info_i]
        info_j = i - self._cumsum_index[info_i]
        
        info_i = info_i // 2
        di = self._duration_info_list[info_i]
        tj = di.turn[ch][info_j]
        wav = di.wav[ch, :]
        wav_len_ms = len(wav) / 16

        start_ms, end_ms, _ = tj
        start_ms -= self._margin_ms
        end_ms += self._margin_ms
        if start_ms < 0:
            start_ms = 0
        if end_ms > wav_len_ms:
            end_ms = wav_len_ms
        duration_ms = end_ms - start_ms
        if duration_ms > self._max_duration:
            duration_ms = self._max_duration
            end_ms = start_ms + self._max_duration
        # if duration_ms > self._max_duration:
        #     self._max_duration = duration_ms
        #     print("MAX DURATION={:.3f}".format(self._max_duration / 1000.0))
        
        dummy_wav = wav[int(start_ms * 16):int(end_ms * 16)].reshape(1, -1)
        dummy_turn_info = []
        for s, e, t in di.turn[ch]:
            if e >= start_ms and s <= end_ms:
                s = s - start_ms
                e = e - start_ms
                if s < 0:
                    s = 0
                if e > duration_ms:
                    e = duration_ms
                dummy_turn_info.append([s, e, t])
        dummy_turn_info = [dummy_turn_info]
        dummy_utterance_info = []
        for s, e in di.utterance[ch]:
            if e >= start_ms and s <= end_ms:
                s = s - start_ms
                e = e - start_ms
                if s < 0:
                    s = 0
                if e > duration_ms:
                    e = duration_ms
                dummy_utterance_info.append([s, e])
        dummy_utterance_info = [dummy_utterance_info]
        dummy_vad_info = []
        for s, e in di.vad[ch]:
            if e >= start_ms and s <= end_ms:
                s = s - start_ms
                e = e - start_ms
                if s < 0:
                    s = 0
                if e > duration_ms:
                    e = duration_ms
                dummy_vad_info.append([s, e])
        dummy_vad_info = [dummy_vad_info]
        result = DummyDurationInfo(dummy_wav, dummy_vad_info,
                                   dummy_utterance_info, dummy_turn_info)
        
        # 最後にアクセスしたiと今回のiが違った場合は前のキャッシュをクリアする
        if info_i != self._last_accessed_i:
            self._duration_info_list[self._last_accessed_i].clear_cache()
        self._last_accessed_i = info_i

        return result


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
    def __init__(self,
                 feature_extractor: TurnDetectorFeatureExtractor,
                 noise_adder: NoiseAdder = None,
                 interm_noise_adder: IntermittentNoiseAdder = None):
        self._feature_extractor = feature_extractor
        self._noise_adder = noise_adder
        self._interm_noise_adder = interm_noise_adder

    def __call__(self, duration_info_list):
        feat_list = []
        feat_len_list = []
        target_list = []
        for ch in range(1):
            self._feature_extractor.reset()
            f, t = self._collate(duration_info_list, ch)
            fp, fl = pad_packed_sequence(f)
            tp, tl = pad_packed_sequence(t)
            # import ipdb; ipdb.set_trace()
            # fpの長さにtpを合わせる（tpの方が長い場合も短い場合もある）
            res_len = fp.shape[0] - tp.shape[0]
            if res_len > 0:
                tp = F.pad(tp, (0, 0, 0, 0, 0, res_len))
            elif res_len < 0:
                tp = tp[:res_len]
            if fp.shape[1] > 1:
                tp = torch.cat([tp] * fp.shape[1], dim=1)
            feat_list.append(fp)
            feat_len_list.append(fl)
            target_list.append(tp)
            # import ipdb; ipdb.set_trace()

        for duration_info in duration_info_list:
            duration_info.clear_cache()

        padded_feat = torch.cat(feat_list, dim=1)
        padded_feat_len = torch.cat(feat_len_list)
        padded_target = torch.cat(target_list, dim=1)

        feat = pack_padded_sequence(padded_feat,
                                    padded_feat_len,
                                    enforce_sorted=False)
        target = pack_padded_sequence(padded_target,
                                      padded_feat_len,
                                      enforce_sorted=False)

        return feat, target, np.random.randint(10000)

    def _collate(self, duration_info_list, ch):
        wav_list = [info.wav[ch] for info in duration_info_list]
        # if self._noise_adder is not None and np.random.randint(2) > 0:
        #   wav_list = [self._noise_adder.add_noise(wav) for wav in wav_list]
        if self._noise_adder is not None or \
           self._interm_noise_adder is not None:
            if self._noise_adder is not None:
                noised_wav_list = [
                    self._noise_adder.add_noise(wav) for wav in wav_list
                ]
            else:
                noised_wav_list = [wav.copy() for wav in wav_list]
            if self._interm_noise_adder is not None:
                noised_wav_list = [
                    self._interm_noise_adder.add_noise(wav)
                    for wav in noised_wav_list
                ]
            wav_list = wav_list + noised_wav_list
        feat = self._feature_extractor.calc(wav_list)

        # DurationInfoから正解ラベルを生成する
        # 正解ラベルは，TURN, UTTERNCE, VAD の三つ組の整数
        # TURN は 0 (OFF), 1 (TURN), 2 (SHORT UTTERANCE) のいずれか
        # UTTERANCE, VAD は 0 (OFF) か 1 (ON) のいずれか
        labels_list = [
            extract_label_from_duration_info(
                info, ch, self._feature_extractor.feature_rate)
            for info in duration_info_list
        ]
        target = pack_sequence(labels_list, enforce_sorted=False)
        # import ipdb; ipdb.set_trace()
        return (feat, target)


class TurnUtteranceVadCrossEntropyLoss(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._turn_loss = nn.CrossEntropyLoss()
        self._utterance_loss = nn.CrossEntropyLoss()
        self._vad_loss = nn.CrossEntropyLoss()

    def forward(self, y, t, y_len):
        total_loss = 0
        _, batch, _ = y.shape
        count = 0
        for i in range(batch):
            if y_len[i] == 0:
                continue
            y_turn = y[:y_len[i], i, 0:3]
            t_turn = t[:y_len[i], i, 0]
            total_loss += 0.6 * self._turn_loss(y_turn, t_turn)
            y_utt = y[:y_len[i], i, 3:5]
            t_utt = t[:y_len[i], i, 1]
            total_loss += 0.1 * self._utterance_loss(y_utt, t_utt)
            y_vad = y[:y_len[i], i, 5:7]
            t_vad = t[:y_len[i], i, 2]
            total_loss += 0.3 * self._vad_loss(y_vad, t_vad)
            count += 1
        if count > 0:
            total_loss = total_loss / count
        return total_loss


class TurnDetectorTrainerWaseda0005(TurnDetectorTrainer):
    def __init__(self, turn_detector: TurnDetector):
        super().__init__(turn_detector)

    def build_torch_trainer(self, turn_detector: TurnDetector):
        criterion = TurnUtteranceVadCrossEntropyLoss()
        if turn_detector.device is not None:
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
            turn_detector.feature_extractor, noise_adder, interm_noise_adder)
        collate_fn_valid = CollateDurationInfo(turn_detector.feature_extractor)

        train_loader = DataLoader(
            dataset_train,
            batch_size=10,
            # batch_size=10,
            collate_fn=collate_fn_trainer,
            # num_workers=8,
            num_workers=4,
            # num_workers=1,
            # num_workers=0,
            shuffle=False,
        )
        vali_loader = DataLoader(
            dataset_vali,
            # batch_size=1,
            batch_size=10,
            collate_fn=collate_fn_valid,
            # num_workers=8,
            num_workers=4,
            # num_workers=1,
            shuffle=False,
        )
        callbacks = [
            ClippingGrad(1.),
            StandardReporter(train_report_interval=10,
                             validation_report_interval=10),
            CsvWriterReporter(self.get_csv_log_filename()),
            Snapshot(final_filename=self.get_model_filename()),
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
        )

        return trainer
