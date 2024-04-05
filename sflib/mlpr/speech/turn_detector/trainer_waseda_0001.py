from .base import TorchTrainerForTurnDetector, TurnDetectorTrainer, \
    TurnDetector, TurnDetectorFeatureExtractor
from ....ext.torch.callbacks.reporters \
    import StandardReporter, CsvWriterReporter
from ....ext.torch.callbacks.early_stopper import EarlyStopper
from ....ext.torch.callbacks.snapshot import Snapshot
from ....ext.torch.callbacks.train import ClippingGrad

from ....corpus.speech.duration import DurationInfoManager, DurationInfo
from ....corpus.speech.waseda_soma import WASEDA_SOMA, DurationInfoManagerWasedaSoma
from sflib.sound.sigproc.noise import NoiseAdder
from sflib.corpus.noise import JEIDA

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
    def __init__(self,
                 feature_extractor: TurnDetectorFeatureExtractor,
                 noise_adder: NoiseAdder = None):
        self._feature_extractor = feature_extractor
        self._noise_adder = noise_adder

    def __call__(self, duration_info_list):
        feat_list = []
        feat_len_list = []
        target_list = []
        for ch in range(2):
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
            feat_list.append(fp)
            feat_len_list.append(fl)
            target_list.append(tp)

        for duration_info in duration_info_list:
            duration_info.clear_cache()

        padded_feat = torch.cat(feat_list, dim=1)
        padded_feat_len = torch.cat(feat_len_list)
        padded_target = torch.cat(target_list, dim=1)

        feat = pack_padded_sequence(
            padded_feat, padded_feat_len, enforce_sorted=False)
        target = pack_padded_sequence(
            padded_target, padded_feat_len, enforce_sorted=False)

        return feat, target, np.random.randint(10000)

    def _collate(self, duration_info_list, ch):
        wav_list = [info.wav[ch] for info in duration_info_list]
        if self._noise_adder is not None and np.random.randint(2) > 0:
            wav_list = [self._noise_adder.add_noise(wav) for wav in wav_list]
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


class TurnDetectorTrainerWaseda0001(TurnDetectorTrainer):
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

        collate_fn_trainer = CollateDurationInfo(
            turn_detector.feature_extractor, noise_adder)
        collate_fn_valid = CollateDurationInfo(turn_detector.feature_extractor)

        train_loader = DataLoader(
            dataset_train,
            batch_size=1,
            collate_fn=collate_fn_trainer,
            num_workers=2,
            # num_workers=0,
            shuffle=False,
        )
        vali_loader = DataLoader(
            dataset_vali,
            batch_size=1,
            collate_fn=collate_fn_valid,
            num_workers=2,
            # num_workers=0,
            shuffle=False,
        )
        callbacks = [
            ClippingGrad(1.),
            StandardReporter(
                train_report_interval=1, validation_report_interval=1),
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


def construct(device=None):
    from .feature_extractor_0001 \
        import TurnDetectorFeatureExtractor0001 as FeatureExtractor
    from .turn_detector_0001 \
        import TurnDetector0001 as TurnDetector
    feature_extractor = FeatureExtractor(
        12, 'csj_0006', 'CSJ0006', device=device)
    turn_detector = TurnDetector(feature_extractor, device=device)
    trainer = TurnDetectorTrainerWaseda0001(turn_detector)
    return trainer


def construct0002(device=None):
    from .feature_extractor_0002 \
        import TurnDetectorFeatureExtractor0002 as FeatureExtractor
    from .turn_detector_0001 \
        import TurnDetector0001 as TurnDetector
    feature_extractor = FeatureExtractor(
        12,
        'csj_0006',
        'CSJ0006',
        2,
        5,
        'csj_rwcp_0003',
        'CSJRWCP0003',
        device=device)
    turn_detector = TurnDetector(feature_extractor, device=device)
    trainer = TurnDetectorTrainerWaseda0001(turn_detector)
    return trainer


def train(device=None):
    trainer = construct(device)
    trainer.train()

    from . import base as b
    b.save_turn_detector(trainer, upload=True)
    trainer.upload_csv_log()


def train0002(device=None):
    trainer = construct0002(device)
    trainer.train()

    from . import base as b
    b.save_turn_detector(trainer, upload=True)
    trainer.upload_csv_log()


# -----
def test():
    # ---- collateのテスト ここから ----
    from .feature_extractor_0001 import TurnDetectorFeatureExtractor0001 as FE
    fe = FE(12, 'csj_0006', 'CSJ0006')

    from ....corpus.speech.waseda_soma \
        import WASEDA_SOMA, DurationInfoManagerWasedaSoma
    ws = WASEDA_SOMA()
    dim = DurationInfoManagerWasedaSoma(ws)
    id_list = ws.get_id_list()
    duration_info_list = [dim.get_duration_info(id_list[i]) for i in range(1)]

    collate_fn = CollateDurationInfo()
    collate_fn.set_feature_extractor(fe)
    feat, feat_length, target = collate_fn(duration_info_list)
    print(feat.shape)
    print(feat_length.shape)
    print(target.shape)

    return feat, feat_length, target
    # ---- ここまで ----
