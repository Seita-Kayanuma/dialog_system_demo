# 2019.10.22 実験用
#  - 10グループに分割し，10分割交差検証をする．
#  - バックプロパゲーションの長さを変化させながら実験をする
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pack_padded_sequence

from os import path
import os

from .... import config
from .base import FacialTurnDetectorTrainer
from .base import FacialTurnDetector
from .base import TorchTrainerForFacialTurnDetector
from .dataset import CachedDataset
from ....corpus.video.rinus import Rinus
from ....ext.torch.callbacks.reporters \
    import StandardReporter, CsvWriterReporter
from ....ext.torch.callbacks.early_stopper import EarlyStopper
from ....ext.torch.callbacks.snapshot import Snapshot
from ....ext.torch.callbacks.train import ClippingGrad

# 各順番の人がどのグループに属するか．
# グループ番号は1〜10なのに注意
group_list = [
    10, 5, 5, 1, 3, 2, 3, 2, 6, 4, 10, 1, 2, 7, 5, 6, 7, 5, 6, 7, 8, 8, 8, 7,
    6, 4, 9, 10, 9, 10, 5, 3, 8, 9, 1, 8, 2, 3, 1, 2, 4, 10, 5, 3, 4, 4, 5, 5,
    6, 6, 9, 2, 7, 1, 6, 3, 2, 8, 6, 7, 9, 8, 7, 10, 3, 4, 1, 5, 10, 8, 2, 9,
    10, 1, 9, 1, 4, 6, 3, 5, 7, 7, 8, 1, 7, 8, 2, 9, 3, 9, 9, 10, 10, 3, 4, 4,
    1, 6, 2, 4
]


def generate_group2index():
    group2index = {}
    for group in list(set(group_list)):
        index_list = [
            i for i in range(len(group_list)) if group_list[i] == group
        ]
        group2index[group] = index_list
    return group2index


# グループ番号（１〜10）から，そのグループに所属するデータのインデクスリストに
# 変換するディクショナリ
group2index = generate_group2index()


def generate_cross_validataion_list(n=list(range(1, 11))):
    result = []
    for i in range(len(n)):
        j = (i + 1) % len(n)
        valid = n[i]
        test = n[j]
        train = [x for x in n if x != valid and x != test]
        result.append([train, valid, test])
    return result


def collate_fn(feat_target_list, apply_offset=True):
    import torch
    feat_list = []
    target_list = []
    for f, t in feat_target_list:
        if apply_offset:
            # import ipdb; ipdb.set_trace()
            # 最大10秒程度のオフセットを設ける
            # （ある程度ずらさないと毎回同じタイミングで学習が行われるため）
            offset = torch.randint(0, 300, (1, ))[0]
            f = f[offset:, :]
            t = t[offset:]
        feat_list.append(f)
        target_list.append(t.reshape(-1, 1))
    return feat_list, target_list


def collate_fn_train(feat_target_list):
    return collate_fn(feat_target_list)


def collate_fn_validation(feat_target_list):
    return collate_fn(feat_target_list, apply_offset=False)


class CrossEntropyLossForLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.__cross_entropy_loss = nn.CrossEntropyLoss(reduction='mean')

    def __call__(self, y, t, length):
        y_ = y[:, length != 0, :]
        t_ = t[:, length != 0, :]
        length_ = length[length != 0]
        y_packed = pack_padded_sequence(y_, length_, enforce_sorted=False)
        t_packed = pack_padded_sequence(t_, length_, enforce_sorted=False)
        # import ipdb; ipdb.set_trace()
        return self.__cross_entropy_loss(y_packed.data,
                                         t_packed.data.reshape(-1, ))


class FacialTurnDetectorTrainerRinus0004(FacialTurnDetectorTrainer):
    def __init__(self, turn_detector: FacialTurnDetector, train_groups,
                 validation_group, test_group, backprop_len):
        super().__init__(turn_detector)
        train_indices = []
        for group in train_groups:
            train_indices.extend(group2index[group])
            self._train_indices = train_indices
            self._validation_indices = group2index[validation_group]
            self._test_indices = group2index[test_group]
            self._test_group = test_group
            self._backprop_len = backprop_len

    def get_filename_base(self):
        return super().get_filename_base() + "B{:04d}".format(
            self._backprop_len)

    def get_model_filename(self):
        body, _ = path.splitext(super().get_model_filename())
        fullpath = "{}+{:02d}".format(body, self._test_group)
        return fullpath

    def get_csv_log_filename(self):
        body, _ = path.splitext(super().get_csv_log_filename())
        fullpath = "{}+{:02d}.csv".format(body, self._test_group)
        return fullpath

    def build_torch_trainer(self, turn_detector: FacialTurnDetector):
        criterion = CrossEntropyLossForLSTM()
        if turn_detector.device is not None:
            criterion = criterion.to(turn_detector.device)

        optimizer = optim.Adam(turn_detector.torch_model.parameters(),
                               weight_decay=0.01)

        rinus = Rinus()
        rinus_id_list = rinus.get_id_list()
        train_id_list = [rinus_id_list[i] for i in self._train_indices]
        valid_id_list = [rinus_id_list[i] for i in self._validation_indices]
        # print("---- TRAIN ----")
        # print(train_id_list)
        # print("---- VALID ----")
        # print(valid_id_list)
        train_list = [(rinus.get_mp4_path(id),
                       rinus.get_eaf_kobayashi_path(id))
                      for id in train_id_list]
        valid_list = [(rinus.get_mp4_path(id),
                       rinus.get_eaf_kobayashi_path(id))
                      for id in valid_id_list]
        train_ds = CachedDataset(self.facial_turn_detector.feature_extractor,
                                 train_list)
        valid_ds = CachedDataset(self.facial_turn_detector.feature_extractor,
                                 valid_list)
        train_loader = DataLoader(train_ds,
                                  batch_size=10,
                                  collate_fn=collate_fn_train,
                                  num_workers=0,
                                  shuffle=True)
        valid_loader = DataLoader(valid_ds,
                                  batch_size=1,
                                  collate_fn=collate_fn_validation,
                                  num_workers=0,
                                  shuffle=False)
        callbacks = [
            ClippingGrad(1.),
            StandardReporter(train_report_interval=1,
                             validation_report_interval=1),
            CsvWriterReporter(self.get_csv_log_filename()),
            Snapshot(final_filename=self.get_model_filename()),
            EarlyStopper(patience=10, verbose=True),
        ]
        trainer = TorchTrainerForFacialTurnDetector(
            self.facial_turn_detector,
            criterion,
            optimizer,
            train_loader,
            valid_loader,
            callbacks=callbacks,
            device=self.facial_turn_detector.device,
            epoch=100,
            backprop_len=self._backprop_len)

        return trainer


def get_test_outdir(trainer: FacialTurnDetectorTrainer):
    detector = trainer.facial_turn_detector
    feature_extractor = detector.feature_extractor
    outdir = "{}+{}+{}+TEST".format(detector.get_filename_base(),
                                    feature_extractor.get_filename_base(),
                                    trainer.get_filename_base())
    outdir = path.join(config.get_package_data_dir(__package__), outdir)
    return outdir


def test(trainer: FacialTurnDetectorTrainer, test_group, device):
    detector = trainer.facial_turn_detector
    feature_extractor = detector.feature_extractor

    test_indices = group2index[test_group]
    rinus = Rinus()
    rinus_id_list = rinus.get_id_list()
    test_id_list = [rinus_id_list[i] for i in test_indices]
    test_list = [(rinus.get_mp4_path(id), rinus.get_eaf_kobayashi_path(id))
                 for id in test_id_list]
    test_ds = CachedDataset(feature_extractor, test_list)

    outdir = get_test_outdir(trainer)

    if not path.exists(outdir):
        os.makedirs(outdir, mode=0o755, exist_ok=True)

    for data, id in zip(test_ds, test_id_list):
        print(id)
        # import ipdb; ipdb.set_trace()
        feat = data[0]
        feat = feat.unsqueeze(1)
        detector.reset()
        detector.torch_model.eval()
        x = detector.forward_core(feat.to(device))
        x = torch.softmax(x, 2)
        score = x.detach().cpu().squeeze(1).numpy()[:, 1]
        result = x.squeeze(1).argmax(dim=1).cpu().numpy()
        target = data[1].cpu().numpy()
        import numpy as np
        import pandas as pd
        df = pd.DataFrame(np.stack([target, result, score]).T,
                          columns=['target', 'result', 'score'])
        outpath = path.join(outdir, "{}.csv".format(id))
        df.to_csv(outpath)


def calc_metrics(trainer: FacialTurnDetectorTrainer):
    import glob
    import numpy as np
    import pandas as pd
    import sklearn.metrics as m
    dir = get_test_outdir(trainer)
    outfile = dir + "_summary.csv"
    y_true_all = []
    y_pred_all = []
    id_list = []
    prec_list = []
    rec_list = []
    f1_list = []
    p_list = []
    acc_list = []
    for csv_fullpath in sorted(glob.glob(path.join(dir, "*.csv"))):
        id, _ = path.splitext(path.basename(csv_fullpath))
        df = pd.read_csv(csv_fullpath, index_col=0)
        y_true = df['target'].tolist()
        y_pred = df['result'].tolist()
        score_prec = m.precision_score(y_true, y_pred, pos_label=1)
        score_rec = m.recall_score(y_true, y_pred, pos_label=1)
        score_f1 = m.f1_score(y_true, y_pred, pos_label=1)
        pos_rate = df['target'].sum() / len(y_true)
        score_acc = m.accuracy_score(y_true, y_pred)
        print("{:03d} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f}".format(
            int(id), score_prec, score_rec, score_f1, pos_rate, score_acc))
        id_list.append(id)
        prec_list.append(score_prec)
        rec_list.append(score_rec)
        f1_list.append(score_f1)
        p_list.append(pos_rate)
        acc_list.append(score_acc)
        y_true_all.extend(y_true)
        y_pred_all.extend(y_pred)
    score_prec = m.precision_score(y_true_all, y_pred_all, pos_label=1)
    score_rec = m.recall_score(y_true_all, y_pred_all, pos_label=1)
    score_f1 = m.f1_score(y_true_all, y_pred_all, pos_label=1)
    pos_rate = np.array(y_true_all).sum() / len(y_true_all)
    score_acc = m.accuracy_score(y_true_all, y_pred_all)
    print("ALL {:.3f} {:.3f} {:.3f} {:.3f} {:.3f}".format(
        score_prec, score_rec, score_f1, pos_rate, score_acc))
    id_list.append('ALL')
    prec_list.append(score_prec)
    rec_list.append(score_rec)
    f1_list.append(score_f1)
    p_list.append(pos_rate)
    acc_list.append(score_acc)
    df = pd.DataFrame(
        data={
            'id': id_list,
            'precision': prec_list,
            'recall': rec_list,
            'f1': f1_list,
            'pos_rate': p_list,
            'accuracy': acc_list
        })
    df.to_csv(outfile)


def train_and_test_with_bplen_changing(constructor,
                                       device='cuda',
                                       bplen_list=(10000, 1800, 900, 450, 300,
                                                   150, 30)):
    for bplen in bplen_list:
        print("BACKPROP LEN = {}".format(bplen))
        cvlist = generate_cross_validataion_list()
        for i, cv in enumerate(cvlist):
            print("No. {} (TEST GROUP {}) STARTED".format(i, cv[2]))
            trainer = constructor(train_groups=cv[0],
                                  validation_group=cv[1],
                                  test_group=cv[2],
                                  device=device,
                                  backprop_len=bplen)
            trainer.train()
            test(trainer, cv[2], device=device)

            # from . import base as b
            # b.save_facial_turn_detector(trainer, upload=True)
            # trainer.upload_csv_log()
        calc_metrics(trainer)

        
def train_and_test_with_n_changing(constructor,
                                   device='cuda',
                                   n_list=(8, 4, 2, 1)):
    for n_size in n_list:
        print("N SIZE = {}".format(n_size))
        cvlist = generate_cross_validataion_list()
        for i, cv in enumerate(cvlist):
            print("No. {} (TEST GROUP {}) STARTED".format(i, cv[2]))
            trainer = constructor(train_groups=cv[0],
                                  validation_group=cv[1],
                                  test_group=cv[2],
                                  device=device, n=n_size)
            trainer.train()
            test(trainer, cv[2], device=device)

            # from . import base as b
            # b.save_facial_turn_detector(trainer, upload=True)
            # trainer.upload_csv_log()
        calc_metrics(trainer)

        
def train_and_test_with_W_and_C_changing(constructor,
                                         device='cuda'):
    for frame_window in (30, 20, 10):
        for cnn_out_channels in (40, 20, 10, 5):
            print("W={}, C={}".format(frame_window, cnn_out_channels))
            cvlist = generate_cross_validataion_list()
            for i, cv in enumerate(cvlist):
                print("No. {} (TEST GROUP {}) STARTED".format(i, cv[2]))
                trainer = constructor(train_groups=cv[0],
                                      validation_group=cv[1],
                                      test_group=cv[2],
                                      device=device,
                                      frame_window=frame_window,
                                      cnn_out_channels=cnn_out_channels)
                trainer.train()
                test(trainer, cv[2], device=device)

                # from . import base as b
                # b.save_facial_turn_detector(trainer, upload=True)
                # trainer.upload_csv_log()
            calc_metrics(trainer)
        

def construct_D0002_F0005_bplen(train_groups=list(range(3, 11)),
                                validation_group=1,
                                test_group=2,
                                device='cuda',
                                backprop_len=100000):
    from .facial_turn_detector_0002 import FacialTurnDetector0002
    from .feature_extractor_0005 import FacialTurnDetectorFeatureExtractor0005

    feature_extractor = FacialTurnDetectorFeatureExtractor0005(
        device=device, auto_reset=False)
    facial_turn_detector = FacialTurnDetector0002(feature_extractor,
                                                  device=device)
    trainer = FacialTurnDetectorTrainerRinus0004(
        facial_turn_detector,
        train_groups=train_groups,
        validation_group=validation_group,
        test_group=test_group,
        backprop_len=backprop_len)
    return trainer


def construct_D0003W010C020_F0005_bplen(train_groups=list(range(3, 11)),
                                        validation_group=1,
                                        test_group=2,
                                        device='cuda',
                                        backprop_len=100000):
    from .facial_turn_detector_0003 import FacialTurnDetector0003
    from .feature_extractor_0005 import FacialTurnDetectorFeatureExtractor0005

    feature_extractor = FacialTurnDetectorFeatureExtractor0005(
        device=device, auto_reset=False)
    facial_turn_detector = FacialTurnDetector0003(feature_extractor,
                                                  device=device,
                                                  frame_window=10,
                                                  cnn_out_channels=20)
    trainer = FacialTurnDetectorTrainerRinus0004(
        facial_turn_detector,
        train_groups=train_groups,
        validation_group=validation_group,
        test_group=test_group,
        backprop_len=backprop_len)
    return trainer


def construct_D00020101_n(train_groups=list(range(3, 11)),
                          validation_group=1,
                          test_group=2,
                          device='cuda',
                          n=8):
    from .facial_turn_detector_000201 import FacialTurnDetector000201
    from .feature_extractor_0005 import FacialTurnDetectorFeatureExtractor0005

    feature_extractor = FacialTurnDetectorFeatureExtractor0005(
        device=device, auto_reset=False)
    facial_turn_detector = FacialTurnDetector000201(feature_extractor,
                                                    device=device,
                                                    n_size=n)
    trainer = FacialTurnDetectorTrainerRinus0004(
        facial_turn_detector,
        train_groups=train_groups,
        validation_group=validation_group,
        test_group=test_group,
        backprop_len=1800)
    return trainer


def construct_D00020102_n(train_groups=list(range(3, 11)),
                          validation_group=1,
                          test_group=2,
                          device='cuda',
                          n=8):
    from .facial_turn_detector_000202 import FacialTurnDetector000202
    from .feature_extractor_0005 import FacialTurnDetectorFeatureExtractor0005

    feature_extractor = FacialTurnDetectorFeatureExtractor0005(
        device=device, auto_reset=False)
    facial_turn_detector = FacialTurnDetector000202(feature_extractor,
                                                    device=device,
                                                    n_size=n)
    trainer = FacialTurnDetectorTrainerRinus0004(
        facial_turn_detector,
        train_groups=train_groups,
        validation_group=validation_group,
        test_group=test_group,
        backprop_len=1800)
    return trainer


def construct_D00020101_n_0007(train_groups=list(range(3, 11)),
                               validation_group=1,
                               test_group=2,
                               device='cuda',
                               n=8):
    from .facial_turn_detector_000201 import FacialTurnDetector000201
    from .feature_extractor_0007 import FacialTurnDetectorFeatureExtractor0007

    feature_extractor = FacialTurnDetectorFeatureExtractor0007(
        1, 6000, 'lfw_0002', 'LFW0002', device=device, auto_reset=False)
    facial_turn_detector = FacialTurnDetector000201(feature_extractor,
                                                    device=device,
                                                    n_size=n)
    trainer = FacialTurnDetectorTrainerRinus0004(
        facial_turn_detector,
        train_groups=train_groups,
        validation_group=validation_group,
        test_group=test_group,
        backprop_len=1800)
    return trainer


def construct_trainer_D0004_F0007_W_C(train_groups=list(range(3, 11)),
                                      validation_group=1,
                                      test_group=2,
                                      device='cuda',
                                      frame_window=10,
                                      cnn_out_channels=10):
    from .feature_extractor_0007 import FacialTurnDetectorFeatureExtractor0007
    from .facial_turn_detector_0004 import FacialTurnDetector0004
    
    feature_extractor = FacialTurnDetectorFeatureExtractor0007(
        1, 6000, 'lfw_0002', 'LFW0002', device=device, auto_reset=False)
    facial_turn_detector = FacialTurnDetector0004(feature_extractor,
                                                  frame_window=frame_window,
                                                  cnn_out_channels=cnn_out_channels,
                                                  device=device)
    trainer = FacialTurnDetectorTrainerRinus0004(
        facial_turn_detector,
        train_groups=train_groups,
        validation_group=validation_group,
        test_group=test_group,
        backprop_len=1800)
    return trainer


def construct_trainer_D0004_F0006_W_C(train_groups=list(range(3, 11)),
                                      validation_group=1,
                                      test_group=2,
                                      device='cuda',
                                      frame_window=10,
                                      cnn_out_channels=10):
    from .feature_extractor_0006 import FacialTurnDetectorFeatureExtractor0006
    from .facial_turn_detector_0004 import FacialTurnDetector0004
    
    feature_extractor = FacialTurnDetectorFeatureExtractor0006(
        1, 6000, 'lfw_0002', 'LFW0002', device=device, auto_reset=False)
    facial_turn_detector = FacialTurnDetector0004(feature_extractor,
                                                  frame_window=frame_window,
                                                  cnn_out_channels=cnn_out_channels,
                                                  device=device)
    trainer = FacialTurnDetectorTrainerRinus0004(
        facial_turn_detector,
        train_groups=train_groups,
        validation_group=validation_group,
        test_group=test_group,
        backprop_len=1800)
    return trainer


def construct_D00020101_n_F0008D(train_groups=list(range(3, 11)),
                                 validation_group=1,
                                 test_group=2,
                                 device='cuda',
                                 n=8):
    from .facial_turn_detector_000201 import FacialTurnDetector000201
    from .feature_extractor_0008 import FacialTurnDetectorFeatureExtractor0008

    feature_extractor = FacialTurnDetectorFeatureExtractor0008(
        use_landmarks=False, use_dxdydw=True, device=device, auto_reset=False)
    facial_turn_detector = FacialTurnDetector000201(feature_extractor,
                                                    device=device,
                                                    n_size=n)
    trainer = FacialTurnDetectorTrainerRinus0004(
        facial_turn_detector,
        train_groups=train_groups,
        validation_group=validation_group,
        test_group=test_group,
        backprop_len=1800)
    return trainer


def construct_D00020101_n_F0008LD(train_groups=list(range(3, 11)),
                                  validation_group=1,
                                  test_group=2,
                                  device='cuda',
                                  n=8):
    from .facial_turn_detector_000201 import FacialTurnDetector000201
    from .feature_extractor_0008 import FacialTurnDetectorFeatureExtractor0008

    feature_extractor = FacialTurnDetectorFeatureExtractor0008(
        use_landmarks=True, use_dxdydw=True, device=device, auto_reset=False)
    facial_turn_detector = FacialTurnDetector000201(feature_extractor,
                                                    device=device,
                                                    n_size=n)
    trainer = FacialTurnDetectorTrainerRinus0004(
        facial_turn_detector,
        train_groups=train_groups,
        validation_group=validation_group,
        test_group=test_group,
        backprop_len=1800)
    return trainer
