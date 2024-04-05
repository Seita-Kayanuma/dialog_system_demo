import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pack_padded_sequence

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


def collate_fn(feat_target_list):
    feat_list = []
    target_list = []
    for f, t in feat_target_list:
        feat_list.append(f)
        target_list.append(t.reshape(-1, 1))
    return feat_list, target_list


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


class FacialTurnDetectorTrainerRinus0002(FacialTurnDetectorTrainer):
    def __init__(self, turn_detector: FacialTurnDetector):
        super().__init__(turn_detector)

    def build_torch_trainer(self, turn_detector: FacialTurnDetector):
        criterion = CrossEntropyLossForLSTM()
        if turn_detector.device is not None:
            criterion = criterion.to(turn_detector.device)
            
        # optimizer = optim.SGD(turn_detector.torch_model.parameters(),
        #                       lr=0.01,
        #                       weight_decay=0.01)
        optimizer = optim.Adam(turn_detector.torch_model.parameters(),
                               weight_decay=0.0)
    
        rinus = Rinus()
        rinus_id_list = rinus.get_id_list()
        train_id_list = rinus_id_list[:90]
        valid_id_list = rinus_id_list[90:]
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
                                  batch_size=3,
                                  collate_fn=collate_fn,
                                  num_workers=0,
                                  shuffle=True)
        valid_loader = DataLoader(valid_ds,
                                  batch_size=1,
                                  collate_fn=collate_fn,
                                  num_workers=0,
                                  shuffle=False)
        callbacks = [
            ClippingGrad(1.),
            StandardReporter(train_report_interval=1,
                             validation_report_interval=1),
            CsvWriterReporter(self.get_csv_log_filename()),
            Snapshot(final_filename=self.get_model_filename()),
            EarlyStopper(patience=5, verbose=True),
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
            backprop_len=1000)

        return trainer


def train(device='cuda'):
    from .feature_extractor_0001 import FacialTurnDetectorFeatureExtractor0001
    from .facial_turn_detector_0001 import FacialTurnDetector0001

    feature_extractor = FacialTurnDetectorFeatureExtractor0001(1,
                                                               6000,
                                                               'lfw_0002',
                                                               'LFW0002',
                                                               device=device)
    facial_turn_detector = FacialTurnDetector0001(feature_extractor,
                                                  device=device)
    trainer = FacialTurnDetectorTrainerRinus0002(facial_turn_detector)

    trainer.train()

    from . import base as b
    b.save_facial_turn_detector(trainer, upload=True)
    trainer.upload_csv_log()


def train0002(device='cuda'):
    from .feature_extractor_0001 import FacialTurnDetectorFeatureExtractor0001
    from .facial_turn_detector_0002 import FacialTurnDetector0002

    feature_extractor = FacialTurnDetectorFeatureExtractor0001(1,
                                                               6000,
                                                               'lfw_0002',
                                                               'LFW0002',
                                                               device=device)
    facial_turn_detector = FacialTurnDetector0002(feature_extractor,
                                                  device=device)
    trainer = FacialTurnDetectorTrainerRinus0002(facial_turn_detector)

    trainer.train()

    from . import base as b
    b.save_facial_turn_detector(trainer, upload=True)
    trainer.upload_csv_log()

    
def train0003(device='cuda'):
    from .feature_extractor_0002 import FacialTurnDetectorFeatureExtractor0002
    from .facial_turn_detector_0002 import FacialTurnDetector0002

    feature_extractor = FacialTurnDetectorFeatureExtractor0002(1,
                                                               6000,
                                                               'lfw_0002',
                                                               'LFW0002',
                                                               device=device)
    facial_turn_detector = FacialTurnDetector0002(feature_extractor,
                                                  device=device)
    trainer = FacialTurnDetectorTrainerRinus0002(facial_turn_detector)

    trainer.train()

    from . import base as b
    b.save_facial_turn_detector(trainer, upload=True)
    trainer.upload_csv_log()

    
def train0004(device='cuda'):
    from .feature_extractor_0003 import FacialTurnDetectorFeatureExtractor0003
    from .facial_turn_detector_0002 import FacialTurnDetector0002

    feature_extractor = FacialTurnDetectorFeatureExtractor0003(1,
                                                               6000,
                                                               'lfw_0002',
                                                               'LFW0002',
                                                               device=device,
                                                               auto_reset=False)
    facial_turn_detector = FacialTurnDetector0002(feature_extractor,
                                                  device=device)
    trainer = FacialTurnDetectorTrainerRinus0002(facial_turn_detector)

    trainer.train()

    from . import base as b
    b.save_facial_turn_detector(trainer, upload=True)
    trainer.upload_csv_log()
    
