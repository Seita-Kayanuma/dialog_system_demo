# trainer0006 の Torch版
import numpy as np
from ....corpus.speech.csj import CSJ, SpecImageDataManagerTorchCSJ
from ....sound.sigproc.spec_image_torch \
    import SpectrogramImageGeneratorTorch
from ....corpus.speech.spec_image_torch \
    import SpecImageRandomAccessorTorch
from ....sound.sigproc.noise import NoiseAdder
from ....corpus.noise import JEIDA
from .base import SpectrogramImageAutoEncoderTrainer
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
import torch.nn as nn
import torch.optim as optim
from ....ext.torch.trainer import TorchTrainerCallback


class SpectrogramImageDataset(Dataset):
    def __init__(self, cond=None, max_utterance_num=100, add_noise=True, device=None):
        self._csj = CSJ()
        self._id_list = self._csj.get_id_list(cond)
        self._generator = SpectrogramImageGeneratorTorch()
        if device is not None:
            self._generator.to(device)
        if add_noise:
            jeida = JEIDA()
            self._noise_adder = NoiseAdder(jeida.get_wav_path_list())
        else:
            self._noise_adder = None
        self._spec_image_data_manager = SpecImageDataManagerTorchCSJ(
            self._csj, self._generator, self._noise_adder)
        self._spec_image_accessor = SpecImageRandomAccessorTorch(
            self._spec_image_data_manager,
            self._id_list,
            max_num_data_for_id=max_utterance_num)

        if add_noise:
            self._max_index = self._spec_image_accessor.num_images * 2
            self._real_max_index = self._spec_image_accessor.num_images
            self._indices = list(range(self._max_index))
            # 完全なシャッフルにするとキャッシュが大きくなりすぎて結局
            # 効率的ではなくなる
            # np.random.shuffle(self._indices)
            # クリーンとノイズをテレコにする
            self._indices = np.array(self._indices,
                                     'int').reshape(2, -1).T.ravel().tolist()
        else:
            self._max_index = self._spec_image_accessor.num_images
            self._real_max_index = self._spec_image_accessor.num_images
            self._indices = list(range(self._max_index))

    def __len__(self):
        return self._max_index

    def __getitem__(self, i):
        x = None
        y = None
        index = self._indices[i]
        if index < self._real_max_index:
            clean_image = \
                self._spec_image_accessor.get_clean_image(index % self._real_max_index)
            x = clean_image
            y = clean_image
        else:
            clean_image, noised_image = \
                self._spec_image_accessor.get_image_pair(index % self._real_max_index)
            x = noised_image
            y = clean_image
        # チャネル軸を付与
        # x = x.reshape((1, ) + x.shape)
        # y = y.reshape((1, ) + y.shape)
        x = x.unsqueeze(0)
        y = y.unsqueeze(0)

        # return (np.float32(x), np.float32(y))
        # import ipdb; ipdb.set_trace()
        return (x, y)

    def shuffle(self):
        self._spec_image_accessor.clear_cache()
        self._spec_image_accessor.shuffle()

    def clear_cache(self):
        self._spec_image_accessor.clear_cache()


class MyCollate:
    """
    バッチ取得後，毎回キャッシュをクリアするためのフック．
    マルチスレッド処理をするときにこれを使わないと
    メモリを使いすぎる．
    """

    def __init__(self, dataset):
        self._dataset = dataset

    def __call__(self, batch):
        # print("MyCollate __call__ called")
        result = default_collate(batch)
        # import ipdb; ipdb.set_trace()
        self._dataset.clear_cache()
        return result

    
class DatasetShuffleCallback(TorchTrainerCallback):
    def __init__(self, dataset: SpectrogramImageDataset):
        self._dataset = dataset

    def train_epoch_finish_callback(self, trainer, epoch, total_epoch,
                                    train_loss):
        self._dataset.shuffle()

        
class SpectrogramImageAutoEncoderTrainer0011(
        SpectrogramImageAutoEncoderTrainer):
    """
    """

    def __init__(self):
        super().__init__()
        self._train_loader = None
        self._test_loader = None
        
    def build_data(self):
        self._train_dataset = SpectrogramImageDataset(
            cond='[AS].*[^1]$',
            # cond='[AS].*1$',
            # cond='A01M0014',
            max_utterance_num=10,
            device=self._device,
        )
        self._train_batch_size = 1000
        train_collate = MyCollate(self._train_dataset)
        self._train_loader = DataLoader(
            self._train_dataset,
            batch_size=self._train_batch_size,
            # batch_size=500,
            # batch_size=1000,
            shuffle=False,
            # num_workers=8,
            num_workers=0,
            # batch_sampler=train_batch_sampler,
            collate_fn=train_collate)
        self._test_dataset = SpectrogramImageDataset(
            cond='[AS].*1$',
            # cond='A01M0015',
            max_utterance_num=10,
            add_noise=False,
            device=self._device,
        )
        self._test_batch_size = 100
        test_collate = MyCollate(self._test_dataset)
        self._test_loader = DataLoader(
            self._test_dataset,
            batch_size=self._test_batch_size,
            # batch_size=500,
            # batch_size=1000,
            shuffle=False,
            # num_workers=8,
            num_workers=0,
            # batch_sampler=test_batch_sampler,
            collate_fn=test_collate,
        )
        # import ipdb; ipdb.set_trace()

    def get_criterion(self):
        return nn.MSELoss()

    def get_optimizer(self, model):
        return optim.Adam(model.parameters())

    def get_train_loader(self):
        if self._train_loader is None:
            self.build_data()
        return self._train_loader

    def get_validation_loader(self):
        if self._test_loader is None:
            self.build_data()
        return self._test_loader

    def get_additional_callbacks(self):
        if self._train_loader is None:
            self.build_data()
        return [DatasetShuffleCallback(self._train_dataset)]
