from torch.utils.data import Dataset
from ....corpus.speech.spec_image \
    import SpecImageDataManager, SpecImageRandomAccessor
from ....corpus.noise.spec_image \
    import SpecImageDataManagerForNoise
import numpy as np
from torch.utils.data.dataloader import default_collate
from torch.utils.data import ConcatDataset, Subset


class SpectrogramImageDataset(Dataset):
    def __init__(self,
                 spec_image_data_manager: SpecImageDataManager,
                 id_list: list,
                 max_utterance_num=None):
        self._spec_image_data_manager = spec_image_data_manager
        self._id_list = id_list

        self._spec_image_accessor = SpecImageRandomAccessor(
            self._spec_image_data_manager,
            self._id_list,
            max_num_data_for_id=max_utterance_num)

        if self._spec_image_data_manager.noise_adder is not None:
            self._max_index = self._spec_image_accessor.num_images * 2
            self._real_max_index = self._spec_image_accessor.num_images
        else:
            self._max_index = self._spec_image_accessor.num_images
            self._real_max_index = self._max_index

    def __len__(self):
        return self._max_index

    def __getitem__(self, i):
        x = None
        y = None
        if i < self._real_max_index:
            clean_image = \
                self._spec_image_accessor.get_clean_image(i % self._real_max_index)
            x = clean_image
            y = clean_image
        else:
            clean_image, noised_image = \
                self._spec_image_accessor.get_image_pair(i % self._real_max_index)
            x = noised_image
            y = clean_image
        # チャネル軸を付与
        x = x.reshape((1, ) + x.shape)
        y = y.reshape((1, ) + y.shape)

        return (np.float32(x), np.float32(y))

    def shuffle(self):
        self._spec_image_accessor.clear_cache()
        self._spec_image_accessor.shuffle()

    def clear_cache(self):
        self._spec_image_accessor.clear_cache()


class SpectrogramImageDatasetForNoise(Dataset):
    def __init__(self,
                 spec_image_data_manager: SpecImageDataManagerForNoise,
                 id_list: list,
                 max_utterance_num=None):
        self._spec_image_data_manager = spec_image_data_manager
        self._id_list = id_list

        self._spec_image_accessor = SpecImageRandomAccessor(
            self._spec_image_data_manager,
            self._id_list,
            max_num_data_for_id=max_utterance_num)

        if self._spec_image_data_manager.noise_adder is not None:
            self._max_index = self._spec_image_accessor.num_images * 2
            self._real_max_index = self._spec_image_accessor.num_images
        else:
            self._max_index = self._spec_image_accessor.num_images
            self._real_max_index = self._max_index

        self.__clean_image = None

    def __len__(self):
        return self._max_index

    def __getitem__(self, i):
        x = None
        if i < self._real_max_index:
            clean_image = \
                self._spec_image_accessor.get_clean_image(i % self._real_max_index)
            x = clean_image
        else:
            clean_image, noised_image = \
                self._spec_image_accessor.get_image_pair(i % self._real_max_index)
            x = noised_image

        # チャネル軸を付与
        x = x.reshape((1, ) + x.shape)
        if self.__clean_image is None:
            # self.__clean_image = 20.0 * np.log10(
            #     np.ones(x.shape, np.float32) * 1e-40)
            self.__clean_image = np.zeros(x.shape, np.float32)
        y = self.__clean_image

        return (np.float32(x), y)

    def shuffle(self):
        self._spec_image_accessor.clear_cache()
        self._spec_image_accessor.shuffle()

    def clear_cache(self):
        self._spec_image_accessor.clear_cache()


class CollateForSpectrogramImageDataset:
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

        def _clear_cache(dataset):
            if isinstance(dataset, ConcatDataset):
                for ds in dataset.datasets:
                    _clear_cache(ds)
            elif isinstance(dataset, Subset):
                _clear_cache(dataset.dataset)
            else:
                dataset.clear_cache()

        _clear_cache(self._dataset)
        
        return result
