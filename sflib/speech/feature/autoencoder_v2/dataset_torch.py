from torch.utils.data import Dataset
from corpus.speech.spec_image_torch \
    import SpecImageDataManagerTorch, SpecImageRandomAccessorTorch
from corpus.noise.spec_image_torch \
    import SpecImageDataManagerForNoiseTorch
import numpy as np
import torch
from torch.utils.data.dataloader import default_collate
from torch.utils.data import ConcatDataset, Subset
import copy


class SpectrogramImageDatasetTorch(Dataset):
    def __init__(self,
                 spec_image_data_manager: SpecImageDataManagerTorch,
                 id_list: list,
                 max_utterance_num=None):
        self._spec_image_data_manager = spec_image_data_manager
        self._id_list = id_list

        self._spec_image_accessor = SpecImageRandomAccessorTorch(
            self._spec_image_data_manager,
            self._id_list,
            max_num_data_for_id=max_utterance_num)

        if self._spec_image_data_manager.noise_adder is not None:
            self._max_index = self._spec_image_accessor.num_images * 2
            self._real_max_index = self._spec_image_accessor.num_images
            self._indices = list(range(self._max_index))
            self._indices = np.array(self._indices,
                                     'int').reshape(2, -1).T.ravel().tolist()
        else:
            self._max_index = self._spec_image_accessor.num_images
            self._real_max_index = self._max_index
            self._indices = list(range(self._max_index))

    def __len__(self):
        return self._max_index

    def __getitem__(self, i):
        x = None
        y = None
        index = self._indices[i]
        if index < self._real_max_index:
            clean_image = self._spec_image_accessor.get_clean_image(
                index % self._real_max_index)
            x = clean_image
            y = clean_image
        else:
            clean_image, noised_image = \
                self._spec_image_accessor.get_image_pair(
                    index % self._real_max_index)
            x = noised_image
            y = clean_image
        # チャネル軸を付与
        x = x.unsqueeze(0)
        y = y.unsqueeze(0)

        return (x, y)

    def shuffle(self):
        self._spec_image_accessor.clear_cache()
        self._spec_image_accessor.shuffle()

    def clear_cache(self):
        self._spec_image_accessor.clear_cache()


class SpectrogramImageDatasetForNoiseTorch(Dataset):
    def __init__(self,
                 spec_image_data_manager: SpecImageDataManagerForNoiseTorch,
                 id_list: list,
                 max_utterance_num=None):
        self._spec_image_data_manager = spec_image_data_manager
        self._id_list = id_list

        self._spec_image_accessor = SpecImageRandomAccessorTorch(
            self._spec_image_data_manager,
            self._id_list,
            max_num_data_for_id=max_utterance_num)

        if self._spec_image_data_manager.noise_adder is not None:
            self._max_index = self._spec_image_accessor.num_images * 2
            self._real_max_index = self._spec_image_accessor.num_images
            self._indices = list(range(self._max_index))
            self._indices = np.array(self._indices,
                                     'int').reshape(2, -1).T.ravel().tolist()
        else:
            self._max_index = self._spec_image_accessor.num_images
            self._real_max_index = self._max_index
            self._indices = list(range(self._max_index))

        self.__clean_image = None

    def __len__(self):
        return self._max_index

    def __getitem__(self, i):
        x = None
        index = self._indices[i]
        if index < self._real_max_index:
            clean_image = \
                self._spec_image_accessor.get_clean_image(
                    index % self._real_max_index)
            x = clean_image
        else:
            clean_image, noised_image = \
                self._spec_image_accessor.get_image_pair(
                    index % self._real_max_index)
            x = noised_image

        # チャネル軸を付与
        x = x.unsqueeze(0)
        if self.__clean_image is None:
            # self.__clean_image = 20.0 * np.log10(
            #     np.ones(x.shape, np.float32) * 1e-40)
            # self.__clean_image = np.zeros(x.shape, np.float32)
            self.__clean_image = torch.zeros(
                x.shape, dtype=torch.float32).to(x.device)
        y = self.__clean_image

        return (x, y)

    def shuffle(self):
        self._spec_image_accessor.clear_cache()
        self._spec_image_accessor.shuffle()

    def clear_cache(self):
        self._spec_image_accessor.clear_cache()


class ConcatDatasetWrapper(ConcatDataset):
    def __init__(self, datasets):
        super().__init__(datasets)

    def shuffle(self):
        new_datasets = copy.copy(self.datasets)
        # np.random.shuffle(new_datasets)
        for ds in new_datasets:
            if hasattr(ds, 'shuffle'):
                ds.shuffle()
        super().__init__(new_datasets)

        
def test_concat_dataset_wrapper():
    from ....corpus.speech.csj import CSJ
    from ....sound.sigproc.spec_image_torch \
        import SpectrogramImageGeneratorTorch
    csj = CSJ()
    id_list = csj.get_id_list(
        cond='[AS].*1$',
        # cond='[AS].*[^1]$',
        # cond='A01M0014',
    )
    generator = SpectrogramImageGeneratorTorch()
    sidm = csj.get_spec_image_data_manager_torch(generator)
    dataset1 = SpectrogramImageDatasetTorch(sidm, id_list, 10)
    dataset2 = SpectrogramImageDatasetTorch(sidm, id_list, 5)
    return ConcatDatasetWrapper([dataset1, dataset2])
    
        
class CollateForSpectrogramImageDatasetTorch:
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
