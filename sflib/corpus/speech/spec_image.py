# coding: utf-8
from sflib.sound.sigproc.spec_image \
    import SpectrogramImageArray, SpectrogramImageGenerator
from sflib.sound.sigproc.noise import NoiseAdder
from .wav import WavDataWithTrans, WavDataWithTransManager
import numpy as np
import tqdm
import copy


class SpecImageData:
    def __init__(self,
                 wav_data: WavDataWithTrans,
                 generator: SpectrogramImageGenerator,
                 noise_adder: NoiseAdder = None):
        self._wav_data = wav_data
        self._generator = generator
        self._noise_adder = noise_adder
        self.__cache_clean = None
        self.__cache_noised = None

        # 生成できる画像の本数を予測しておく
        num_samples = self._wav_data.num_samples
        num_samples -= generator.num_samples_per_image
        if num_samples <= 0:
            self.__num_images = 1
        else:
            self.__num_images = int(
                np.ceil(num_samples / generator.num_samples_per_image_shift))

    @property
    def num_images(self):
        return self.__num_images

    @property
    def clean_images(self):
        if self.__cache_clean is not None:
            return self.__cache_clean
        return self.get_image()

    @property
    def noised_images(self):
        if self.__cache_noised is not None:
            return self.__cache_noised
        return self.get_image(noised=True)

    @property
    def wav_data(self):
        return self._wav_data

    @property
    def trans(self):
        return self._wav_data.trans

    def get_image(self, noised=False):
        if noised and self._noise_adder is None:
            raise ValueError()
        wav = self._wav_data.wav
        # 必要に応じてゼロ詰めをする
        num_samples = len(wav)
        num_samples -= self._generator.num_samples_per_image
        if num_samples < 0:
            wav = np.concatenate([wav, np.zeros(-num_samples, np.int16)])
        else:
            residual_num_samples = num_samples % self._generator.num_samples_per_image_shift
            if residual_num_samples > 0:
                n = self._generator.num_samples_per_image_shift - residual_num_samples
                wav = np.concatenate([wav, np.zeros(n, np.int16)])
        # ノイズが加えられればノイズも加える
        if noised and self._noise_adder is not None:
            wav = self._noise_adder.add_noise(wav)
            self._generator.reset()
            self.__cache_noised = np.stack(self._generator.input_wave(wav))
            return self.__cache_noised
        self._generator.reset()
        self.__cache_clean = np.stack(self._generator.input_wave(wav))
        return self.__cache_clean

    def clear(self):
        self.__cache_clean = None
        self.__cache_noised = None
        self._wav_data.clear()

    def __repr__(self):
        return "SpecImageData(cached=%s, num_images=%d, trans=%s)" % (
            self.__cache_clean is not None, self.__num_images,
            self._wav_data.trans.trans)


class SpecImageDataManager:
    def __init__(self,
                 wav_data_manager: WavDataWithTransManager,
                 generator: SpectrogramImageGenerator,
                 noise_adder: NoiseAdder = None):
        self.__wav_data_manager = wav_data_manager
        self.__generator = generator
        self.__noise_adder = noise_adder
        self.__id2data = {}

    @property
    def noise_adder(self):
        return self.__noise_adder

    def get(self, id):
        if id in self.__id2data:
            return self.__id2data[id]

        wav_datas = self.__wav_data_manager.get(id)
        result = []
        for wav_data_list in wav_datas:
            data_list = []
            for wav_data in wav_data_list:
                data_list.append(
                    SpecImageData(wav_data, self.__generator,
                                  self.__noise_adder))
            result.append(data_list)
        self.__id2data[id] = result
        return result

    def clear_cache(self):
        """
        保持する全てのSpecImageDataのキャッシュをクリアする
        """
        for datas in self.__id2data.values():
            for data_list in datas:
                for data in data_list:
                    data.clear()

    def clear(self):
        """
        全てのデータをクリアする
        """
        self.__id2data = {}


class SpecImageRandomAccessor:
    def __init__(self,
                 manager: SpecImageDataManager,
                 id_list,
                 max_num_data_for_id=None):
        self.__manager = manager
        self.__id_list = id_list

        self.__spec_image_data_list = []
        self.__base_indices = []
        self.__num_images = 0
        self.__max_num_data_for_id = max_num_data_for_id

        self.__build_data_list()

    def __build_data_list(self):
        data_list = []
        base_indices = [0]
        count = 0
        for id in tqdm.tqdm(self.__id_list):
            wav_datas = self.__manager.get(id)
            for wav_data_list in wav_datas:
                if self.__max_num_data_for_id is not None and \
                   len(wav_data_list) > self.__max_num_data_for_id:
                    wav_data_list = copy.copy(wav_data_list)
                    # 完全なシャッフルだと偏る可能性があるので，0, 10, 20, ... のように
                    # 飛ばし飛ばしでデータを抽出する
                    # np.random.shuffle(wav_data_list)
                    n = len(wav_data_list)
                    m = self.__max_num_data_for_id
                    ind = np.arange((n // m + 1) * m).reshape(-1, m).T.ravel()
                    ind = ind[ind < n]
                    count += 1
                    count = count % len(ind)
                    ind = np.concatenate([ind[count:], ind[:count]])
                    wav_data_list = np.array(wav_data_list)[ind]
                    wav_data_list = wav_data_list[:self.__max_num_data_for_id].tolist()
                    # import ipdb; ipdb.set_trace()
                for wav_data in wav_data_list:
                    data_list.append(wav_data)
                    base_indices.append(base_indices[-1] + wav_data.num_images)
        self.__spec_image_data_list = data_list
        self.__base_indices = np.array(base_indices[:-1], 'int')
        self.__num_images = base_indices[-1]
        self.__manager.clear()
        self.shuffle()

    def shuffle(self):
        np.random.shuffle(self.__spec_image_data_list)
        base_indices = [0]
        for data in self.__spec_image_data_list:
            base_indices.append(base_indices[-1] + data.num_images)
        self.__base_indices = np.array(base_indices[:-1], 'int')

    def __len__(self):
        return self.__num_images

    @property
    def num_images(self):
        return self.__num_images

    def calc_index(self, i):
        ii = i - self.__base_indices
        image_spec_data_index = np.where(ii >= 0)[0][-1]
        local_index = ii[image_spec_data_index]
        return image_spec_data_index, local_index

    def get_clean_image(self, i):
        data_index, local_index = self.calc_index(i)
        data = self.__spec_image_data_list[data_index]
        image = data.clean_images[local_index]
        return image

    def get_noised_image(self, i):
        data_index, local_index = self.calc_index(i)
        data = self.__spec_image_data_list[data_index]
        image = data.noised_images[local_index]
        return image

    def get_image_pair(self, i):
        clean_image = self.get_clean_image(i)
        noised_image = self.get_noised_image(i)
        return (clean_image, noised_image)

    def clear_cache(self):
        for data in self.__spec_image_data_list:
            data.clear()
