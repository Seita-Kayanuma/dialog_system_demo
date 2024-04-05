import numpy as np
from .base import convert_phone_to_id
from keras.utils import Sequence


class PhoneTypeWriterDataIterator(Sequence):
    def __init__(self, spectrogram_image_array_with_vad_phones,
                 feature_extractor, batch_size):
        self._sia = spectrogram_image_array_with_vad_phones
        self._feature_extractor = feature_extractor
        self._indices = list(range(len(self._sia)))
        np.random.shuffle(self._indices)
        self._batch_size = batch_size
        self._current = 0

    @property
    def feature_dim(self):
        return self._feature_extractor.get_feature_dim()

    @property
    def steps_per_epoch(self):
        return len(self._indices) // self._batch_size + 1

    def __iter__(self):
        return self

    def get_batch(self, indices):
        x_list = []
        x_length_list = []
        y_list = []
        y_length_list = []

        for i in indices:
            (image, phones) = self._sia[i]

            x = self._feature_extractor.calc(image)
            y = np.array(convert_phone_to_id(phones))

            # 入力サイズより出力サイズが大きかったら
            # 出力サイズを削る
            if y.shape[0] > x.shape[0]:
                y = y[:x.shape[0]]

            x_list.append(x)
            x_length_list.append(x.shape[0])
            y_list.append(y)
            y_length_list.append(y.shape[0])

        x_length = np.array(x_length_list).reshape((-1, 1))
        y_length = np.array(y_length_list).reshape((-1, 1))

        x_length_max = np.max(x_length)
        x_out = np.zeros((
            len(indices),
            x_length_max,
            self._feature_extractor.get_feature_dim(),
        ))
        for i, x in enumerate(x_list):
            x_out[i, :len(x), :] = x

        y_length_max = np.max(y_length)
        y_out = np.ones((len(indices), y_length_max)) * -1
        for i, y in enumerate(y_list):
            y_out[i, :len(y)] = y

        # 入力
        inputs = {
            'the_input': x_out,
            'the_labels': y_out,
            'input_length': x_length,
            'label_length': y_length
        }
        # ダミー出力
        outputs = {'ctc': np.zeros(len(indices))}

        return (inputs, outputs)

    def __next__(self):
        start = self._current
        end = self._current + self._batch_size
        if end > len(self._indices):
            end = len(self._indices)
        ind = self._indices[start:end]

        batch = self.get_batch(ind)
        if end == len(self._indices):
            np.random.shuffle(self._indices)
            self._current = 0
        else:
            self._current = end

        return batch

    def __len__(self):
        return self.steps_per_epoch

    def __getitem__(self, idx):
        start = idx * self._batch_size
        end = start + self._batch_size
        if end > len(self._indices):
            end = len(self._indices)
        ind = self._indices[start:end]
        batch = self.get_batch(ind)
        return batch

    def on_epoch_end(self):
        np.random.shuffle(self._indices)


class CombinedSpectrogramImageArrayWithVadPhones:
    def __init__(self, spectrogram_image_array_with_vad_phones_list):
        self._sia_list = spectrogram_image_array_with_vad_phones_list
        index_base_list = [0] + [len(sia) for sia in self._sia_list]
        index_base_list = index_base_list[:-1]
        self._index_base_list = np.array(index_base_list, 'int')

    def __len__(self):
        r = 0
        for sia in self._sia_list:
            r += len(sia)
        return r

    def __getitem__(self, key):
        if isinstance(key, int):
            return self.get_item_by_index(key)
        elif isinstance(key, slice):
            r = [
                self.get_item_by_index(i)
                for i in range(*(key.indices(len(self))))
            ]
            return tuple(r)
        else:
            raise ValueError()

    def get_item_by_index(self, i):
        ii = i - self._index_base_list
        sia_index = np.where(ii >= 0)[0][-1]
        local_index = int(ii[sia_index])
        return self._sia_list[sia_index][local_index]
