from .base import SpectrogramImageAutoEncoderTrainer
# from sflib.corpus.csj import CSJ
from sflib.corpus.csj.process import SpectrogramImageArrayWithVAD
import numpy as np
import keras.callbacks
from .util import PdfWriterCallback
from sflib.sound.sigproc.noise import NoiseAdder
from sflib.corpus.noise import JEIDA


class SpectrogramImageIterator:
    def __init__(self, spectrogram_image_array, max_index, batch_size):
        self._sia = spectrogram_image_array
        self._max_index = max_index * 2
        self._real_max_index = max_index
        self._batch_size = batch_size
        self._indices = list(range(self._max_index))
        np.random.shuffle(self._indices)

        self._current = 0

    @property
    def steps_per_epoch(self):
        return self._max_index // self._batch_size + 1

    def __iter__(self):
        return self

    def __next__(self):
        start = self._current
        end = self._current + self._batch_size
        if end > self._max_index:
            end = self._max_index

        ind = self._indices[start:end]
        result_list = [self._sia[i % self._real_max_index] for i in ind]
        # print(type(result_list[0]))
        
        x_list = []
        y_list = []
        for ind, result in zip(ind, result_list):
            if ind >= self._real_max_index:
                x_list.append(result[1])
                y_list.append(result[0])
            else:
                x_list.append(result[0])
                y_list.append(result[0])
                
        x = np.stack(x_list)
        x = x.reshape(x.shape + (1, ))
        y = np.stack(y_list)
        y = y.reshape(y.shape + (1, ))

        # print(len(result_list))
        # print(x.shape, y.shape)

        if end == self._max_index:
            np.random.shuffle(self._indices)
            self._sia.refresh_noise()
            self._current = 0
        else:
            self._current = end

        return (x, y)


class SpectrogramImageAutoEncoderTrainerCSJDraft(
        SpectrogramImageAutoEncoderTrainer):
    """
    新しい機能を手元で試すためのドラフトトレーナー
    """

    def __init__(self):
        super().__init__()

        self.x_train = None
        self.x_test = None
        self._train_data_generator = None

    def build_data(self):
        jeida = JEIDA()
        noise_adder = NoiseAdder(jeida.get_wav_path_list())
        sia = SpectrogramImageArrayWithVAD(
            cond='A01M001[34]',
            tag='VAD-L',
            max_wav_num_for_each_file=10,
            shuffle=True,
            noise_adder=noise_adder)
        sia.construct()
        print("total %d samples loaded" % len(sia))

        self._train_data_generator = SpectrogramImageIterator(
            sia, max_index=len(sia), batch_size=500)

        sia = SpectrogramImageArrayWithVAD(
            cond='A01M001[5]',
            tag='VAD-L',
            max_wav_num_for_each_file=10,
            shuffle=True,
            noise_adder=noise_adder)
        sia.construct()
        print("total %d samples loaded" % len(sia))

        self._test_data_generator = SpectrogramImageIterator(
            sia, max_index=len(sia), batch_size=500)

        # import ipdb; ipdb.set_trace()

    def compile(self, *args, **kwargs):
        super().compile(
            *args, optimizer='adam', loss='mean_squared_error', **kwargs)

    def get_train_data(self):
        return None
        # if self.x_train is None:
        #     self.build_data()
        # return (self.x_train, self.x_train)

    def get_validation_data(self):
        return None
        # if self.x_test is None:
        #     self.build_data()
        # return (self.x_test, self.x_test)

    def get_train_data_generator(self):
        if self._train_data_generator is None:
            self.build_data()
        return self._train_data_generator

    def get_validation_data_generator(self):
        if self._test_data_generator is None:
            self.build_data()
        return self._test_data_generator

    def get_extra_kwargs(self):
        cb_early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=0,
            verbose=1,
            mode='auto',
            restore_best_weights=True)
        cb_pdf = PdfWriterCallback(
            filename='test.pdf',
            title=("%s + %s" % (self.autoencoder.get_filename_base(),
                                self.get_filename_base())))

        return {'callbacks': [cb_early_stopping, cb_pdf]}
