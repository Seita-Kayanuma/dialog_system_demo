from .base import SpectrogramImageAutoEncoderTrainer
from ....corpus.speech.spec_image import SpecImageRandomAccessor
from ....corpus.speech.csj import CSJ, SpecImageDataManagerCSJ 
from ....sound.sigproc.spec_image import SpectrogramImageGenerator
from .util import PdfWriterCallback
from sflib.sound.sigproc.noise import NoiseAdder
from sflib.corpus.noise import JEIDA
from ....ext.keras.callbacks import CsvLoggerCallback

import numpy as np
import keras.callbacks
from keras.utils import Sequence


class SpectrogramImageSequence(Sequence):
    def __init__(self, batch_size, cond=None):
        self._csj = CSJ()
        self._id_list = self._csj.get_id_list(cond)
        self._generator = SpectrogramImageGenerator()
        jeida = JEIDA()
        self._noise_adder = NoiseAdder(jeida.get_wav_path_list())
        self._spec_image_data_manager = SpecImageDataManagerCSJ(
            self._csj, self._generator, self._noise_adder)
        self._spec_image_accessor = SpecImageRandomAccessor(
            self._spec_image_data_manager,
            self._id_list,
            max_num_data_for_id=2)

        self._max_index = self._spec_image_accessor.num_images * 2
        self._real_max_index = self._spec_image_accessor.num_images
        self._batch_size = batch_size
        self._indices = list(range(self._max_index))
        np.random.shuffle(self._indices)

    def __len__(self):
        return int(np.ceil(self._max_index / self._batch_size))

    @property
    def steps_per_epoch(self):
        return len(self)

    def get_batch(self, indices):
        x_list = []
        y_list = []
        for i in indices:
            clean_image, noised_image = \
                self._spec_image_accessor.get_image_pair(i % self._real_max_index)
            if i < self._real_max_index:
                x_list.append(clean_image)
                y_list.append(clean_image)
            else:
                x_list.append(noised_image)
                y_list.append(clean_image)
        self._spec_image_accessor.clear_cache()
        x = np.stack(x_list)
        y = np.stack(y_list)
        x = x.reshape(x.shape + (1, ))
        y = y.reshape(y.shape + (1, ))
        return (x, y)

    def __getitem__(self, i):
        start = i * self._batch_size
        end = start + self._batch_size
        if end > self._max_index:
            end = self._max_index
        indices = self._indices[start:end]
        return self.get_batch(indices)


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
        self._train_data_generator = SpectrogramImageSequence(
            batch_size=500, cond='A01M001[34]')
        self._test_data_generator = SpectrogramImageSequence(
            batch_size=500, cond='A01M0015')
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
        cb_log = CsvLoggerCallback('test.csv')
                
        return {'callbacks': [cb_early_stopping, cb_pdf, cb_log]}
