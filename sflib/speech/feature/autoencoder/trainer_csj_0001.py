import numpy as np
from .base import SpectrogramImageAutoEncoderTrainer
# warning (deprecated)
from sflib.corpus.csj.process import SpectrogramImageArrayWithVAD


class SpectrogramImageIterator:
    def __init__(self, spectrogram_image_array, max_index, batch_size):
        self._sia = spectrogram_image_array
        self._indices = list(range(max_index))
        np.random.shuffle(self._indices)
        self._max_index = max_index
        self._batch_size = batch_size

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
        r = np.stack([self._sia[i] for i in ind])
        r = r.reshape(r.shape + (1, ))

        if end == self._max_index:
            np.random.shuffle(self._indices)
            self._current = 0
        else:
            self._current = end

        return (r, r)


class SpectrogramImageAutoEncoderTrainerCSJ0001(
        SpectrogramImageAutoEncoderTrainer):
    def __init__(self):
        super().__init__()

        self.x_train = None
        self.x_test = None
        self._train_data_generator = None

    def build_data(self):
        sia = SpectrogramImageArrayWithVAD(
            cond='A01', max_wav_num_for_each_file=10, shuffle=True)
        sia.construct()
        print("total %d samples loaded" % len(sia))
        self.x_test = np.stack(sia[-1000:])
        self.x_test = self.x_test.reshape(self.x_test.shape + (1, ))

        self._train_data_generator = SpectrogramImageIterator(
            sia, max_index=len(sia) - 1000, batch_size=500)

    def train(self):
        self._build()
        self.build_data()
        self._model.compile(optimizer='adam', loss='mean_squared_error')
        self._model.fit_generator(
            generator=self._train_data_generator,
            steps_per_epoch=self._train_data_generator.steps_per_epoch,
            epochs=20,
            validation_data=(self.x_test, self.x_test))


def train(autoencoder_version):
    module_name = 'sflib.speech.feature.autoencoder.autoencoder%04d' \
        % autoencoder_version
    class_name = "SpectrogramImageAutoEncoder%04d" % autoencoder_version
    import importlib
    mod = importlib.import_module(module_name)
    cls = getattr(mod, class_name)
    autoencoder = cls()
    trainer = SpectrogramImageAutoEncoderTrainerCSJ0001()
    trainer.set_autoencoder(autoencoder)
    trainer.train()
    trainer.save(upload=True)
