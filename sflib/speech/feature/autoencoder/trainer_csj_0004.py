import numpy as np
import keras.callbacks
from .base import SpectrogramImageAutoEncoderTrainer
from sflib.sound.sigproc.noise import NoiseAdder
from sflib.corpus.noise import JEIDA
# warning (deprecated)
from sflib.corpus.csj.process import SpectrogramImageArrayWithVAD


# ノイズを重畳した入力，重畳していない両方のデータを扱えるように
# したIterator
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


class SpectrogramImageAutoEncoderTrainerCSJ0004(
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
            cond='A.*[^1]$',
            tag='VAD-L',
            max_wav_num_for_each_file=10,
            shuffle=True,
            noise_adder=noise_adder)
        sia.construct()
        print("total %d samples loaded" % len(sia))

        self._train_data_generator = SpectrogramImageIterator(
            sia, max_index=len(sia), batch_size=500)

        sia = SpectrogramImageArrayWithVAD(
            cond='A.*1$',
            tag='VAD-L',
            max_wav_num_for_each_file=10,
            shuffle=True,
            noise_adder=noise_adder)
        sia.construct()
        print("total %d samples loaded" % len(sia))

        self._test_data_generator = SpectrogramImageIterator(
            sia, max_index=len(sia), batch_size=500)

    def train(self):
        self._build()
        self.build_data()
        self._model.compile(optimizer='adam', loss='mean_squared_error')

        cb_early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=0,
            verbose=1,
            mode='auto',
            restore_best_weights=True)

        self._model.fit_generator(
            generator=self._train_data_generator,
            steps_per_epoch=self._train_data_generator.steps_per_epoch,
            epochs=20,
            validation_data=self._test_data_generator,
            validation_steps=self._test_data_generator.steps_per_epoch,
            callbacks=[cb_early_stopping])


def train(autoencoder_version):
    module_name = 'sflib.speech.feature.autoencoder.autoencoder%04d' \
        % autoencoder_version
    class_name = "SpectrogramImageAutoEncoder%04d" % autoencoder_version
    import importlib
    mod = importlib.import_module(module_name)
    cls = getattr(mod, class_name)
    autoencoder = cls()
    trainer = SpectrogramImageAutoEncoderTrainerCSJ0004()
    trainer.set_autoencoder(autoencoder)
    trainer.train()
    trainer.save(upload=True)
