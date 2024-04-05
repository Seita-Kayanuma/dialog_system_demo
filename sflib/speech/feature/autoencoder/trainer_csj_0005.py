import numpy as np
import keras.callbacks
from keras.utils import Sequence
from .base import SpectrogramImageAutoEncoderTrainer
from ....corpus.speech.spec_image import SpecImageRandomAccessor
from ....corpus.speech.csj import CSJ, SpecImageDataManagerCSJ
from ....sound.sigproc.spec_image import SpectrogramImageGenerator
from ....sound.sigproc.noise import NoiseAdder
from ....corpus.noise import JEIDA


class SpectrogramImageSequence(Sequence):
    def __init__(self, batch_size, cond=None, max_utterance_num=100):
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
            max_num_data_for_id=max_utterance_num)

        self._max_index = self._spec_image_accessor.num_images * 2
        self._real_max_index = self._spec_image_accessor.num_images
        self._batch_size = batch_size
        self._indices = list(range(self._max_index))
        # 完全なシャッフルにするとキャッシュが大きくなりすぎて結局
        # 効率的ではなくなる
        # np.random.shuffle(self._indices)
        # クリーンとノイズをテレコにする
        self._indices = np.array(self._indices,
                                 'int').reshape(2, -1).T.ravel().tolist()

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

    def on_epoch_end(self):
        self._spec_image_accessor.shuffle()


class SpectrogramImageAutoEncoderTrainerCSJ0005(
        SpectrogramImageAutoEncoderTrainer):
    """
    """
    def __init__(self):
        super().__init__()

        self.x_train = None
        self.x_test = None
        self._train_data_generator = None

    def build_data(self):
        self._train_data_generator = SpectrogramImageSequence(
            batch_size=1000, cond='A.*[^1]$', max_utterance_num=100)
        # batch_size=500, cond='A01M0014')
        self._test_data_generator = SpectrogramImageSequence(
            batch_size=1000, cond='A.*1$', max_utterance_num=10)
        # batch_size=500, cond='A01M0015')
        # import ipdb; ipdb.set_trace()


    def train(self):
        self._build()
        self.build_data()
        self._model.compile(optimizer='adam', loss='mean_squared_error')

        cb_early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=3,
            verbose=1,
            mode='auto',
            restore_best_weights=True)

        import multiprocessing
        process_count = multiprocessing.cpu_count() // 4 + 1

        self._model.fit_generator(
            generator=self._train_data_generator,
            steps_per_epoch=self._train_data_generator.steps_per_epoch,
            epochs=20,
            validation_data=self._test_data_generator,
            validation_steps=self._test_data_generator.steps_per_epoch,
            callbacks=[cb_early_stopping],
            max_queue_size=process_count * 10,
            use_multiprocessing=True,
            workers=process_count)

        
def train(autoencoder_version):
    module_name = 'sflib.speech.feature.autoencoder.autoencoder%04d' \
        % autoencoder_version
    class_name = "SpectrogramImageAutoEncoder%04d" % autoencoder_version
    import importlib
    mod = importlib.import_module(module_name)
    cls = getattr(mod, class_name)
    autoencoder = cls()
    trainer = SpectrogramImageAutoEncoderTrainerCSJ0005()
    trainer.set_autoencoder(autoencoder)
    trainer.train()
    trainer.save(upload=True)
