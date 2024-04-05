import numpy as np
import keras.callbacks
from keras.utils import Sequence
import keras.backend as K
from keras.losses import mean_squared_error
from .base import SpectrogramImageAutoEncoderTrainer
from ....corpus.speech.spec_image import SpecImageRandomAccessor
from ....corpus.speech.csj import CSJ, SpecImageDataManagerCSJ
from ....sound.sigproc.spec_image import SpectrogramImageGenerator
from ....sound.sigproc.noise import NoiseAdder
from ....corpus.noise import JEIDA


def normalize_with_l2norm(x):
    l2norm = K.sqrt(K.sum(K.square(K.batch_flatten(x)), axis=1, keepdims=True))
    # import ipdb; ipdb.set_trace()
    # l2norm = l2norm / K.prod(x.shape[1:])
    # l2norm = l2norm / K.sum(K.ones(x.shape), axis=1, keepdims=True)
    l2norm = l2norm / (512 * 10)
    l2norm = l2norm + 1e-5
    coef = K.reshape(l2norm, (-1, 1, 1, 1))
    return x / coef


def mean_squared_error_l2norm(y_true, y_pred):
    return mean_squared_error(
        normalize_with_l2norm(y_true), normalize_with_l2norm(y_pred))


class SpectrogramImageSequence(Sequence):
    def __init__(self,
                 batch_size,
                 cond=None,
                 max_utterance_num=100,
                 add_noise=True):
        self._count = 0
        self._csj = CSJ()
        self._id_list = self._csj.get_id_list(cond)
        self._generator = SpectrogramImageGenerator()
        if add_noise:
            jeida = JEIDA()
            self._noise_adder = NoiseAdder(jeida.get_wav_path_list())
        else:
            self._noise_adder = None
        self._spec_image_data_manager = SpecImageDataManagerCSJ(
            self._csj, self._generator, self._noise_adder)
        self._spec_image_accessor = SpecImageRandomAccessor(
            self._spec_image_data_manager,
            self._id_list,
            max_num_data_for_id=max_utterance_num)

        if add_noise:
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
        else:
            self._max_index = self._spec_image_accessor.num_images
            self._real_max_index = self._spec_image_accessor.num_images
            self._batch_size = batch_size
            self._indices = list(range(self._max_index))

    def __len__(self):
        return int(np.ceil(self._max_index / self._batch_size))

    @property
    def steps_per_epoch(self):
        return len(self)

    def get_batch(self, indices):
        x_list = []
        y_list = []
        for i in indices:
            if i < self._real_max_index:
                clean_image = \
                    self._spec_image_accessor.get_clean_image(i % self._real_max_index)
                x_list.append(clean_image)
                y_list.append(clean_image)
            else:
                clean_image, noised_image = \
                    self._spec_image_accessor.get_image_pair(i % self._real_max_index)
                x_list.append(noised_image)
                y_list.append(clean_image)
        self._count += 1
        if self._count % 20 == 0:
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


class SpectrogramImageAutoEncoderTrainerCSJ0007(
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
            batch_size=100, cond='[AS].*[^1]$', max_utterance_num=10)
        # batch_size=500, cond='A01M0014')
        self._test_data_generator = SpectrogramImageSequence(
            batch_size=100,
            cond='[AS].*1$',
            max_utterance_num=10,
            add_noise=False)
        # batch_size=500, cond='A01M0015')
        # import ipdb; ipdb.set_trace()

    def train(self):
        self._build()
        self.build_data()
        self._model.compile(optimizer='adam', loss=mean_squared_error_l2norm)

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
    trainer = SpectrogramImageAutoEncoderTrainerCSJ0007()
    trainer.set_autoencoder(autoencoder)
    trainer.train()
    trainer.save(upload=True)
