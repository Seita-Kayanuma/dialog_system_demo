# coding: utf-8
import warnings
from os import path
from keras import backend as K
from keras.models import Model
from ..... import config
from .....cloud.google import GoogleDriveInterface


class FaceAutoEncoder:
    """
    FaceAutoEncoderの基底クラス
    """

    def get_filename_base(self):
        return self.__class__.__name__

    def build(self, input, encoded_out, encoded_in, decoded):
        self.model = Model(input, decoded)
        self.encoder = K.function([input], [encoded_out])
        self.decoder = K.function([encoded_in], [decoded])
        self.encoded_dim = encoded_out.shape[1].value
        # import ipdb; ipdb.set_trace()

    def fit(self, *args, **kwargs):
        return self.model.fit(*args, **kwargs)

    def encode(self, img):
        img = img.reshape(-1, 96, 96, 1) / 255.0
        result = self.encoder([img])
        return result[0]

    def decode(self, vec):
        input_dim = int(self.decoder.inputs[0].shape[1])
        vec_in = vec.reshape(-1, input_dim)
        imgs = self.decoder([vec_in])
        return imgs[0].reshape(-1, 96, 96) * 255.0

    def save_weights(self, filename):
        self.model.save_weights(filename)

    def load_weights(self, filename):
        self.model.load_weights(filename)


class FaceAutoEncoderTrainer:
    def __init__(self):
        self.autoencoder = None
        self.history = None

    def get_filename_base(self):
        return self.__class__.__name__

    def set_autoencoder(self, autoencoder):
        self.autoencoder = autoencoder
        self.history = None

    def compile(self, *args, **kwargs):
        self.autoencoder.model.compile(*args, **kwargs)

    def get_train_data(self):
        pass

    def get_validation_data(self):
        return None

    def train(self, epochs=20, batch_size=256, shuffle=True, **kwargs):
        x_train, y_train = self.get_train_data()
        validation_data = self.get_validation_data()

        self.history = self.autoencoder.fit(
            x=x_train,
            y=y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            shuffle=shuffle,
            **kwargs)


def train_face_auto_encoder(autoencoder, trainer):
    trainer.set_autoencoder(autoencoder)
    trainer.compile()
    trainer.train()


def get_weights_filename(autoencoder, trainer, option=''):
    filename = "%s+%s+%s.h5" % (autoencoder.get_filename_base(),
                                trainer.get_filename_base(), option)
    fullpath = path.join(config.get_package_data_dir(__package__), filename)
    return fullpath


def save_face_auto_encoder_weights(autoencoder, trainer, upload=False):
    if trainer.history is None:
        warnings.warn("autoencoder is not trained yet")
    filename = get_weights_filename(autoencoder, trainer)
    autoencoder.save_weights(filename)
    if upload is True:
        g = GoogleDriveInterface(read_only=False)
        g.upload(filename, path.basename(filename))


def load_face_auto_encoder_weights(autoencoder, trainer, download=False):
    filename = get_weights_filename(autoencoder, trainer)
    if download is True:
        g = GoogleDriveInterface()
        g.download(path.basename(filename), filename)
    autoencoder.load_weights(filename)
