import warnings
from os import path
from keras import backend as K
from keras.models import Model
from .... import config
from ....cloud.google import GoogleDriveInterface


class SpectrogramImageAutoEncoder:
    """
    スペクトログラム画像の自己符号化器
    """

    def get_filename_base(self):
        """
        パラメータ保存などのためのファイル名のヒントを与える．
        クラス名をそのまま返す．
        """
        return self.__class__.__name__

    def build(self, input, encoded, l2, decoded):
        """
        エンコーダ，デコーダを構成する．
        継承先のクラスから呼ばれる．
        """
        self.input = input
        self.decoded = decoded
        self.encoder = K.function([input], [encoded, l2])
        self.decoder = K.function([encoded, l2], [decoded])
        self.image_shape = (input.shape[1].value, input.shape[2].value)
        self.encoded_dim = encoded.shape[1].value

    def encode(self, images):
        """
        与えられた画像のエンコードを行う．
        imagesは，(バッチ, 幅, 高さ)のテンソル（np.array）．
        結果はデコーディング結果である（バッチ, ベクトル次元）のテンソルと，
        正規化係数である(バッチ, 1)のテンソルのリスト．
        """
        if images.ndim == 3:
            images = images.reshape(images.shape + (1, ))
        result = self.encoder([images])
        return result

    def decode(self, vec, l2):
        """
        与えられた特徴量ででコードを行う．
        vecは(バッチ, ベクトル次元)のテンソル，l2は正規化係数である(バッチ, 1）のテンソル．
        出力は，(バッチ, 幅, 高さ)の画像状のテンソル．
        """
        input_dim = int(self.decoder.inputs[0].shape[1])
        vec_in = vec.reshape(-1, input_dim)
        imgs = self.decoder([vec_in, l2])
        return imgs[0].reshape((-1, ) + self.image_shape)


class SpectrogramImageAutoEncoderTrainer:
    def __init__(self):
        self.autoencoder = None
        self._model = None

    def get_filename_base(self):
        return self.__class__.__name__

    def set_autoencoder(self, autoencoder):
        self.autoencoder = autoencoder

    def _build(self):
        self._model = Model(self.autoencoder.input, self.autoencoder.decoded)

    def train(self):
        pass

    def get_model_filename(self, option=''):
        filename = "%s+%s+%s.h5" % (self.autoencoder.get_filename_base(),
                                    self.get_filename_base(), option)
        fullpath = path.join(
            config.get_package_data_dir(__package__), filename)
        return fullpath

    def get_csv_log_filename(self, option=''):
        filename = "%s+%s+%s.csv" % (self.autoencoder.get_filename_base(),
                                     self.get_filename_base(), option)
        fullpath = path.join(
            config.get_package_data_dir(__package__), filename)
        return fullpath

    def save(self, upload=False):
        if self._model is None:
            warnings.warn('autoencoder is not trained yet')
            self._build()
        filename = self.get_model_filename()
        self._model.save_weights(filename)
        if upload is True:
            g = GoogleDriveInterface(read_only=False)
            g.upload(filename, path.basename(filename))

    def load(self, download=False):
        filename = self.get_model_filename()
        if download is True or not path.exists(filename):
            g = GoogleDriveInterface()
            g.download(path.basename(filename), filename)
        self._build()
        self._model.load_weights(filename)

    def upload_csv_log(self):
        filename = self.get_csv_log_filename()
        g = GoogleDriveInterface(read_only=False)
        g.upload(filename, path.basename(filename), mediaType='text/csv')


def load_autoencoder(version):
    module_name = "sflib.speech.feature.autoencoder.autoencoder%04d" \
        % version
    class_name = "SpectrogramImageAutoEncoder%04d" % version
    import importlib
    mod = importlib.import_module(module_name)
    cls = getattr(mod, class_name)
    autoencoder = cls()
    return autoencoder


def load_trainer(module_postfix, class_postfix):
    module_name = "sflib.speech.feature.autoencoder.trainer_%s" \
        % module_postfix
    class_name = "SpectrogramImageAutoEncoderTrainer%s" \
        % class_postfix
    import importlib
    mod = importlib.import_module(module_name)
    cls = getattr(mod, class_name)
    trainer = cls()
    return trainer


def load(autoencoder_version, trainer_module_postfix, trainer_class_postfix):
    autoencoder = load_autoencoder(autoencoder_version)
    trainer = load_trainer(trainer_module_postfix, trainer_class_postfix)
    trainer.set_autoencoder(autoencoder)
    trainer.load()
    return trainer
