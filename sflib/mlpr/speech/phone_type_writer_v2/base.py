# coding: utf-8
from ....cloud.google import GoogleDriveInterface
from os import path
from .... import config
from abc import ABCMeta, abstractmethod
from ....ext.torch.trainer import TorchTrainer
from ....ext.torch.nn.utils.rnn \
    import pack_sequence_with_dummy_length, unpack_sequence
import torch
import torch.nn as nn
from torch.nn.utils.rnn import PackedSequence, pad_sequence
import re
import glob

# 音素のリスト（ヌルが0番目）
phone_list = [
    'nil', 'N', 'N:', 'a', 'a:', 'b', 'by', 'ch', 'd', 'dy', 'e', 'e:', 'f',
    'g', 'gy', 'h', 'hy', 'i', 'i:', 'j', 'k', 'ky', 'm', 'my', 'n', 'ny', 'o',
    'o:', 'p', 'py', 'q', 'r', 'ry', 's', 'sh', 'sp', 't', 'ts', 'ty', 'u',
    'u:', 'w', 'y', 'z', 'zy'
]

# 音素記号から音素番号へ変換するディクショナリ
phone2id = dict([(p, i) for (i, p) in enumerate(phone_list)])


def convert_phone_to_id(phone_list):
    """
    音素リスト（文字列のリスト）を音素ID（整数のリスト）に変換する．
    """
    return [phone2id[phone] for phone in phone_list]


def convert_id_to_phone(label_list):
    """
    音素ID（整数のリスト）を音素リスト（文字列のリスト）に変換する．
    """
    return [phone_list[i] for i in label_list]


class PhoneTypeWriterFeatureExtractor(metaclass=ABCMeta):
    """特徴量抽出器
    """
    def __init__(self):
        # class name check
        m = re.match(r'PhoneTypeWriterFeatureExtractor(\d+)',
                     self.__class__.__name__)
        if m is None:
            raise RuntimeError('class name should match with PhoneTypeWriterFeatureExtractor')
        self.__number = int(m[1])

    @property
    def filename_base(self):
        """特徴抽出器のファイル名"""
        return 'F{:02d}'.format(self.__number)

    @property
    @abstractmethod
    def feature_dim(self):
        """特徴ベクトルの次元数を取得する．
        """
        pass

    def reset(self):
        """状態をリセットする．
        新しいバッチを入力する際などに呼ぶ必要がある．
        """
        pass

    @abstractmethod
    def calc(self, wav_list: list) -> list:
        """波形データから特徴量を計算する．
        
        Args:
          wav_list: 波形データのリスト．
             wav_list[i] は i 番目のバッチの波形データ．
             波形データは任意の長さの1次元 numpy.array(np.int16)
        
        Returns:
          list: 特徴ベクトルのlist
        """
        pass

    @property
    def feature_rate(self):
        raise NotImplementedError

class PhoneTypeWriterTorchModel(nn.Module):
    """PhoneTypeWriterが持つニューラルネット
    """
    def __init__(self):
        super().__init__()
        self.__first_parameter = None

    def __get_first_parameter(self):
        """モデルパラメタの最初のものを取得する．
        モデルがCPUかCUDAのどちらかを判定させるため"""
        if self.__first_parameter is None:
            self.__first_parameter = next(self.parameters())
        return self.__first_parameter

    @property
    def device(self) -> torch.device:
        """デバイス(CPU or CUDA)"""
        return self.__get_first_parameter().device

    @property
    @abstractmethod
    def hidden_feature_dim(self):
        """隠れ層特徴量の次元数
        """
        raise NotImplementedError

    def reset_context(self):
        """コンテクスト（LSTMなどのリカレントな状態）を
        リセットする"""
        pass

    def detach_context(self):
        """コンテクスト（LSTMなどのリカレントな状態）を
        デタッチする．値は残すが，これ以前の状態がバックワード
        計算の対象から外れる"""
        pass

    @abstractmethod
    def calc_hidden_feature(self, feat: PackedSequence) -> PackedSequence:
        """PhoneTypeWriterFeatureExtractorで抽出された特徴量から，
        隠れ層の特徴量を抽出する．
        
        Args:
          feat (PackedSequence): 入力特徴量系列のPackedSequence
        
        Returns:
          PackedSequence: 隠れ層の特徴量列のPackedSequence
        """
        raise NotImplementedError

    @abstractmethod
    def calc_output_from_hidden_feature(self, feat: PackedSequence
                                        ) -> PackedSequence:
        """calc_hidden_featureで計算された隠れ層の特徴量から，
        出力である音素の対数尤度分布列を計算する．
        
        Args:
          feat (PackedSequence): 隠れ層特徴量列のPackedSequence
        
        Returns:
          PackedSequence: 対数尤度分布列のPackedSequence
        """
        raise NotImplementedError

    def forward(self, feat: PackedSequence) -> PackedSequence:
        """フォワード計算．
        内部で calc_hidden_feature -> predict_log_probs_with_hidden_feature
        の順で呼び出される．
        """
        h = self.calc_hidden_feature(feat)
        return self.calc_output_from_hidden_feature(h)
    
    def extract_feature(self, feat: PackedSequence) -> PackedSequence:
        """フォワード計算．
        内部で calc_hidden_feature -> predict_log_probs_with_hidden_feature
        の順で呼び出される．
        """
        h = self.calc_hidden_feature(feat)
        return h


class PhoneTypeWriter(metaclass=ABCMeta):
    """
    PhoneTypeWriter（音素タイプライタ）の基底クラス
    """

    DEFAULT_TRAINER_NUMBER = 3
    """int:
    学習器番号のデフォルト値．事情があって1からではなく3．
    """

    DEFAULT_FEATURE_EXTRACTOR_NUMBER = 2
    """int:
    特徴抽出器番号のデフォルト値．事情があって1からではなく3．
    """

    DEFAULT_FEATURE_EXTRACTOR_CONSTRUCT_ARGS = \
        ([],
         {'autoencoder_number': 12,
          'autoencoder_tariner_number': 6,
          'autoencoder_model_version': 0},)
    """int:
    特徴抽出器コンストラクタ引数のデフォルト値
    """
    def __init__(
            self,
            trainer_number=DEFAULT_TRAINER_NUMBER,
            feature_extractor_number=DEFAULT_FEATURE_EXTRACTOR_NUMBER,
            feature_extractor_construct_args=DEFAULT_FEATURE_EXTRACTOR_CONSTRUCT_ARGS
    ):
        # class name check
        m = re.match(r'PhoneTypeWriter(\d+)', self.__class__.__name__)
        if m is None:
            raise RuntimeError('class name should match with' +
                               'r"PhoneTypeWriter\\d+"')
        self.__number = int(m[1])
        self.__trainer_number = trainer_number
        self._feature_extractor = construct_feature_extractor(
            feature_extractor_number, feature_extractor_construct_args)

    @property
    def feature_extractor(self) -> PhoneTypeWriterFeatureExtractor:
        return self._feature_extractor

    @property
    def device(self):
        return self.torch_model.device

    def to(self, device):
        self.torch_model.to(device)
        self.feature_extractor.to(device)

    @property
    def filename_base(self):
        return 'PTW{:02d}T{:02d}{}'.format(
            self.__number, self.__trainer_number,
            self.feature_extractor.filename_base)

    @property
    @abstractmethod
    def torch_model(self) -> PhoneTypeWriterTorchModel:
        pass

    @property
    def hidden_feature_dim(self) -> int:
        return self.torch_model.hidden_feature_dim

    def reset(self):
        """状態をリセットする"""
        self.feature_extractor.reset()
        self.torch_model.reset_context()

    def detach(self):
        """コンテクストをデタッチする．
        LSTMのバックプロパゲーションを打ち切る場合に利用．
        """
        self.torch_model.detach_context()

    def calc_feature(self, wav_list: list) -> list:
        """wavのリストから，特徴量シーケンスリストを生成する"""
        return self.feature_extractor.calc(wav_list)

    def calc_hidden_feature(self, feat_list: list) -> list:
        """特徴量リストから，隠れ層ベクトルシーケンスリストを生成する"""
        packed_feat_seq, lengths = pack_sequence_with_dummy_length(feat_list)
        packed_hidden_feat_seq = \
            self.torch_model.calc_hidden_feature(packed_feat_seq)
        return unpack_sequence(packed_hidden_feat_seq, lengths)

    def calc_output(self, hidden_feat_list: list) -> list:
        """隠れ層ベクトルシーケンスリストから出力シーケンスリストを生成する"""
        packed_hidden_feat_seq, lengths = \
            pack_sequence_with_dummy_length(hidden_feat_list)
        packed_output = self.torch_model.calc_output_from_hidden_feature(
            packed_hidden_feat_seq)
        return unpack_sequence(packed_output, lengths)

    def calc_output_from_wav(self, wav_list: list) -> list:
        feat_list = self.feature_extractor.calc(wav_list)
        if feat_list is None or len(feat_list) == 0:
            return None
        packed_feat_seq, lengths = pack_sequence_with_dummy_length(feat_list)
        packed_output = self.torch_model.forward(packed_feat_seq)
        return unpack_sequence(packed_output, lengths)

    def predict(self, wav_list: list) -> list:
        """波形データからの予測を行う．

        Args:
          wav_list: 波形データのリスト．
             wav_list[i] は i 番目のバッチの波形データ．
             波形データは任意の長さの1次元 numpy.array(np.int16)

        Returns:
          list: 各バッチの音素文字列のリスト

        PyTorchの制限: デコーダがないので，フレーム同期の記号列しか出ない．
        """
        output_list = self.calc_output_from_wav(wav_list)
        if output_list is None:
            return None
        result = []
        for i, output in enumerate(output_list):
            output = output.detach().cpu().numpy()
            idx_list = output.argmax(axis=1).tolist()
            result.append(convert_id_to_phone(idx_list))
        return result

    # ---- 以下は学習，永続化関係の処理で抽象化可能(?)
    # ---- 実際，SpectrogramImageAutoEncoderとほぼ同じ
    def get_latest_model_version(self):
        """保存済の学習モデルの最新バージョン番号を取得する"""
        pattern = '{}.[0-9]*.torch'.format(self.filename_base)
        pattern = path.join(config.get_package_data_dir(__package__), pattern)
        paths = glob.glob(pattern)
        version = None
        pat = re.compile(r'{}\.(\d+)\.torch'.format(self.filename_base))
        for p in paths:
            m = pat.match(path.basename(p))
            if m is None:
                continue
            v = int(m[1])
            if version is None or version < v:
                version = v
        return version

    def get_model_filename_base(self, version=None, overwrite=False):
        """学習モデルファイルの名前（拡張子を除く）を取得する.

        Args:
          version: 明示的にバージョンを指定する場合はその番号．
                   Noneの場合は最新のものになる．
          overwrite: version=Noneのとき，このオプションがFalseだと最新+1の
                   バージョンのファイル名となる
        """
        if version is None:
            version = self.get_latest_model_version()
            if version is None:
                version = 0
            elif not overwrite:
                version += 1
        filename_base = "{}.{:02d}".format(self.filename_base, version)
        return filename_base

    def get_csv_log_filename(self, version=None, overwrite=False):
        """学習ログを保存するファイル名を取得する"""
        filename = self.get_model_filename_base(version, overwrite) + ".csv"
        filename = path.join(config.get_package_data_dir(__package__),
                             filename)
        return filename

    def get_model_filename(self, version=None, overwrite=False):
        filename = self.get_model_filename_base(version, overwrite) + ".torch"
        filename = path.join(config.get_package_data_dir(__package__),
                             filename)
        return filename

    def save(self, version=None, overwrite=False, upload=True):
        """モデルパラメタの保存
        
        Args:
          version: バージョン番号．Noneの場合は最新版として保存する.
          overwrite: Trueの場合，最新バージョンのファイルに上書きする．
        """
        filename = self.get_model_filename(version, overwrite)
        self.torch_model.eval()
        torch.save(self.torch_model.state_dict(), filename)
        if upload is True:
            g = GoogleDriveInterface(read_only=False)
            g.upload(filename, path.basename(filename))

    def load(self, version=None, download=False, download_overwrite=False):
        """モデルパラメタの読み込み
        
        Args:
          version: バージョン番号. Noneの場合は最新のものを読み込む.
        """
        if download is True:
            g = GoogleDriveInterface()
            g.download_with_filename_pattern(
                self.filename_base,
                r"{}.\d+.torch".format(self.filename_base),
                config.get_package_data_dir(__package__),
                overwrite=download_overwrite)
        if version is None:
            version = self.get_latest_model_version()
        if version is None:
            raise RuntimeError('file not found')
        filename = "{}.{:02d}.torch".format(self.filename_base, version)
        filename = path.join(config.get_package_data_dir(__package__),
                             filename)

        self.torch_model.eval()
        self.torch_model.load_state_dict(
            torch.load(filename, map_location=self.device))

    def upload_csv_log(self):
        """CSVログをGoogle Driveにアップロードする"""
        filename = self.get_csv_log_filename(overwrite=True)
        g = GoogleDriveInterface(read_only=False)
        g.upload(filename, path.basename(filename), mediaType='text/csv')

    def train_phone_type_writer(self):
        trainer = construct_trainer(self.__trainer_number)
        trainer.train(self)


class TorchTrainerForPhoneTypeWriter(TorchTrainer):
    """PhoneTypeWriter用のTorchTrainer"""
    def __init__(self, phone_type_writer: PhoneTypeWriter, *args, **kwargs):
        self._phone_type_writer = phone_type_writer

        # 入力の自動転送は必ず無効化する
        kwargs.update({'automatic_input_transfer': False})
        super().__init__(self._phone_type_writer.torch_model, *args, **kwargs)

    def _forward(self, batch, update=True):
        """
        バッチの内容が特殊
        batch[0] は波形データ(np.int16)のリスト
        batch[1] は音素番号列のリスト
        """
        wav_list, target_list = batch

        self._phone_type_writer.reset()
        output_list = self._phone_type_writer.calc_output_from_wav(wav_list)
        padded_log_probs = pad_sequence(output_list)
        log_probs_len = [len(output) for output in output_list]
        padded_target = torch.t(pad_sequence(target_list)).to(padded_log_probs.device)
        target_len = [len(t) for t in target_list]
        loss = self._criterion(padded_log_probs, padded_target, log_probs_len,
                               target_len)
        if update:
            self._optimzier.zero_grad()
            loss.backward()
            self._callback_train_before_optimizer_step()
            self._optimzier.step()
        return loss


class PhoneTypeWriterTrainer(metaclass=ABCMeta):
    """
    PhoneTypeWriterTrainer（音素タイプライタ学習器）の基底クラス．
    """
    def __init__(self):
        pass

    @abstractmethod
    def build_torch_trainer(self, phone_type_writer):
        pass

    def train(self, phone_type_writer):
        self.torch_trainer = self.build_torch_trainer(phone_type_writer)
        self.torch_trainer.train()


def construct_feature_extractor(feature_extractor_number,
                                feature_extractor_construct_args=(
                                    [],
                                    {},
                                )):
    """特徴抽出器を構築する
    
    Args:
      feature_extractor_number: 構築する特徴抽出器の番号
      feature_extractor_construct_args:
        コンストラクタに与える引数．
        args（リスト）と，kwargs（ディクショナリ）のタプル
    """
    module_name = "sflib.mlpr.speech.phone_type_writer_v2." + \
                  "feature_extractor{:04d}".format(feature_extractor_number)
    class_name = "PhoneTypeWriterFeatureExtractor" + \
                 "{:04d}".format(feature_extractor_number)
    import importlib
    mod = importlib.import_module(module_name)
    cls = getattr(mod, class_name)
    args, kwargs = feature_extractor_construct_args
    feature_extractor = cls(*args, **kwargs)
    return feature_extractor


def construct_phone_type_writer(phone_type_writer_number,
                                trainer_number,
                                feature_extractor_number,
                                feature_extractor_construct_args=(
                                    [],
                                    {},
                                )):
    module_name = "sflib.mlpr.speech.phone_type_writer_v2." + \
                  "phone_type_writer{:04d}".format(phone_type_writer_number)
    class_name = "PhoneTypeWriter{:04d}".format(phone_type_writer_number)
    import importlib
    mod = importlib.import_module(module_name)
    cls = getattr(mod, class_name)
    phone_type_writer = cls(trainer_number, feature_extractor_number,
                            feature_extractor_construct_args)
    return phone_type_writer


def construct_trainer(trainer_number):
    module_name = "sflib.mlpr.speech.phone_type_writer_v2." + \
                  "trainer{:04d}".format(trainer_number)
    class_name = "PhoneTypeWriterTrainer{:04d}".format(trainer_number)
    import importlib
    mod = importlib.import_module(module_name)
    cls = getattr(mod, class_name)
    trainer = cls()
    return trainer
