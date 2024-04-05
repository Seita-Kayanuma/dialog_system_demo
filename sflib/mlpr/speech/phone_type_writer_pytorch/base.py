# coding: utf-8
import numpy as np
from ....cloud.google import GoogleDriveInterface
from os import path
from .... import config
from abc import ABCMeta, abstractmethod
from ....ext.torch.trainer import TorchTrainer
import torch
import torch.nn as nn
from torch.nn.utils.rnn import PackedSequence, pad_packed_sequence

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

    def get_filename_base(self):
        """モデルファイルの名前のヒントに使う文字列を取得する
        """
        return self.__class__.__name__

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
    def calc(self, wav_list: list) -> PackedSequence:
        """波形データから特徴量を計算する．
        
        Args:
          wav_list: 波形データのリスト．
             wav_list[i] は i 番目のバッチの波形データ．
             波形データは任意の長さの1次元 numpy.array(np.int16)
        
        Returns:
          PackedSequence: 特徴ベクトルのPackedSequence．
        """
        pass


class PhoneTypeWriterTorchModel(nn.Module):
    """PhoneTypeWriterが持つニューラルネット
    """

    def __init__(self):
        super(PhoneTypeWriterTorchModel, self).__init__()

    @property
    @abstractmethod
    def hidden_feature_dim(self):
        """隠れ層特徴量の次元数
        """
        pass

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
        pass

    @abstractmethod
    def predict_log_probs_with_hidden_feature(self, feat: PackedSequence
                                              ) -> PackedSequence:
        """calc_hidden_featureで計算された隠れ層の特徴量から，
        出力である音素の対数尤度分布列を計算する．
        
        Args:
          feat (PackedSequence): 隠れ層特徴量列のPackedSequence
        
        Returns:
          PackedSequence: 対数尤度分布列のPackedSequence
        """
        pass

    def forward(self, feat: PackedSequence) -> PackedSequence:
        """フォワード計算．
        内部で calc_hidden_feature -> predict_log_probs_with_hidden_feature
        の順で呼び出される．
        """
        h = self.calc_hidden_feature(feat)
        return self.predict_log_probs_with_hidden_feature(h)


class PhoneTypeWriter(metaclass=ABCMeta):
    """
    PhoneTypeWriter（音素タイプライタ）の基底クラス
    """

    def __init__(self,
                 feature_extractor: PhoneTypeWriterFeatureExtractor,
                 device=None):
        self._feature_extractor = feature_extractor
        self._device = device
        
    @property
    def feature_extractor(self) -> PhoneTypeWriterFeatureExtractor:
        return self._feature_extractor

    @property
    def device(self):
        return self._device

    @device.setter
    def device(self, device):
        self._device = device
        self.torch_model.to(self._device)

    @property
    @abstractmethod
    def torch_model(self) -> PhoneTypeWriterTorchModel:
        pass

    @property
    def hidden_feature_dim(self) -> int:
        return self.torch_model.hidden_feature_dim

    def get_filename_base(self):
        """モデルファイルの名前のヒントに使う文字列を取得する
        """
        return self.__class__.__name__

    def reset(self):
        """状態をリセットする"""
        self.feature_extractor.reset()
        self.torch_model.reset_context()

    def detach(self):
        """コンテクストをデタッチする．
        LSTMのバックプロパゲーションを打ち切る場合に利用．
        """
        self.torch_model.detach_context()

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
        feat = self.feature_extractor.calc(wav_list)
        if feat is None:
            return [[] * len(wav_list)]
        packed_log_probs = self.forward_core(feat)
        log_probs, log_probs_len = pad_packed_sequence(packed_log_probs)
        log_probs = log_probs.detach().cpu().numpy()
        log_probs_len = log_probs_len.detach().cpu().numpy().tolist()

        result = []
        for i, l in enumerate(log_probs_len):
            lp = log_probs[:l, i, :]
            idx_list = lp.argmax(axis=1).tolist()
            result.append(convert_id_to_phone(idx_list))
        return result

    def calc_hidden_feature_from_extracted_feature(self, feat: PackedSequence
                                                   ) -> PackedSequence:
        """特徴抽出器で抽出済のデータから隠れ層特徴量を抽出する．
        隠れ層特徴量は後段の識別器等に使う．"""
        return self.torch_model.calc_hidden_feature(feat)

    def calc_hidden_feature_from_wav(self, wav_list) -> PackedSequence:
        """wavから隠れ層特徴量を抽出する．デモに利用"""
        feat = self.feature_extractor.calc(wav_list)
        return self.calc_hidden_feature_from_extracted_feature(feat)

    def calc_log_probs_from_wav(self, wav_list) -> (np.array, np.array):
        """wavから対数尤度列を計算．デモに利用

        Returns:
          np.array: パッドされた対数尤度列のバッチ．(L, B, C)
            Lは最長のシーケンス長
            Bはバッチサイズ
            Cはカテゴリ数
          np.array: 各バッチのシーケンス長．(B,)
        """
        feat = self.feature_extractor.calc(wav_list)
        log_probs = self.forward_core(feat)
        padded_log_probs, log_probs_len = pad_packed_sequence(log_probs)
        padded_log_probs = padded_log_probs.detach().cpu().numpy()
        log_probs_len = log_probs_len.detach().cpu().numpy()
        return padded_log_probs, log_probs_len

    def forward_core(self, feat: PackedSequence) -> PackedSequence:
        """
        特徴抽出後のフォワード計算．
        学習の際利用されるので，必ず実装すること．

        Args:
          feat: 特徴量列のPackedSequence

        Returns:
          対数尤度列のPackedSequence
        """
        return self.torch_model(feat)

    def save_model(self, filename):
        self.torch_model.eval()
        torch.save(self.torch_model.state_dict(), filename)

    def load_model(self, filename, map_location=None):
        self.torch_model.eval()
        self.torch_model.load_state_dict(
            torch.load(filename, map_location=map_location))


def packed_sequence_to(ps: PackedSequence, device) -> PackedSequence:
    """PackedSequenceをデバイスに転送する"""
    # return PackedSequence(ps.data.to(device),
    #                       ps.batch_sizes.to(device),
    #                       ps.sorted_indices.to(device),
    #                       ps.unsorted_indices.to(device))
    return PackedSequence(ps.data.to(device),
                          ps.batch_sizes,
                          ps.sorted_indices.to(device),
                          ps.unsorted_indices.to(device))


class TorchTrainerForPhoneTypeWriter(TorchTrainer):
    """PhoneTypeWriter用のTorchTrainer"""

    def __init__(self, phone_type_writer: PhoneTypeWriter, *args, **kwargs):
        self._phone_type_writer = phone_type_writer

        # 入力の自動転送は必ず無効化する
        kwargs.update({'automatic_input_transfer': False})
        super(TorchTrainerForPhoneTypeWriter,
              self).__init__(self._phone_type_writer.torch_model, *args,
                             **kwargs)

    def _forward(self, batch, update=True):
        """
        バッチの内容が特殊
        batch[0] は特徴量列のPackedSequence
        batch[1] は音素番号列のPackedSequence
        """
        feat, target = batch

        if self._device:
            feat = packed_sequence_to(feat, self._device)
            target = packed_sequence_to(target, self._device)
            # feat = feat.to(self._device)
            # target = target.to(self._device)

        self._phone_type_writer.reset()
        log_probs = self._phone_type_writer.forward_core(feat)
        padded_log_probs, log_probs_len = pad_packed_sequence(log_probs)
        padded_target, target_len = pad_packed_sequence(target, batch_first=True)
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

    def __init__(self, phone_type_writer):
        self.phone_type_writer = phone_type_writer

    def get_filename_base(self):
        """モデルファイルの名前のヒントに使う文字列を取得する
        """
        return self.__class__.__name__

    def get_model_filename(self):
        filename = self.phone_type_writer.get_filename_base()
        filename += '+' + self.phone_type_writer.feature_extractor.get_filename_base(
        )
        filename += '+' + self.get_filename_base()
        fullpath = path.join(config.get_package_data_dir(__package__),
                             filename)
        return fullpath

    def get_csv_log_filename(self):
        filename = self.phone_type_writer.get_filename_base()
        filename += '+' + self.phone_type_writer.feature_extractor.get_filename_base(
        )
        filename += '+' + self.get_filename_base() + '.csv'
        fullpath = path.join(config.get_package_data_dir(__package__),
                             filename)
        return fullpath

    @abstractmethod
    def build_torch_trainer(self, phone_type_writer):
        pass

    def train(self):
        self.torch_trainer = self.build_torch_trainer(self.phone_type_writer)
        self.torch_trainer.train()

    def upload_csv_log(self):
        filename = self.get_csv_log_filename()
        if path.exists(filename):
            g = GoogleDriveInterface(read_only=False)
            g.upload(filename, path.basename(filename), mediaType='text/csv')


def save_phone_type_writer(trainer, upload=False):
    filename = trainer.get_model_filename()
    trainer.phone_type_writer.save_model(filename)
    if upload is True:
        g = GoogleDriveInterface(read_only=False)
        g.upload(filename, path.basename(filename))


def load_phone_type_writer(trainer, download=False, map_location=None):
    filename = trainer.get_model_filename()
    if download is True or not path.exists(filename):
        g = GoogleDriveInterface()
        g.download(path.basename(filename), filename)
    trainer.phone_type_writer.load_model(filename, map_location=map_location)


# ----
def construct_feature_extractor_with_autoencoder(
        feature_extractor_version, autoencoder_version,
        autoencoder_trainer_module_postfix, autoencoder_trainer_class_postfix):
    module_name = "sflib.mlpr.speech.phone_type_writer_pytorch.feature_autoencoder_%04d" \
        % feature_extractor_version
    class_name = "PhoneTypeWriterFeatureExtractorAutoEncoder%04d" \
        % feature_extractor_version
    import importlib
    mod = importlib.import_module(module_name)
    cls = getattr(mod, class_name)
    feature_extractor = cls(autoencoder_version,
                            autoencoder_trainer_module_postfix,
                            autoencoder_trainer_class_postfix)
    return feature_extractor


def construct_phone_type_writer(phone_type_writer_version, feature_extractor):
    module_name = "sflib.mlpr.speech.phone_type_writer_pytorch.phone_type_writer_%04d" \
        % phone_type_writer_version
    class_name = "PhoneTypeWriter%04dPyTorch" % phone_type_writer_version
    import importlib
    mod = importlib.import_module(module_name)
    cls = getattr(mod, class_name)
    phone_type_writer = cls(feature_extractor)
    return phone_type_writer


def construct_phone_type_writer_trainer(trainer_module_postfix,
                                        trainer_class_postfix,
                                        phone_type_writer):
    module_name = "sflib.mlpr.speech.phone_type_writer_pytorch.trainer_%s" \
        % trainer_module_postfix
    class_name = "PhoneTypeWriterTrainer%s" % trainer_class_postfix
    import importlib
    mod = importlib.import_module(module_name)
    cls = getattr(mod, class_name)
    trainer = cls(phone_type_writer)
    return trainer
