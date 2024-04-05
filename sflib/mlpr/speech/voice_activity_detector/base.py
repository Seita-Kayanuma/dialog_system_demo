# シンプルなVAD
#  1次遅れタイミング検出のための u(t) を計算するためのもの
from abc import ABCMeta, abstractmethod
import torch
import torch.nn as nn
from ....ext.torch.trainer import TorchTrainer
from os import path
from ....cloud.google import GoogleDriveInterface
from .... import config
from torch.nn.utils.rnn import \
    PackedSequence, pad_packed_sequence, pack_padded_sequence, pack_sequence, pad_sequence
import re
import glob
import numpy as np


class VoiceActivityDetectorFeatureExtractor(metaclass=ABCMeta):
    """ターン検出の特徴抽出器
    """

    def __init__(self):
        # class name check
        m = re.match(r'VoiceActivityDetectorFeatureExtractor(\d+)',
                     self.__class__.__name__)
        if m is None:
            raise RuntimeError('class name should match with' +
                               'r"VoiceActivityDetectorFeatureExtractor\\d+"')
        self.__number = int(m[1])

    @property
    def filename_base(self):
        """モデルファイルの名前のヒントに使う文字列"""
        return 'F{:02d}'.format(self.__number)

    @property
    @abstractmethod
    def feature_dim(self):
        """特徴ベクトルの次元数
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def feature_rate(self):
        """特徴量のフレームレート[fps]
        """
        raise NotImplementedError

    @property
    def device(self):
        return None

    def to(self, device):
        pass

    def reset(self):
        """状態をリセットする．
        新しいバッチを入力する際などに呼ぶ必要がある．
        """
        pass

    def detach(self):
        """コンテクストをデタッチする．
        RNN系のニューラルネットを内部に持つ場合に，
        バックワード計算の対象から外すために呼ぶ必要がある．
        """
        pass

    @abstractmethod
    def calc(self, wav_list: list) -> list:
        """波形データから特徴量を計算する．

        Args:
          wav_list: 波形データのリスト．
             wav_list[i] は i 番目のバッチの波形データ．
             波形データは任意の長さの1次元 numpy.array(np.int16)．
        
        Returns:
           list: 抽出された特徴ベクトルの列．
             shapeは(L, D)
        """
        raise NotImplementedError


class VoiceActivityDetectorTorchModel(nn.Module):
    """VoiceActivityDetectorの中の学習対象の部分．
    特徴抽出後のフォワード計算部分までを表す．
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

    def reset_context(self):
        """コンテクストをリセットする
        """
        pass

    def detach_context(self):
        """コンテクストをデタッチする（値は残すが，逆伝播の対象から外す）
        """
        pass

    def forward(self, feat: PackedSequence) -> PackedSequence:
        """フォワード計算．
        
        Args:
          feat (PackedSequence): 入力特徴量系列のPackedSequence
        
        Returns:
          PackedSequence: 対数尤度分布列のPackedSequence
            何次元になるかは何を出力にするかによって変化する．
        """
        pass

    def forward_padded(self, feat: torch.Tensor,
                       lengths: torch.Tensor = None) -> torch.Tensor:
        """フォワード計算のpaddedテンソル版
        """
        if lengths is None:
            length, num_batches, dim = feat.shape
            lengths = torch.Tensor([length] * num_batches)
        lengths[lengths == 0] = 1
        feat_packed = pack_padded_sequence(feat, lengths)
        out_packed = self.forward(feat_packed)
        out_padded, _ = pad_packed_sequence(out_packed)
        return out_padded


class VoiceActivityDetector(metaclass=ABCMeta):
    """VoiceActivityDetector（ターン検出器）の基底クラス
    """

    DEFAULT_TRAINER_NUMBER = 1
    """int:
    学習器番号のデフォルト値．
    """

    DEFAULT_FEATURE_EXTRACTOR_NUMBER = 1
    """int:
    特徴抽出器番号のデフォルト値．
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
        """
        Args:
          trainer_number: 学習器番号
          feature_extractor_number: 特徴抽出器番号
          feature_extractor_construct_args: 特徴抽出器の構築時に利用する引数
            リスト形式の引数と，辞書形式の引数のタプル
        """
        # class name check
        m = re.match(r'VoiceActivityDetector(\d+)', self.__class__.__name__)
        if m is None:
            raise RuntimeError('class name should match with' +
                               'r"VoiceActivityDetector\\d+"')
        self.__number = int(m[1])
        self.__trainer_number = trainer_number
        self._feature_extractor = construct_feature_extractor(
            feature_extractor_number, feature_extractor_construct_args)

    @property
    @abstractmethod
    def torch_model(self) -> VoiceActivityDetectorTorchModel:
        """学習対象のモデルを取得する．
        実際の学習ループの際に利用"""
        raise NotImplementedError

    @property
    def feature_extractor(self) -> VoiceActivityDetectorFeatureExtractor:
        return self._feature_extractor

    @property
    def device(self):
        return self.torch_model.device

    def to(self, device):
        self.torch_model.to(device)
        self.feature_extractor.to(device)

    @property
    def filename_base(self):
        return 'VAD{:02d}T{:02d}{}'.format(
            self.__number, self.__trainer_number,
            self.feature_extractor.filename_base)

    def reset(self):
        """状態をリセットする．
        """
        self.feature_extractor.reset()
        self.torch_model.reset_context()

    def detach(self):
        """コンテクストをデタッチする．
        LSTMのバックプロパゲーションを打ち切る場合に利用．
        """
        self.torch_model.detach_context()

    @abstractmethod
    def convert_result_to_task_softmax(self, result: PackedSequence) -> list:
        """forward_core()の結果得られる対数尤度の列を，各タスク毎に分けて
        softmaxをかけて確率値に直す．

        Args:
          result (PackedSequence): forward_core()の結果．
        
        Returns:
          list: 
            out[i]は，各バッチに対応する．
            out[i][j] は i 番目のバッチの j 番目のタスクのソフトマックス値のTensor.
            Tensorのサイズは (L, Cj) で，Lが系列長，Cjが当該タスクのクラス数
        """
        pass

    def predict(self, wav_list: list, out_type: str = 'raw'):
        """波形データからの予測を行う．デモに利用する．

        Args:
          wav_list: 波形データのリスト．
             wav_list[i] は i 番目のバッチの波形データ．
             波形データは任意の長さの1次元 numpy.array(np.int16)
          out_type: 出力方法を指定する．デフォルトは 'raw'
             'raw': 対数尤度のPackedSequence（forward_core()の出力そのまま）
             'softmax': 各タスクのソフトマックス値の系列のリスト．
                       convert_result_to_task_softmax() を参照のこと．

        Returns:
          out: (max_output_length, batch, dim) の np.array(np.float)．
                解釈は物によって異なることになる．
          out_length: (batch,) のnp.array(np.int32)．バッチ内の各サンプルの出力長に対応．
        """
        feat_list = self.feature_extractor.calc(wav_list)
        if feat_list is None:
            return None
        feat = pad_sequence(feat_list)
        if feat.shape[0] == 0:
            return None
        out = self.forward_core_padded(feat)
        out = pack_padded_sequence(out, [len(f) for f in feat_list])
        # 出力長と特徴量列長は同じ（フレーム同期で出てくるから）
        if out_type == 'raw':
            return out
        if out_type == 'softmax':
            return self.convert_result_to_task_softmax(out)
        raise RuntimeError("Unknown out_type '{}' is given".format(out_type))

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

    def forward_core_padded(self,
                            feat: torch.Tensor,
                            lengths: torch.Tensor = None) -> torch.Tensor:
        """フォワード計算のpaddedテンソル版
        """
        return self.torch_model.forward_padded(feat, lengths)
        # L = feat.shape[0]
        # out_list = []
        # for i in range(L):
        #     out_list.append(self.torch_model.forward_padded(feat[i:(i+1)]))
        # # import ipdb; ipdb.set_trace()
        # return torch.cat(out_list, dim=0)

    # ---- 以下は学習，保存，読み込み関係のメソッド
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

    def train_voice_activity_detector(self, *args, **kwargs):
        trainer = construct_trainer(self.__trainer_number, *args, **kwargs)
        trainer.train(self)


class _FeatureExtractorWrapper:
    """Trainerのための特徴抽出器のラッパー
    """

    def __init__(self,
                 feature_extractor: VoiceActivityDetectorFeatureExtractor,
                 wav_list: list):
        self.feature_extractor = feature_extractor
        self.feature_extractor.reset()
        self.wav_list = wav_list

        # 音声をどこまで入力したか
        self._wav_ptr = 0
        # 出力していない分の特徴量キャッシュのリスト
        self._res_feat_list = [
            torch.zeros((0, self.feature_extractor.feature_dim),
                        dtype=torch.float32).to(feature_extractor.device)
        ] * len(wav_list)
        # res_feat_list内の最長要素の長さ
        self._max_res_feat_len = 0
        # 各wavの長さ
        self._wav_lengths = [len(wav) for wav in wav_list]
        # wav最大長
        self._max_wav_len = max(self._wav_lengths)

        # 音声波形を一度に何サンプルずつ入力するか
        self._wav_input_window = 1600

    @property
    def finished(self) -> bool:
        return self._wav_ptr == self._max_wav_len

    def get_features(self, num_frames: int) -> (PackedSequence, torch.Tensor):
        """（最大）指定したフレーム分の特徴量を取得する
        
        Args:
          num_frames(int): 取得したい（最大フレーム数）
        
        Returns:
          PackedSequence: 特徴量のPackedSequence
          torch.Tensor: 特徴列の長さを並べたベクトル
            なお，deviceはCPU限定．

        Note:
          
        """
        while self._wav_ptr < self._max_wav_len and self._max_res_feat_len < num_frames:
            in_wav_list = []
            wav_start = self._wav_ptr
            wav_end = self._wav_ptr + self._wav_input_window
            if wav_end > self._max_wav_len:
                wav_end = self._max_wav_len
            for wav, wav_len in zip(self.wav_list, self._wav_lengths):
                if wav_start > wav_len:
                    in_wav_list.append(np.zeros(0, dtype=np.uint16))
                elif wav_end > wav_len:
                    in_wav_list.append(wav[wav_start:])
                else:
                    in_wav_list.append(wav[wav_start:wav_end])
            feat_list = self.feature_extractor.calc(in_wav_list)
            self._wav_ptr = wav_end
            if feat_list is None:
                continue
            res_feat_list = []
            max_res_feat_len = 0
            for res_feat, feat in zip(self._res_feat_list, feat_list):
                new_res_feat = torch.cat([res_feat, feat], dim=0)
                if new_res_feat.shape[0] > max_res_feat_len:
                    max_res_feat_len = new_res_feat.shape[0]
                res_feat_list.append(new_res_feat)
            self._res_feat_list = res_feat_list
            self._max_res_feat_len = max_res_feat_len
        # import ipdb; ipdb.set_trace()
        feat_list = []
        feat_len_list = []
        res_feat_list = []
        max_res_feat_len = 0
        for res_feat in self._res_feat_list:
            if res_feat.shape[0] > num_frames:
                feat = res_feat[:num_frames]
                feat_len = num_frames
                res_feat = res_feat[num_frames:]
            elif res_feat.shape[0] == 0:
                # empty feature, dummy feature is appended
                feat = torch.zeros((1, self.feature_extractor.feature_dim),
                                   dtype=torch.float32)
                feat = feat.to(self.feature_extractor.device)
                feat_len = 0
            else:
                feat = res_feat
                feat_len = feat.shape[0]
                res_feat = torch.zeros((0, self.feature_extractor.feature_dim),
                                       dtype=torch.float32)
                res_feat = res_feat.to(self.feature_extractor.device)
            feat_list.append(feat)
            feat_len_list.append(feat_len)
            res_feat_list.append(res_feat)
            if res_feat.shape[0] > max_res_feat_len:
                max_res_feat_len = res_feat.shape[0]
        self._res_feat_list = res_feat_list
        self._max_res_feat_len = max_res_feat_len
        feat_packed = pack_sequence(feat_list, enforce_sorted=False)
        lengths = torch.tensor(feat_len_list, dtype=torch.int16)
        # import ipdb; ipdb.set_trace()
        return feat_packed, lengths


class TorchTrainerForVoiceActivityDetector(TorchTrainer):
    """VoiceActivityDetector用のTorchTrainer"""

    def __init__(self,
                 voice_activity_detector: VoiceActivityDetector,
                 *args,
                 backprop_len=500,
                 **kwargs):
        self._voice_activity_detector = voice_activity_detector
        self._backprop_len = backprop_len

        # 入力の自動転送は必ず無効化する
        kwargs.update({'automatic_input_transfer': False})
        super().__init__(
            self._voice_activity_detector.torch_model, *args, **kwargs)

    def _forward(self, batch, update=True):
        """バッチの内容が特殊かつ，アップデートの仕方も特殊．
        batch[0] はwavのリスト
        batch[1] はtarget情報のリスト
        
        （入出力の最大長は一致している＝入力同期で出力されるのが前提）
        フォワード計算，パラメータのアップデートは self._backprop_len 毎に
        行い，その度にコンテクストはデタッチする（リセットではないので注意）
        """
        wavs = batch[0]
        targets = batch[1]

        self._voice_activity_detector.reset()
        targets = [t.to(self._voice_activity_detector.device) for t in targets]

        feature_extractor_wrapper = _FeatureExtractorWrapper(
            self._voice_activity_detector.feature_extractor, wavs)

        count = 0
        total_loss = 0
        offset_target = 0
        while not feature_extractor_wrapper.finished:
            feat_packed, lengths = \
                feature_extractor_wrapper.get_features(self._backprop_len)
            y_packed = self._voice_activity_detector.forward_core(feat_packed)
            y_padded, _ = pad_packed_sequence(y_packed)
            sub_targets = [
                t[offset_target:(offset_target + self._backprop_len)]
                for t in targets
            ]
            sub_target = pad_sequence(sub_targets)
            lengths = lengths.detach().cpu().numpy().tolist()
            cri_lengths = [
                min(t.shape[0], yl) for t, yl in zip(sub_targets, lengths)
            ]
            
            # ロスを求める
            loss = self._criterion(y_padded, sub_target, cri_lengths)

            if loss != 0:
                # ロスを積算する
                total_loss = (total_loss * count + loss.detach()) / (count + 1)

                # （必要なら）アップデートする
                if update:
                    self._optimzier.zero_grad()
                    loss.backward()
                    self._callback_train_before_optimizer_step()
                    self._optimzier.step()
                    ###
                    # from torchviz import make_dot
                    # dt = make_dot(loss)
                    # dt.save('hoge.dot')
                    # import ipdb; ipdb.set_trace()
                    ###

            # デタッチする
            self._voice_activity_detector.feature_extractor.detach()
            self._voice_activity_detector.detach()

            count += 1
            offset_target += self._backprop_len

        # total_loss /= count
        return total_loss


class VoiceActivityDetectorTrainer(metaclass=ABCMeta):
    """VoiceActivityDetectorTrainerの基底クラス．
    """

    def __init__(self):
        pass

    @abstractmethod
    def build_torch_trainer(self, phone_type_writer):
        pass

    def train(self, voice_activity_detector: VoiceActivityDetector):
        self.torch_trainer = self.build_torch_trainer(voice_activity_detector)
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
    module_name = "sflib.mlpr.speech.voice_activity_detector." + \
                  "feature_extractor{:04d}".format(feature_extractor_number)
    class_name = "VoiceActivityDetectorFeatureExtractor" + \
                 "{:04d}".format(feature_extractor_number)
    # import importlib
    # mod = importlib.import_module(module_name)
    # cls = getattr(mod, class_name)
    # args, kwargs = feature_extractor_construct_args
    # feature_extractor = cls(*args, **kwargs)
    from ....util.cls import construct
    feature_extractor = construct(
        module_name, class_name, feature_extractor_construct_args)
    return feature_extractor


def construct_voice_activity_detector(voice_activity_detector_number,
                                      trainer_number,
                                      feature_extractor_number,
                                      feature_extractor_construct_args=(
                                          [],
                                          {},
                                      )):
    module_name = "sflib.mlpr.speech.voice_activity_detector." + \
        "voice_activity_detector{:04d}".format(voice_activity_detector_number)
    class_name = "VoiceActivityDetector{:04d}".format(
        voice_activity_detector_number)
    # import importlib
    # mod = importlib.import_module(module_name)
    # cls = getattr(mod, class_name)
    # voice_activity_detector = cls(trainer_number, feature_extractor_number,
    #                               feature_extractor_construct_args)
    from ....util.cls import construct
    voice_activity_detector = construct(
        module_name, class_name, [trainer_number, feature_extractor_number,
                                  feature_extractor_construct_args])
    return voice_activity_detector


def construct_trainer(trainer_number, *args, **kwargs):
    module_name = "sflib.mlpr.speech.voice_activity_detector." + \
        "trainer{:04d}".format(trainer_number)
    class_name = "VoiceActivityDetectorTrainer{:04d}".format(trainer_number)
    # import importlib
    # mod = importlib.import_module(module_name)
    # cls = getattr(mod, class_name)
    # trainer = cls(*args, **kwargs)
    from ....util.cls import construct
    trainer = construct(module_name, class_name, (args, kwargs))
    return trainer
