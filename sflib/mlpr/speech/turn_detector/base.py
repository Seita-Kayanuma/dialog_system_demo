# coding: utf-8
from abc import ABCMeta, abstractmethod
import torch
import torch.nn as nn
from ....ext.torch.trainer import TorchTrainer
from os import path
from ....cloud.google import GoogleDriveInterface
from .... import config
from torch.nn.utils.rnn import PackedSequence, pad_packed_sequence, pack_padded_sequence


class TurnDetectorFeatureExtractor(metaclass=ABCMeta):
    """ターン検出の特徴抽出器
    """

    def get_filename_base(self):
        """モデルファイルの名前のヒントに使う文字列を取得する
        """
        return self.__class__.__name__

    @property
    @abstractmethod
    def feature_dim(self):
        """特徴ベクトルの次元数
        """
        pass

    @property
    @abstractmethod
    def feature_rate(self):
        """特徴量のフレームレート[fps]
        """
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


class TurnDetectorTorchModel(nn.Module):
    """TurnDetectorの中の学習対象の部分．
    特徴抽出後のフォワード計算部分までを表す．
    """

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
            何次元になるかは場合によって異なる．
        """
        pass

    def forward_padded(self, feat: torch.Tensor, lengths: torch.Tensor=None) -> torch.Tensor:
        """フォワード計算のpaddedテンソル版
        """
        pass


class TurnDetector(metaclass=ABCMeta):
    """TurnDetector（ターン検出器）の基底クラス
    """

    def __init__(self,
                 feature_extractor: TurnDetectorFeatureExtractor,
                 device=None):
        """
        Args:
          feature_extractor: 特徴抽出器
          device: GPUで実行する場合はここにデバイスを指定する
        """
        self._feature_extractor = feature_extractor
        self._device = device

    @property
    def feature_extractor(self) -> TurnDetectorFeatureExtractor:
        return self._feature_extractor

    @property
    def device(self):
        return self._device

    @property
    @abstractmethod
    def torch_model(self) -> TurnDetectorTorchModel:
        """学習対象のモデルを取得する．
        実際の学習ループの際に利用
        """
        pass

    def get_filename_base(self):
        """モデルファイルの名前のヒントに使う文字列を取得する
        """
        return self.__class__.__name__

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
        feat = self.feature_extractor.calc(wav_list)
        if feat is None:
            return None
        out = self.forward_core(feat)
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

    def forward_core_padded(self, feat: torch.Tensor, lengths: torch.Tensor=None) -> torch.Tensor:
        """フォワード計算のpaddedテンソル版
        """
        return self.torch_model.forward_padded(feat, lengths)

    def save_model(self, filename):
        self.torch_model.eval()
        torch.save(self.torch_model.state_dict(), filename)

    def load_model(self, filename, map_location=None):
        self.torch_model.eval()
        self.torch_model.load_state_dict(
            torch.load(filename, map_location=map_location))


class TorchTrainerForTurnDetector(TorchTrainer):
    def __init__(self,
                 turn_detector: TurnDetector,
                 *args,
                 backprop_len=1000,
                 **kwargs):
        self._turn_detector = turn_detector
        self._backprop_len = backprop_len

        # 入力の自動転送は必ず無効化する
        kwargs.update({'automatic_input_transfer': False})
        super().__init__(self._turn_detector.torch_model, *args, **kwargs)

    def _forward(self, batch, update=True):
        """バッチの内容が特殊かつ，アップデートの仕方も特殊．
        batch[0] は特徴量列のPackedSequence
        batch[1] は出力情報列のPackedSequence
        
        （入手出力の最大長は一致している＝入力同期で出力されるのが前提）
        フォワード計算，パラメータのアップデートは self._backprop_len 毎に
        行い，その度にコンテクストはデタッチする（リセットではないので注意）
        """
        feat = batch[0]
        target = batch[1]

        # PackedSequenceが長さ0に対応していないという
        # 罠回避のためのパラメタ．いずれは外部からいじれるようにした方がいいかも
        padded = True
        
        self._turn_detector.reset()

        # 非常に負けた感じがするが，現状は pad_packed_sequenceで
        # パディングされた tensor を作って，時間方向に分割していく
        # しかなさそう．
        # 折角メモリを端折れていたのに勿体無い...
        padded_feat, feat_len = pad_packed_sequence(feat)
        max_feat_len = padded_feat.shape[0]
        padded_target, target_len = pad_packed_sequence(target)

        count = 0
        total_loss = 0
        while True:
            # print(count, total_loss)
            # バッチを時間方向にbackprop分の長さに分ける
            pos_start = count * self._backprop_len
            pos_end = min((count + 1) * self._backprop_len, max_feat_len)
            sub_feat = padded_feat[pos_start:pos_end].clone().detach()
            sub_feat_len = (feat_len - pos_start).clone()
            sub_feat_len.apply_(lambda x: 0 if x < 0 else self._backprop_len
                                if x > self._backprop_len else x)
            sub_target = padded_target[pos_start:pos_end, :, :].clone().detach(
            )
            # 必要であればGPUへ転送
            if self._device:
                sub_feat = sub_feat.to(self._device)
                sub_feat_len = sub_feat_len.to(self._device)
                sub_target = sub_target.to(self._device)

            if not padded:
                # フォワード計算
                packed_sub_feat = pack_padded_sequence(sub_feat,
                                                       sub_feat_len,
                                                       enforce_sorted=False)
                y = self._turn_detector.forward_core(packed_sub_feat)
                padded_y, _ = pad_packed_sequence(y)
            else:
                padded_y = self._turn_detector.forward_core_padded(sub_feat, sub_feat_len)

            # ロスを求める
            loss = self._criterion(padded_y, sub_target, sub_feat_len)

            # ロスを積算する
            total_loss = (total_loss * count + loss.detach()) / (count + 1)

            # （必要なら）アップデートする
            if update:
                self._optimzier.zero_grad()
                loss.backward()
                self._callback_train_before_optimizer_step()
                self._optimzier.step()

            # デタッチする
            self._turn_detector.detach()

            count += 1

            # 終了条件
            if pos_end >= max_feat_len:
                break

        # total_loss /= count
        return total_loss


class TurnDetectorTrainer(metaclass=ABCMeta):
    """TurnDetectorTrainerの基底クラス．
    """

    def __init__(self, turn_detector):
        self.turn_detector = turn_detector

    def get_filename_base(self):
        """
        モデルファイルの名前のヒントに使う文字列を取得する
        """
        return self.__class__.__name__

    def get_model_filename(self):
        filename = self.turn_detector.get_filename_base()
        filename += '+' + self.turn_detector.feature_extractor.get_filename_base(
        )
        filename += '+' + self.get_filename_base()
        fullpath = path.join(config.get_package_data_dir(__package__),
                             filename)
        return fullpath

    def get_csv_log_filename(self):
        filename = self.turn_detector.get_filename_base()
        filename += '+' + self.turn_detector.feature_extractor.get_filename_base(
        )
        filename += '+' + self.get_filename_base() + '.csv'
        fullpath = path.join(config.get_package_data_dir(__package__),
                             filename)
        return fullpath

    @abstractmethod
    def build_torch_trainer(self, turn_detector):
        pass

    def train(self):
        self.torch_trainer = self.build_torch_trainer(self.turn_detector)
        self.torch_trainer.train()

    def upload_csv_log(self):
        filename = self.get_csv_log_filename()
        if path.exists(filename):
            g = GoogleDriveInterface(read_only=False)
            g.upload(filename, path.basename(filename), mediaType='text/csv')


def save_turn_detector(trainer, upload=False):
    filename = trainer.get_model_filename()
    trainer.turn_detector.save_model(filename)
    if upload is True:
        g = GoogleDriveInterface(read_only=False)
        g.upload(filename, path.basename(filename))


def load_turn_detector(trainer, download=False, map_location=None):
    filename = trainer.get_model_filename()
    if download is True or not path.exists(filename):
        g = GoogleDriveInterface()
        g.download(path.basename(filename), filename)
    trainer.turn_detector.load_model(filename, map_location=map_location)
