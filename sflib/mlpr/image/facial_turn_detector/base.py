from os import path
from abc import abstractmethod, ABCMeta

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

from ....ext.torch.trainer import TorchTrainer
from .... import config
from ....cloud.google import GoogleDriveInterface


class FacialTurnDetectorFeatureExtractor:
    """顔表情ベースのターン検出の特徴抽出器
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
    def calc(self, img_seq_list: list) -> list:
        """画像系列データから特徴量を計算する．

        Args:
          img_seq_list: 画像シーケンスのリスト
             img_seq_list[i] は i 番目のバッチの画像のリスト
        
        Returns:
          torch.Tensor: 特徴ベクトル列のリスト
            リストのサイズは len(img_seq_list) と同じ．
            リストの中身は，形状が(length, feature_dim) である特徴ベクトルのテンソル．
            lengthは各画像列に応じて異なる．
        """
        pass


class FacialTurnDetectorTorchModel(nn.Module):
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

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        """フォワード計算．
        
        Args:
          feat (torch.Tensor): 入力特徴量系列のテンソル．
            形状は (max_length, batch, feat_dim)
        
        Returns:
          torch.Tensor: 出力系列
            形状は (max_length, batch, dim)
            dim は出力の次元数．基本的には2次元．
            （0が非発話中，1が発話中の対数尤度）
        """
        pass


class FacialTurnDetector(metaclass=ABCMeta):
    """FacialTurnDetector（顔情報ベースのターン検出器）の基底クラス
    """

    def __init__(self,
                 feature_extractor: FacialTurnDetectorFeatureExtractor,
                 device=None):
        """
        Args:
          feature_extractor: 特徴抽出器
          device: GPUで実行する場合はここにデバイスを指定する
        """
        self._feature_extractor = feature_extractor
        self._device = device

    @property
    def feature_extractor(self) -> FacialTurnDetectorFeatureExtractor:
        return self._feature_extractor

    @property
    def device(self):
        return self._device

    @property
    @abstractmethod
    def torch_model(self) -> FacialTurnDetectorTorchModel:
        """学習対象のモデルを取得する．
        実際の学習ループの際に利用
        """
        pass

    def get_filename_base(self) -> str:
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
    def convert_result_to_task_softmax(self,
                                       result: torch.Tensor) -> torch.Tensor:
        """forward_core()の結果得られる対数尤度の列を，softmaxをかけて確率値に直す．
        
        Args:
          result (torch.Tensor): forward_core()の結果．
             (max_length, batch, dim)
        
        Returns:
          result と同じサイズのTensor.
          softmaxの計算結果．
        """
        pass

    def predict(self, img_list: list, out_type: str = 'raw'):
        """波形データから予測を行う．デモに利用する．
        
        Args:
          img_list: 画像リストのリスト．
            img_list[i] は i番目のバッチの画像のリスト．
            画像はグレイスケールの np.array (height x width，np.uint8）
          out_type: 出力方法を指定する．デフォルトは 'raw'
             'raw': 対数尤度のPackedSequence（forward_core()の出力そのまま）
             'softmax': 各タスクのソフトマックス値の系列のリスト．
                       convert_result_to_task_softmax() を参照のこと．

        Returns:
          out: (max_output_length, batch, dim) の np.array(np.float)．
                解釈は物によって異なることになる．
          out_length: (batch,) のnp.array(np.int32)．バッチ内の各サンプルの出力長に対応．
        """
        feat = self.feature_extractor.calc(img_list)
        if feat is None:
            return None
        out = self.forward_core(feat)
        # 出力長と特徴量列長は同じ（フレーム同期で出てくるから）
        if out_type == 'raw':
            return out
        if out_type == 'softmax':
            return self.convert_result_to_task_softmax(out)
        raise RuntimeError("Unknown out_type '{}' is given".format(out_type))

    def forward_core(self, feat: torch.Tensor) -> torch.Tensor:
        """
        特徴抽出後のフォワード計算．
        学習の際利用されるので，必ず実装すること．

        Args:
          feat: 特徴量列のテンソル
            形状は，(max_length, batch, dim)
            0パディングされていても，全てについて求める

        Returns:
          対数尤度列のtorch.Tensor
        """
        return self.torch_model(feat)

    def save_model(self, filename):
        self.torch_model.eval()
        torch.save(self.torch_model.state_dict(), filename)

    def load_model(self, filename, map_location=None):
        self.torch_model.eval()
        self.torch_model.load_state_dict(
            torch.load(filename, map_location=map_location))


def calc_accuracy(y, t, l):
    """
    y <- (length, batch, 2)
    t <- (length, batch, 1)
    l <- (batch,)
    """
    length, batch, _ = y.shape
    y = torch.softmax(y, dim=2).argmax(dim=2)  # -> (length, batch)
    t = t.reshape(length, batch)  # -> length,batch
    true = y == t
    total_num = 0
    true_num = 0
    for b in range(batch):
        total_num += int(l[b])
        true_num += int(int(true[:l[b], b].sum()))
    return true_num / total_num


class TorchTrainerForFacialTurnDetector(TorchTrainer):
    def __init__(self,
                 facial_turn_detector: FacialTurnDetector,
                 *args,
                 backprop_len=1000,
                 **kwargs):
        self._facial_turn_detector = facial_turn_detector
        self._backprop_len = backprop_len

        # 入力の自動転送は無効化する
        kwargs.update({'automatic_input_transfer': False})
        super().__init__(self._facial_turn_detector.torch_model, *args,
                         **kwargs)

    def _forward(self, batch, update=True):
        """
        batch[0]は，特徴量テンソルのリスト
          形状 (length, dim) のテンソルが，バッチサイズ分並んでいる．

        batch[1]は，ターゲットのテンソルのリスト
          形状 (length, ) のテンソルが，バッチサイズ分並んでいる．
        （入出力の最大長は一致している＝入力同期で出力されるのが前提）
        フォワード計算，パラメータのアップデートは self._backprop_len 毎に
        行い，その度にコンテクストはデタッチする（リセットではないので注意）
        """
        feat = batch[0]
        target = batch[1]

        # まずリセット
        self._facial_turn_detector.reset()

        # 特徴量列の最大値を求める（これを満たす分だけループが回る）
        max_length = max([len(b) for b in feat])
        # 特徴量の次元数（空の値を作るのに必要．最低長さ1のものがあるという前提）
        feat_dim = feat[0].shape[1]
        target_dim = target[0].shape[1]

        count = 0
        total_loss = 0
        total_length = 0
        total_acc = 0
        while True:
            # 0パディングされた特徴ベクトル系列テンソル sub_batch_tensor を求める．
            # 最長は backprop_len に制約する．
            pos_start = count * self._backprop_len
            pos_end = min((count + 1) * self._backprop_len, max_length)
            sub_feat = []
            sub_target = []
            sub_lengths = []
            for feat, target in zip(*batch):
                if pos_start >= feat.shape[0]:
                    sub_feat.append(
                        torch.zeros(0, feat_dim, dtype=torch.float32))
                    sub_target.append(
                        torch.zeros(0, target_dim, dtype=torch.int64))
                    sub_lengths.append(0)
                elif pos_end >= feat.shape[0]:
                    sub_feat.append(feat[pos_start:, :])
                    sub_target.append(target[pos_start:, :])
                    sub_lengths.append(int(feat.shape[0]) - pos_start)
                else:
                    sub_feat.append(feat[pos_start:pos_end, :])
                    sub_target.append(target[pos_start:pos_end, :])
                    sub_lengths.append(pos_end - pos_start)
            sub_feat_padded = pad_sequence(sub_feat)
            sub_target_padded = pad_sequence(sub_target)
            sub_lengths_tensor = torch.tensor(sub_lengths)

            # print("{} / {}".format(sub_target_padded.sum(), sub_lengths_tensor.sum()))
            # import ipdb; ipdb.set_trace()

            # 必要であればGPUへ転送
            if self._device:
                sub_feat_padded = sub_feat_padded.to(self._device)
                sub_target_padded = sub_target_padded.to(self._device)
                sub_lengths_tensor = sub_lengths_tensor.to(self._device)

            # フォワード計算
            y = self._facial_turn_detector.forward_core(sub_feat_padded)

            # ロスを求める
            loss = self._criterion(y, sub_target_padded, sub_lengths_tensor)

            # （必要なら）アップデートする
            if update:
                self._optimzier.zero_grad()
                loss.backward()
                self._callback_train_before_optimizer_step()
                self._optimzier.step()

            # accuracy
            acc = calc_accuracy(y, sub_target_padded, sub_lengths_tensor)
            # print("{:.2f}".format(acc))

            # ロスを積算する
            sum_length = int(sub_lengths_tensor.sum().detach().cpu().numpy())
            total_loss = (total_loss * total_length + loss.detach() *
                          sum_length) / (total_length + sum_length)
            # import ipdb; ipdb.set_trace()
            total_acc = (total_acc * total_length +
                         acc * sum_length) / (total_length + sum_length)
            # total_acc = total_length + total_acc
            # total_acc = acc
            total_length = total_length + sum_length

            # import ipdb; ipdb.set_trace()

            # デタッチする
            self._facial_turn_detector.detach()

            count += 1

            if pos_end >= max_length:
                break

        # print("")
        # pos = 0
        # num = 0
        # for t in target:
        #     pos += int(t.sum())
        #     num += int(t.shape[0])
        # print("ACC: {:.2f}, P-RATE: {:.2f}, N-RATE {:.2f}".format(
        #     total_acc, pos / num, (num - pos) / num))
        return total_loss


class FacialTurnDetectorTrainer(metaclass=ABCMeta):
    """FacialTurnDetectorTrainerの基底クラス．
    """

    def __init__(self, facial_turn_detector):
        self.facial_turn_detector = facial_turn_detector

    def get_filename_base(self):
        """
        モデルファイルの名前のヒントに使う文字列を取得する
        """
        return self.__class__.__name__

    def get_model_filename(self):
        filename = self.facial_turn_detector.get_filename_base()
        filename += '+' + self.facial_turn_detector.feature_extractor.get_filename_base(
        )
        filename += '+' + self.get_filename_base()
        fullpath = path.join(config.get_package_data_dir(__package__),
                             filename)
        return fullpath

    def get_csv_log_filename(self):
        filename = self.facial_turn_detector.get_filename_base()
        filename += '+' + self.facial_turn_detector.feature_extractor.get_filename_base(
        )
        filename += '+' + self.get_filename_base() + '.csv'
        fullpath = path.join(config.get_package_data_dir(__package__),
                             filename)
        return fullpath

    @abstractmethod
    def build_torch_trainer(self, facial_turn_detector: FacialTurnDetector
                            ) -> TorchTrainerForFacialTurnDetector:
        pass

    def train(self):
        self.torch_trainer = self.build_torch_trainer(
            self.facial_turn_detector)
        self.torch_trainer.train()

    def upload_csv_log(self):
        filename = self.get_csv_log_filename()
        if path.exists(filename):
            g = GoogleDriveInterface(read_only=False)
            g.upload(filename, path.basename(filename), mediaType='text/csv')


def save_facial_turn_detector(trainer: FacialTurnDetectorTrainer,
                              upload=False):
    filename = trainer.get_model_filename()
    trainer.facial_turn_detector.save_model(filename)
    if upload is True:
        g = GoogleDriveInterface(read_only=False)
        g.upload(filename, path.basename(filename))


def load_turn_detector(trainer, download=False, map_location=None):
    filename = trainer.get_model_filename()
    if download is True or not path.exists(filename):
        g = GoogleDriveInterface()
        g.download(path.basename(filename), filename)
    trainer.facial_turn_detector.load_model(filename,
                                            map_location=map_location)
