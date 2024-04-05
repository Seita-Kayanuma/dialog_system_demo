# 1次遅れタイミング検出
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
from typing import Tuple


class InputCalculator(metaclass=ABCMeta):
    """u(t)の計算器の基底クラス
    """
    def __init__(self):
        # class name check
        m = re.match(r'InputCalculator(\d+)',
                     self.__class__.__name__)
        if m is None:
            raise RuntimeError('class name should match with' +
                               'r"InputCalculator\\d+"')
        self.__number = int(m[1])
        
    @property
    def filename_base(self):
        """モデルファイルの名前のヒントに使う文字列"""
        return 'IC{:02d}'.format(self.__number)

    @property
    @abstractmethod
    def feature_dim(self):
        """特徴ベクトルの次元数
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def feature_rate(self):
        """フレームレート[fps]
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def ut_dim(self):
        """u(t)の次元．基本は1次元だが，ものによっては複数次元になる
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
    def calc(self, wav_list: list) -> Tuple[list, list]:
        """波形データから特徴量を計算する．

        Args:
          wav_list: 波形データのリスト．
             wav_list[i] は i 番目のバッチの波形データ．
             波形データは任意の長さの1次元 numpy.array(np.int16)．
        
        Returns:
           list: 計算された入力u(t)．
             shapeは(L, 1)
           list: 抽出された特徴ベクトルの列．
             shapeは(L, D)
        """
        raise NotImplementedError
    

class TurnDetector1dTorchModel(nn.Module):
    """TurnDetector1dの中の学習対象（y(t)算出まで）のモデルを表す．
    入力u(t)計算および特徴抽出後のフォワード計算部分までを表す．
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

    def forward(self, ut: PackedSequence,
                feat: PackedSequence,
                st: PackedSequence = None) -> PackedSequence:
        """フォワード計算．
        
        Args:
          ut (PackedSequence): u(t)のPackedSequence
          feat (PackedSequence): 入力特徴量系列のPackedSequence
        
        Returns:
          PackedSequence: y(t), alpha(t), alpha(t)（ロック前）の三つ組の
            PackedSequence．
        """
        pass

    def forward_padded(self,
                       ut: torch.Tensor,
                       feat: torch.Tensor,
                       lengths: torch.Tensor = None,
                       st: torch.Tensor = None) -> torch.Tensor:
        """フォワード計算のpaddedテンソル版
        """
        if lengths is None:
            length, num_batches, dim = feat.shape
            lengths = torch.Tensor([length] * num_batches)
        lengths[lengths == 0] = 1
        ut_packed = pack_padded_sequence(ut, lengths, enforce_sorted=False)
        feat_packed = pack_padded_sequence(feat, lengths, enforce_sorted=False)
        if st is not None and st.shape[0] >= ut.shape[0]:
            st_packed = pack_padded_sequence(st, lengths)
        else:
            st_packed = None
        out_packed = self.forward(ut_packed, feat_packed, st_packed)
        out_padded, _ = pad_packed_sequence(out_packed)
        return out_padded


class TurnDetector1d(metaclass=ABCMeta):
    """TurnDetector1d（1次遅れ系ターン検出器）の基底クラス
    """

    DEFAULT_TRAINER_NUMBER = 1
    """int:
    学習器番号のデフォルト値．
    """

    DEFAULT_INPUT_CALCULATOR_NUMBER = 1
    """int:
    特徴抽出器番号のデフォルト値．
    """

    DEFAULT_INPUT_CALCULATOR_CONSTRUCT_ARGS = \
        ([],
         {'voice_activity_detector_number': 1,
          'voice_activity_detector_feature_extractor_number': 3,
          'autoencoder_number': 12,
          'autoencoder_tariner_number': 13,
          'autoencoder_model_version': 2},)
    """tuple:
    特徴抽出器コンストラクタ引数のデフォルト値
    """

    def __init__(
            self,
            trainer_number=DEFAULT_TRAINER_NUMBER,
            input_calculator_number=DEFAULT_INPUT_CALCULATOR_NUMBER,
            input_calculator_construct_args=DEFAULT_INPUT_CALCULATOR_CONSTRUCT_ARGS
    ):
        """
        Args:
          trainer_number: 学習器番号
          input_calculator_number: 入力計算器の番号
          input_calculator_construct_args: 入力計算器の構築時に利用する引数
            リスト形式の引数と，辞書形式の引数のタプル
        """
        # class name check
        m = re.match(r'TurnDetector1d(\d+)', self.__class__.__name__)
        if m is None:
            raise RuntimeError('class name should match with' +
                               'r"TurnDetector1d\\d+"')
        self.__number = int(m[1])
        self.__trainer_number = trainer_number
        self._input_calculator = construct_input_calculator(
            input_calculator_number, input_calculator_construct_args)

    @property
    @abstractmethod
    def torch_model(self) -> TurnDetector1dTorchModel:
        """学習対象のモデルを取得する．
        実際の学習ループの際に利用"""
        raise NotImplementedError

    @property
    def input_calculator(self) -> InputCalculator:
        return self._input_calculator

    @property
    def device(self):
        return self.torch_model.device

    def to(self, device):
        self.torch_model.to(device)
        self.input_calculator.to(device)

    @property
    def filename_base(self):
        return 'TD1D{:02d}T{:02d}{}'.format(
            self.__number, self.__trainer_number,
            self.input_calculator.filename_base)
    
    def reset(self):
        """状態をリセットする．
        """
        self.input_calculator.reset()
        self.torch_model.reset_context()

    def detach(self):
        """コンテクストをデタッチする．
        LSTMのバックプロパゲーションを打ち切る場合に利用．
        """
        self.torch_model.detach_context()

    def predict(self, wav_list: list, st_list: list = None):
        """波形データからの予測を行う．デモに利用する．

        Args:
          wav_list: 波形データのリスト．
             wav_list[i] は i 番目のバッチの波形データ．
             波形データは任意の長さの1次元 numpy.array(np.int16)

        Returns:
          out: (max_output_length, batch, 4) の np.array(np.float)．
               最後の3次元は，y(t), alpha(t), alpha(t)(ロック前), u(t)
          out_length: (batch,) のnp.array(np.int32)．バッチ内の各サンプルの出力長に対応．
        """
        ut_list, feat_list = self.input_calculator.calc(wav_list)
        if feat_list is None:
            return None
        ut = pad_sequence(ut_list)
        feat = pad_sequence(feat_list)
        if st_list is not None:
            st_list = [torch.tensor(s, dtype=torch.float32).unsqueeze(1).to(self.device) for s in st_list] 
            st = pad_sequence(st_list)
        else:
            st = None
        if feat.shape[0] == 0:
            return None
        out = self.forward_core_padded(ut, feat, None, st)
        out = torch.cat([out, ut], dim=2)
        out_np = out.detach().clone().cpu().numpy()
        feat_len = np.array([len(f) for f in feat], dtype=np.int32)
        return out_np, feat_len
    
    def forward_core(self,
                     ut: PackedSequence,
                     feat: PackedSequence,
                     st: PackedSequence = None) -> PackedSequence:
        """
        特徴抽出後のフォワード計算．
        学習の際利用されるので，必ず実装すること．

        Args:
          ut: u(t)のPackedSequence
          feat: 特徴量列のPackedSequence
          st: 学習時にy(t)をリセットするためのシステム発話状態
              （1 -> 0 になった時にリセットする）

        Returns:
          y(t), alpha(t), alphat(t)(ロック前) のベクトルのPackedSequence
        """
        return self.torch_model(ut, feat, st)

    def forward_core_padded(self,
                            ut: torch.Tensor,
                            feat: torch.Tensor,
                            lengths: torch.Tensor = None,
                            st: torch.Tensor = None) -> torch.Tensor:
        """フォワード計算のpaddedテンソル版
        """
        return self.torch_model.forward_padded(ut, feat, lengths, st)
    
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

    def train_turn_detector(self, *args, **kwargs):
        trainer = construct_trainer(self.__trainer_number, *args, **kwargs)
        trainer.train(self)


class _InputCalculatorWrapper:
    """Trainerのための特徴抽出器のラッパー
    """

    def __init__(self,
                 input_calculator: InputCalculator,
                 wav_list: list):
        self.input_calculator = input_calculator
        self.input_calculator.reset()
        self.wav_list = wav_list

        # 音声をどこまで入力したか
        self._wav_ptr = 0
        # 出力していない分のu(t)と特徴量キャッシュのリスト
        self._res_ut_list = [
            torch.zeros((0, 1),
                        dtype=torch.float32).to(input_calculator.device)
        ] * len(wav_list)
        self._res_feat_list = [
            torch.zeros((0, self.input_calculator.feature_dim),
                        dtype=torch.float32).to(input_calculator.device)
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

    def get_ut_features(self,
                        num_frames: int) -> Tuple[PackedSequence, PackedSequence, torch.Tensor]:
        """（最大）指定したフレーム分の特徴量を取得する
        
        Args:
          num_frames(int): 取得したい（最大フレーム数）
        
        Returns:
          PackedSequence: u(t)のPackedSequence
          PackedSequence: 特徴量のPackedSequence
          torch.Tensor: 特徴列の長さを並べたベクトル
            なお，deviceはCPU限定．
        
        Note:
          
        """
        while self._wav_ptr < self._max_wav_len and \
              self._max_res_feat_len < num_frames:
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
            ut_list, feat_list = self.input_calculator.calc(in_wav_list)
            self._wav_ptr = wav_end
            if feat_list is None:
                continue
            # ---
            res_ut_list = []
            max_res_ut_len = 0
            for res_ut, ut in zip(self._res_ut_list, ut_list):
                new_res_ut = torch.cat([res_ut, ut], dim=0)
                if new_res_ut.shape[0] > max_res_ut_len:
                    max_res_ut_len = new_res_ut.shape[0]
                res_ut_list.append(new_res_ut)
            self._res_ut_list = res_ut_list
            # ---
            res_feat_list = []
            max_res_feat_len = 0
            for res_feat, feat in zip(self._res_feat_list, feat_list):
                new_res_feat = torch.cat([res_feat, feat], dim=0)
                if new_res_feat.shape[0] > max_res_feat_len:
                    max_res_feat_len = new_res_feat.shape[0]
                res_feat_list.append(new_res_feat)
            self._res_feat_list = res_feat_list
            # ---
            if max_res_feat_len != max_res_ut_len:
                print("WARNING: lengths of residual feature and u(t) are diffrenet")
            self._max_res_feat_len = max_res_feat_len
        # import ipdb; ipdb.set_trace()
        ut_list = []
        feat_list = []
        feat_len_list = []
        res_ut_list = []
        res_feat_list = []
        max_res_feat_len = 0
        for res_ut, res_feat in zip(self._res_ut_list, self._res_feat_list):
            if res_feat.shape[0] > num_frames:
                ut = res_ut[:num_frames]
                feat = res_feat[:num_frames]
                feat_len = num_frames
                res_ut = res_ut[num_frames:]
                res_feat = res_feat[num_frames:]
            elif res_feat.shape[0] == 0:
                # empty feature, dummy feature and ut is appended
                ut = torch.zeros((1, 1), dtype=torch.float32)
                ut = ut.to(self.input_calculator.device)
                feat = torch.zeros((1, self.input_calculator.feature_dim),
                                   dtype=torch.float32)
                feat = feat.to(self.input_calculator.device)
                feat_len = 0
            else:
                ut = res_ut
                feat = res_feat
                feat_len = feat.shape[0]
                res_ut = torch.zeros((0, 1), dtype=torch.float32)
                res_ut = res_ut.to(self.input_calculator.device)
                res_feat = torch.zeros((0, self.input_calculator.feature_dim),
                                       dtype=torch.float32)
                res_feat = res_feat.to(self.input_calculator.device)
            ut_list.append(ut)
            feat_list.append(feat)
            feat_len_list.append(feat_len)
            res_ut_list.append(res_ut)
            res_feat_list.append(res_feat)
            if res_feat.shape[0] > max_res_feat_len:
                max_res_feat_len = res_feat.shape[0]
        self._res_ut_list = res_ut_list
        self._res_feat_list = res_feat_list
        self._max_res_feat_len = max_res_feat_len
        ut_packed = pack_sequence(ut_list, enforce_sorted=False)
        feat_packed = pack_sequence(feat_list, enforce_sorted=False)
        lengths = torch.tensor(feat_len_list, dtype=torch.int16)
        # import ipdb; ipdb.set_trace()
        return ut_packed, feat_packed, lengths


class TorchTrainerForTurnDetector1d(TorchTrainer):
    """TrunDetector1d用のTorchTrainer"""

    def __init__(self,
                 turn_detector: TurnDetector1d,
                 *args,
                 backprop_len=500,
                 **kwargs):
        self._turn_detector = turn_detector
        self._backprop_len = backprop_len

        # 入力の自動転送は必ず無効化する
        kwargs.update({'automatic_input_transfer': False})
        super().__init__(
            self._turn_detector.torch_model, *args, **kwargs)

    def _forward(self, batch, update=True):
        """引数の型によって具体的に呼ぶ関数を変える"""
        if isinstance(batch[0], list):
            return self._forward_from_wav(batch, update)
        else:
            return self._forward_from_feature(batch, update)
        
    def _forward_from_feature(self, batch, update=True):
        """
        2020年2月22日 更新分 予め特徴量等が抽出済の場合
        batchは，torch.Tensorのリスト（バッチサイズによって変わる）
        batch[i]は，(Li, 2, 4 + ut_dim + feature_dim)のテンソル．
        Liはデータによって異なる時系列の長さ．
        次の2は，0, 1 それぞれのチャネルに対応．
        4次元の内訳は，ut(true), st, zt, ft．
        """
        # 各データの長さのリスト
        lengths = [d.shape[0] for d in batch]
        # 最長の長さ
        max_length = max(lengths)
        # u(t)の次元数
        ut_dim = self._turn_detector.input_calculator.ut_dim
        
        dummy_zero = torch.zeros(
            1, 2, batch[0].shape[2],
            device=self._turn_detector.input_calculator.device)
        
        self._turn_detector.reset()
        offset_frame = 0
        total_loss = torch.tensor(0.0)
        while offset_frame < max_length:
            sub_batch = [d[offset_frame:(offset_frame + self._backprop_len)]
                         for d in batch]
            ut_list = []
            feat_list = []
            st_list = []
            target_list = []
            real_lengths = []
            for sb in sub_batch:
                for ch in (0, 1):
                    real_lengths.append(sb.shape[0])
                    if sb.shape[0] == 0:
                        sb = dummy_zero
                    ut_list.append(sb[:, ch, 4:(4 + ut_dim)])
                    feat_list.append(sb[:, ch, (4 + ut_dim):])
                    st_list.append(sb[:, ch, 1:2])
                    target_list.append(sb[:, ch, :4])
            ut_packed = pack_sequence(ut_list, enforce_sorted=False)
            feat_packed = pack_sequence(feat_list, enforce_sorted=False)
            st_packed = pack_sequence(st_list, enforce_sorted=False)
            target_packed = pack_sequence(target_list, enforce_sorted=False)

            y_packed = self._turn_detector.forward_core(
                ut_packed, feat_packed, st_packed)
            y_padded, _ = pad_packed_sequence(y_packed)
            target_padded, _ = pad_packed_sequence(target_packed)

            # import ipdb; ipdb.set_trace()

            loss, loss_count = self._criterion(
                y_padded, target_padded, real_lengths)
            # import ipdb; ipdb.set_trace()
            
            if loss != 0:
                # # ロスを積算する
                # total_loss = \
                # (total_loss * total_loss_count + loss.clone().detach() * loss_count) \
                #  / (total_loss_count + loss_count)
                # total_loss_count = total_loss_count + loss_count
                total_loss += loss.clone().detach().item()
                
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
            self._turn_detector.detach()
            offset_frame += self._backprop_len
            
        return total_loss
        
    def _forward_from_wav(self, batch, update=True):
        """バッチの内容が特殊かつ，アップデートの仕方も特殊．
        batch[0] はwavのリスト
        batch[1] はtarget情報のリスト

        （入出力の最大長は一致している＝入力同期で出力されるのが前提）
        フォワード計算，パラメータのアップデートは self._backprop_len 毎に
        行い，その度にコンテクストはデタッチする（リセットではないので注意）
        """
        wavs = batch[0]
        targets = batch[1]

        self._turn_detector.reset()
        targets = [t.to(self._turn_detector.device) for t in targets]

        input_calculator_wrapper = _InputCalculatorWrapper(
            self._turn_detector.input_calculator, wavs)
        
        count = 0
        total_loss = torch.tensor(0.0)
        total_loss_count = 0
        offset_target = 0
        while not input_calculator_wrapper.finished:
            # 注目するターゲット
            sub_targets = [
                t[offset_target:(offset_target + self._backprop_len)]
                for t in targets
            ]
            ut_packed, feat_packed, lengths = \
                input_calculator_wrapper.get_ut_features(self._backprop_len)
            
            sub_target = pad_sequence(sub_targets)
            sub_st = sub_target[:, :, 1:2]
            if sub_st.shape[0] >= lengths.max():
                _lengths = lengths.clone().detach()
                _lengths[_lengths == 0] = 1
                sub_st_packed = pack_padded_sequence(sub_st, _lengths, enforce_sorted=False)
            else:
                sub_st_packed = None
            y_packed = self._turn_detector.forward_core(
                ut_packed, feat_packed, sub_st_packed)
            y_padded, _ = pad_packed_sequence(y_packed)
            lengths = lengths.detach().cpu().numpy().tolist()
            cri_lengths = [
                min(t.shape[0], yl) for t, yl in zip(sub_targets, lengths)
            ]
            
            # ロスを求める
            loss, loss_count = self._criterion(
                y_padded, sub_target, cri_lengths)
            
            if loss != 0:
                # # ロスを積算する
                # total_loss = \
                # (total_loss * total_loss_count + loss.clone().detach() * loss_count) \
                #  / (total_loss_count + loss_count)
                # total_loss_count = total_loss_count + loss_count
                total_loss += loss.clone().detach().item()
                
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
            self._turn_detector.input_calculator.detach()
            self._turn_detector.detach()

            count += 1
            offset_target += self._backprop_len

        # total_loss /= count
        # return total_loss * 100
        return total_loss

    
class TurnDetector1dTrainer(metaclass=ABCMeta):
    """TurnDetector1dTrainerの基底クラス．
    """

    def __init__(self):
        pass

    @abstractmethod
    def build_torch_trainer(self, turn_detector: TurnDetector1d):
        pass

    def train(self, turn_detector: TurnDetector1d):
        self.torch_trainer = self.build_torch_trainer(turn_detector)
        self.torch_trainer.train()


def construct_input_calculator(input_calculator_number,
                               input_calculator_construct_args=(
                                   [],
                                   {},
                               )):
    """入力計算器を構築する
    
    Args:
      input_calculator_number: 構築する入力計算器の番号
      input_calculator_construct_args:
        コンストラクタに与える引数．
        args（リスト）と，kwargs（ディクショナリ）のタプル
    """
    module_name = "sflib.mlpr.speech.turn_detector_1d." + \
                  "input_calculator{:04d}".format(input_calculator_number)
    class_name = "InputCalculator{:04d}".format(input_calculator_number)
    # import importlib
    # mod = importlib.import_module(module_name)
    # cls = getattr(mod, class_name)
    # args, kwargs = input_calculator_construct_args
    # input_calculator = cls(*args, **kwargs)
    from ....util.cls import construct
    input_calculator = construct(
        module_name, class_name, input_calculator_construct_args)
    return input_calculator


def construct_turn_detector(turn_detector_number,
                            trainer_number,
                            input_calculator_number,
                            input_calculator_construct_args=([], {},)):
    module_name = "sflib.mlpr.speech.turn_detector_1d." + \
                  "turn_detector_1d{:04d}".format(turn_detector_number)
    class_name = "TurnDetector1d{:04d}".format(turn_detector_number)
    # import importlib
    # mod = importlib.import_module(module_name)
    # cls = getattr(mod, class_name)
    # turn_detector = cls(trainer_number, input_calculator_number,
    #                     input_calculator_construct_args)
    from ....util.cls import construct
    turn_detector = construct(
        module_name, class_name,
        [trainer_number, input_calculator_number,
         input_calculator_construct_args])
    return turn_detector


def construct_trainer(trainer_number, *args, **kwargs):
    module_name = "sflib.mlpr.speech.turn_detector_1d." + \
                  "trainer{:04d}".format(trainer_number)
    class_name = "TurnDetector1dTrainer{:04d}".format(trainer_number)
    # import importlib
    # mod = importlib.import_module(module_name)
    # cls = getattr(mod, class_name)
    # trainer = cls(*args, **kwargs)
    from ....util.cls import construct
    trainer = construct(
        module_name, class_name, (args, kwargs))
    return trainer
