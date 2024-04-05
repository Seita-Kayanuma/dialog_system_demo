# 0003をベースに，少しネットワークの構造を大きくしてみたもの
from .base import FacialTurnDetector
from .base import FacialTurnDetectorTorchModel
from .base import FacialTurnDetectorFeatureExtractor
import torch.nn as nn
import torch


class FacialTurnDetectorModel(FacialTurnDetectorTorchModel):
    def __init__(self, in_feature_dim, frame_window=10, cnn_out_channels=10):
        super().__init__()
        self._feature_dim = in_feature_dim
        self._frame_window = frame_window

        # 時間方向のframe_window分の畳み込み
        self.conv1 = nn.Conv2d(1, cnn_out_channels, (1, self._frame_window))
        self.conv1b = nn.BatchNorm1d(cnn_out_channels * self._feature_dim)
        self.linear_in1 = nn.Linear(cnn_out_channels * self._feature_dim, 256)
        self.linear_in1b = nn.BatchNorm1d(256)
        self.lstm1 = nn.LSTM(256, 256)
        self.linear_1 = nn.Linear(256, 128)
        self.linear_1b = nn.BatchNorm1d(128)
        self.linear_out_turn = nn.Linear(128, 2)  # 0 (OFF), 1 (ON)

        self._context = None
        self._input_context = None

    def reset_context(self):
        self._context = None
        self._input_context = None

    def detach_context(self):
        if self._context is None:
            return
        self._context = tuple([c.clone().detach() for c in self._context])

    def forward(self, feat):
        # feat = feat[:, :, 64:]

        max_length, batch, feat_dim = feat.shape

        # 入力履歴と入力でバッチサイズが違ったら一旦Noneにする
        if self._input_context is not None and \
           self._input_context.shape[1] != batch:
            self._input_context = None
        # 入力履歴がNoneの場合は0で初期化する
        if self._input_context is None:
            self._input_context = torch.zeros((self._frame_window * 2,) +
                                              feat.shape[1:])
            self._input_context = self._input_context.to(feat.device)

        if max_length < self._frame_window:
            ll = self._frame_window - max_length
            feat = torch.cat([self._input_context[-ll:], feat], dim=0)
            self._input_context = self._input_context[:-ll]

        feat_list = [feat]
        for i in range(1, self._frame_window):
            input_context = self._input_context[-i:]
            f = torch.cat([input_context, feat[:-i]], dim=0)
            feat_list.append(f)
        if len(feat_list) < self._frame_window:
            for i in range(len(feat_list), self._frame_window):
                feat_list.append(
                    self._input_context[-(i + self._frame_window):-i])
        stacked_feat = torch.stack(feat_list, dim=3)
        stacked_feat = stacked_feat[-max_length:]
        stacked_feat = stacked_feat.reshape(max_length * batch, feat_dim,
                                            self._frame_window)
        stacked_feat = stacked_feat.unsqueeze(1)

        # 入力履歴を更新する
        if feat.shape[0] >= self._frame_window * 2:
            self._input_context = feat[-(self._frame_window * 2):]
        else:
            self._input_context = torch.cat([self._input_context, feat], dim=0)
            self._input_context = \
                self._input_context[-(self._frame_window * 2):]

        h_conv1 = self.conv1(stacked_feat)
        h_conv1 = h_conv1.reshape(max_length * batch, -1)
        h_conv1 = torch.tanh(h_conv1)
        h_conv1 = self.conv1b(h_conv1)
        h_linear_in1 = torch.tanh(self.linear_in1(h_conv1))
        h_linear_in1 = self.linear_in1b(h_linear_in1)
        h_linear_in1 = h_linear_in1.reshape(max_length, batch, -1)
        h_lstm1, context = self.lstm1(h_linear_in1, self._context)
        h_lstm1 = h_lstm1.reshape(max_length * batch, -1)
        h_lstm1 = torch.tanh(h_lstm1)
        h_linear1 = torch.tanh(self.linear_1(h_lstm1))
        h_linear1 = self.linear_1b(h_linear1)
        h_turn = self.linear_out_turn(h_linear1)
        h_turn = h_turn.reshape(max_length, batch, -1)

        self._context = context
        
        return h_turn


class FacialTurnDetector0004(FacialTurnDetector):
    def __init__(self,
                 feature_extractor: FacialTurnDetectorFeatureExtractor,
                 frame_window=10,
                 cnn_out_channels=10,
                 *args,
                 **kwargs):
        super().__init__(feature_extractor, *args, **kwargs)
        self._model = FacialTurnDetectorModel(feature_extractor.feature_dim,
                                              frame_window, cnn_out_channels)
        self._frame_window = frame_window
        self._cnn_out_channels = cnn_out_channels
        if self.device:
            self._model = self._model.to(self.device)

    @property
    def torch_model(self) -> FacialTurnDetectorTorchModel:
        return self._model

    def get_filename_base(self):
        return super().get_filename_base() + \
            "W{:03d}".format(self._frame_window) + \
            "C{:03d}".format(self._cnn_out_channels)

    def convert_result_to_task_softmax(self,
                                       result: torch.Tensor) -> torch.Tensor:
        return torch.softmax(result, dim=2)
