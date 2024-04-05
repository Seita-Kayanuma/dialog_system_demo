from ..trainer import TorchTrainerCallback
import torch


class Snapshot(TorchTrainerCallback):
    def __init__(self, final_filename=None, epoch_filename_pattern=None):
        """
        Args:
          final_filename (str): 最終的な学習結果を保存するファイルの名前
          epoch_filename_pattern (str): エポックごとの学習結果を保存するファイルのパターン．
            例えば， 'model{:04d}' を与えると，5エポック時のパラメータは model0005.h5 に
            保存される
        """
        self._final_filename = final_filename
        self._epoch_filename_pattern = epoch_filename_pattern

    def train_epoch_finish_callback(self, trainer, epoch, total_epoch,
                                    train_loss):
        if self._epoch_filename_pattern:
            filename = self._epoch_filename_pattern.format(epoch)
            torch.save(trainer._model.state_dict(), filename)

    def train_finish_callback(self, trainer):
        if self._final_filename is not None:
            torch.save(trainer._model.state_dict(), self._final_filename)
            
