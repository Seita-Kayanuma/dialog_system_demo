# coding: utf-8
"""

"""
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader
import numpy as np


class TorchTrainerCallback(object):
    def __init__(self):
        pass

    def train_start_callback(self, trainer):
        """Callback at the start of traing.

        Args:
          trainer (TorchTrainer): trainer
        """
        pass

    def train_epoch_start_callback(self, trainer, epoch, total_epoch,
                                   total_batch):
        """Callback at the start of training new epoch.

        Args:
          trainer (TorchTrainer): trainer
          epoch (int): epoch no. (start at 1)
          total_epoch (int): total no. of epoch
          total_batch (int): total no. of batch
        """
        pass

    def train_batch_start_callback(self, trainer, epoch, total_epoch, batch,
                                   total_batch):
        """Callback at the start of training new batch.

        Args:
          trainer (TorchTrainer): trainer
          epoch (int): epoch no. (start at 1)
          total_epoch (int): total no. of epoch
          batch (int): batch no. (start at 1)
          total_batch (int): total no. of batch
        """
        pass

    def train_before_optimizer_step_callback(self, trainer):
        pass

    def train_batch_finish_callback(self, trainer, epoch, total_epoch, batch,
                                    total_batch, train_loss):
        """Callback at the finish of training a batch.

        Args:
          trainer (TorchTrainer): trainer
          epoch (int): epoch no. (start at 1)
          total_epoch (int): total no. of epoch
          batch (int): batch no. (start at 1)
          total_batch (int): total no. of batch
          train_loss (float): total average loss in the epoch so far
        """
        pass

    def train_epoch_finish_callback(self, trainer, epoch, total_epoch,
                                    train_loss):
        """Callback at the finish of training epoch

        Args:
          trainer (TorchTrainer): trainer
          epoch (int): epoch no. (start at 1)
          total_epoch (int): total no. of epoch
          train_loss (float): total average loss in the epoch
        """
        pass

    def validation_epoch_start_callback(self, trainer, epoch, total_epoch,
                                        total_batch):
        """Callback at the start of validating new epoch.

        Args:
          trainer (TorchTrainer): trainer
          epoch (int): epoch no. (start at 1)
          total_batch (int): total no. of batch
          total_epoch (int): total no. of epoch
        """
        pass

    def validation_batch_start_callback(self, trainer, epoch, total_epoch,
                                        batch, total_batch):
        """Callback at the start of validating new batch.

        Args:
          trainer (TorchTrainer): trainer
          epoch (int): epoch no. (start at 1)
          total_epoch (int): total no. of epoch
          batch (int): batch no. (start at 1)
          total_batch (int): total no. of batch
        """
        pass

    def validation_batch_finish_callback(self, trainer, epoch, total_epoch,
                                         batch, total_batch, train_loss):
        """Callback at the finish of validating a batch.

        Args:
          trainer (TorchTrainer): trainer
          epoch (int): epoch no. (start at 1)
          total_epoch (int): total no. of epoch
          batch (int): batch no. (start at 1)
          total_batch (int): total no. of batch
          validation_loss (float): total average loss in the epoch so far
        """
        pass

    def validation_epoch_finish_callback(self, trainer, epoch, total_epoch,
                                         validation_loss):
        """Callback at the finish of validation epoch

        Args:
          trainer (TorchTrainer): trainer
          epoch (int): epoch no. (start at 1)
          total_epoch (int): total no. of epoch
          validation_loss (float): total average loss in the epoch
        """
        pass

    def train_validation_epoch_finish_callback(self, trainer, epoch,
                                               total_epoch):
        """Callback at the finish of training and validation epoch

        Args:
          trainer (TorchTrainer): trainer
          epoch (int): epoch no. (start at 1)
          total_epoch (int): total no. of epoch
        """
        pass

    def train_finish_callback(self, trainer):
        """Callback at the finish of training
        """
        pass


class TorchTrainer(object):
    def __init__(
            self,
            model: Module,
            criterion: Module,
            optimizer: Optimizer,
            train_data_loader: DataLoader,
            validation_data_loader: DataLoader = None,
            epoch=20,
            callbacks=None,
            device=None,
            automatic_input_transfer=True,
            validation_criterion: Module = None,
    ):
        self._model = model
        self._train_criterion = criterion
        self._optimzier = optimizer
        self._train_data_loader = train_data_loader
        self._validation_data_loader = validation_data_loader
        self._epoch = epoch
        self._callbacks = callbacks
        self._device = device
        # 早期終了をリクエストされた時に立つフラグ
        self._early_stop_requested = False
        self._automatic_input_transfer = automatic_input_transfer
        if validation_criterion is None:
            self._validation_criterion = self._train_criterion
        else:
            self._validation_criterion = validation_criterion
        self._criterion = None

    def train(self):
        """
        学習を実行する
        """
        self._callback_train_start()
        total_epoch = self._epoch
        data_size = len(self._train_data_loader.dataset)
        total_batch = int(
            np.ceil(data_size / self._train_data_loader.batch_size))
        for epoch in range(total_epoch):
            self._model.train()
            self._callback_train_epoch_start(epoch + 1, total_epoch,
                                             total_batch)
            total_loss = 0
            for i, batch in enumerate(self._train_data_loader):
                if self._device and self._automatic_input_transfer:
                    batch = [x.to(self._device) for x in batch]
                self._callback_train_batch_start(epoch + 1, total_epoch, i + 1,
                                                 total_batch)
                self._criterion = self._train_criterion
                loss = self._forward(batch, update=True)
                loss = loss.detach()
                if loss.device.type != 'cpu':
                    loss = loss.cpu()
                loss = loss.numpy()
                total_loss += loss
                self._callback_train_batch_finish(epoch + 1, total_epoch,
                                                  i + 1, total_batch,
                                                  total_loss / (i + 1))
            self._callback_train_epoch_finish(epoch + 1, total_epoch,
                                              total_loss / total_batch)
            if self._validation_data_loader is not None:
                self._do_validation(epoch, total_epoch)
            self._callback_train_validation_epoch_finish(
                epoch + 1, total_epoch)
            if self._early_stop_requested:
                break
        self._callback_train_finish()

    def _do_validation(self, epoch, total_epoch):
        self._model.eval()
        data_size = len(self._validation_data_loader.dataset)
        total_batch = int(
            np.ceil(data_size / self._validation_data_loader.batch_size))
        total_loss = 0

        self._callback_validation_epoch_start(epoch + 1, total_epoch,
                                              total_batch)
        for i, batch in enumerate(self._validation_data_loader):
            if self._device and self._automatic_input_transfer:
                batch = [x.to(self._device) for x in batch]
            self._callback_validation_batch_start(epoch + 1, total_epoch,
                                                  i + 1, total_batch)
            self._criterion = self._validation_criterion
            loss = self._forward(batch, update=False)
            loss = loss.detach()
            if loss.device.type != 'cpu':
                loss = loss.cpu()
            loss = loss.numpy()
            total_loss += loss
            self._callback_validation_batch_finish(epoch + 1, total_epoch,
                                                   i + 1, total_batch,
                                                   total_loss / (i + 1))
        self._callback_validation_epoch_finish(epoch + 1, total_epoch,
                                               total_loss / total_batch)

    def _forward(self, batch, update=True):
        """
        1バッチ分学習を進める．

        Args:
          batch (list): バッチ（Tensorのリスト）．
             現状はbatch[0]が入力，batch[1]がターゲットで固定．
             （ターゲットが無いようなタスクは想定していない）
          update (bool): モデルのパラメタを更新する場合はTrue．
             バリデーションの場合など必要ない場合はFalse

        Returns:
          Tensor: ロス．
        """
        x, t = batch
        y = self._model(x)
        loss = self._criterion(y, t)
        if update:
            self._optimzier.zero_grad()
            loss.backward()
            self._callback_train_before_optimizer_step()
            self._optimzier.step()
        return loss

    def _callback_train_start(self):
        if self._callbacks:
            for cb in self._callbacks:
                cb.train_start_callback(self)

    def _callback_train_epoch_start(self, epoch, total_epoch, total_batch):
        if self._callbacks:
            for cb in self._callbacks:
                cb.train_epoch_start_callback(self, epoch, total_epoch,
                                              total_batch)

    def _callback_train_batch_start(self, epoch, total_epoch, batch,
                                    total_batch):
        if self._callbacks:
            for cb in self._callbacks:
                cb.train_batch_start_callback(self, epoch, total_epoch, batch,
                                              total_batch)

    def _callback_train_before_optimizer_step(self):
        if self._callbacks:
            for cb in self._callbacks:
                cb.train_before_optimizer_step_callback(self)

    def _callback_train_batch_finish(self, epoch, total_epoch, batch,
                                     total_batch, train_loss):
        if self._callbacks:
            for cb in self._callbacks:
                cb.train_batch_finish_callback(self, epoch, total_epoch, batch,
                                               total_batch, train_loss)

    def _callback_train_epoch_finish(self, epoch, total_epoch, train_loss):
        if self._callbacks:
            for cb in self._callbacks:
                cb.train_epoch_finish_callback(self, epoch, total_epoch,
                                               train_loss)

    def _callback_validation_epoch_start(self, epoch, total_epoch,
                                         total_batch):
        if self._callbacks:
            for cb in self._callbacks:
                cb.validation_epoch_start_callback(self, epoch, total_epoch,
                                                   total_batch)

    def _callback_validation_batch_start(self, epoch, total_epoch, batch,
                                         total_batch):
        if self._callbacks:
            for cb in self._callbacks:
                cb.validation_batch_start_callback(self, epoch, total_epoch,
                                                   batch, total_batch)

    def _callback_validation_batch_finish(self, epoch, total_epoch, batch,
                                          total_batch, validation_loss):
        if self._callbacks:
            for cb in self._callbacks:
                cb.validation_batch_finish_callback(self, epoch, total_epoch,
                                                    batch, total_batch,
                                                    validation_loss)

    def _callback_validation_epoch_finish(self, epoch, total_epoch,
                                          validation_loss):
        if self._callbacks:
            for cb in self._callbacks:
                cb.validation_epoch_finish_callback(self, epoch, total_epoch,
                                                    validation_loss)

    def _callback_train_validation_epoch_finish(self, epoch, total_epoch):
        if self._callbacks:
            for cb in self._callbacks:
                cb.train_validation_epoch_finish_callback(
                    self, epoch, total_epoch)

    def _callback_train_finish(self):
        if self._callbacks:
            for cb in self._callbacks:
                cb.train_finish_callback(self)
