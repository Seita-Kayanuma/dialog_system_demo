from ..trainer import TorchTrainerCallback
from tqdm import tqdm
import pandas as pd
import time


class StandardReporter(TorchTrainerCallback):
    def __init__(self, train_report_interval=10,
                 validation_report_interval=10):
        super(TorchTrainerCallback, self).__init__()
        self._train_interval = train_report_interval
        self._validation_interval = validation_report_interval

    def train_start_callback(self, trainer):
        print("Training Started")

    def train_epoch_start_callback(self, trainer, epoch, total_epoch,
                                   total_batch):
        self.t = tqdm(desc="Train Epoch {:2d}/{:2d}".format(
            epoch, total_epoch),
                      total=total_batch)
        self._count = 0
        self._reported_count = 0

    def train_batch_finish_callback(self, trainer, epoch, total_epoch, batch,
                                    total_batch, train_loss):
        self._count += 1
        if self._count % self._train_interval == 0 or \
           self._count == total_batch:
            self.t.set_postfix_str("loss={:.2f}".format(train_loss))
            self.t.update(self._count - self._reported_count)
            self._reported_count = self._count

    def train_epoch_finish_callback(self, trainer, epoch, total_epoch,
                                    train_loss):
        self.t.close()

    def validation_epoch_start_callback(self, trainer, epoch, total_epoch,
                                        total_batch):
        self.t = tqdm(desc="Vali. Epoch {:2d}/{:2d}".format(
            epoch, total_epoch),
                      total=total_batch)
        self._count = 0
        self._reported_count = 0

    def validation_batch_finish_callback(self, trainer, epoch, total_epoch,
                                         batch, total_batch, validation_loss):
        self._count += 1
        if self._count % self._validation_interval == 0 or \
           self._count == total_batch:
            self.t.set_postfix_str("loss={:.2f}".format(validation_loss))
            self.t.update(self._count - self._reported_count)
            self._reported_count = self._count

    def validation_epoch_finish_callback(self, trainer, epoch, total_epoch,
                                         validation_loss):
        self.t.close()

    def train_finish_callback(self, trainer):
        print("Entire Training Finished")


class CsvWriterReporter(TorchTrainerCallback):
    def __init__(self, filename):
        super(CsvWriterReporter, self).__init__()
        self._filename = filename
        self.clear()

    def clear(self):
        self._latest_train_loss = None
        self._latest_validation_loss = None
        self._df = None

    def train_start_callback(self, trainer):
        self._df = pd.DataFrame({'epoch': [0], 'time': [time.time()]})
        self._df.to_csv(self._filename)

    def train_epoch_finish_callback(self, trainer, epoch, total_epoch,
                                    train_loss):
        self._latest_train_loss = train_loss

    def validation_epoch_finish_callback(self, trainer, epoch, total_epoch,
                                         validation_loss):
        self._latest_validation_loss = validation_loss

    def train_validation_epoch_finish_callback(self, trainer, epoch,
                                               total_epoch):
        data = {
            'epoch': [epoch],
            'time': [time.time()],
            'train_loss': [self._latest_train_loss]
        }
        if self._latest_validation_loss is not None:
            data.update({'validation_loss': [self._latest_validation_loss]})
        df = pd.DataFrame(data)
        self._df = self._df.append(df, ignore_index=True, sort=False)
        self._df.to_csv(self._filename)

    def train_finish_callback(self, trainer):
        self.clear()
