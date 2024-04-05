# coding: utf-8
from keras.callbacks import History
import pandas as pd


class CsvLoggerCallback(History):
    """
    エポックごとに学習結果をCSVに書き出すコールバック
    """

    def __init__(self, filename):
        super().__init__()
        self._filename = filename
        self._epochs = []

    def on_epoch_end(self, epoch, logs=None):
        super().on_epoch_end(epoch, logs)
        self._epochs.append(epoch + 1)

        # import ipdb; ipdb.set_trace()

        # 大したコストではないので毎回新規ファイルで作成する
        hist = {'epoch': self._epochs}
        hist.update(self.history)
        df = pd.DataFrame(hist)
        df.to_csv(self._filename)


class PdfWriterCallback(History):
    """
    訓練履歴の様子をPDFに書き出すコールバック.
    エポック終了時に順次書き出す．
    履歴を使うので History コールバックを継承している
    """

    def __init__(self, filename, title=None):
        super().__init__()
        self._filename = filename
        self._title = None

    def on_epoch_end(self, epoch, logs=None):
        super().on_epoch_end(epoch, logs)

        h = self.history
        epochs = range(1, len(h['loss']) + 1)
        loss_values = h['loss']
        if 'val_loss' in h:
            val_loss_values = h['val_loss']
        else:
            val_loss_values = None

        from matplotlib.backends.backend_pdf import PdfPages
        pdf = PdfPages(self._filename)

        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(epochs, loss_values, 'bo', label='Training Loss')
        if val_loss_values:
            plt.plot(epochs, val_loss_values, 'b', label='Validation Loss')
        if self._title:
            plt.title(self._title)
        else:
            plt.title('Training (and Validation) Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        if val_loss_values:
            plt.legend()

        pdf.savefig()
        pdf.close()
