from .base import PhoneTypeWriterTrainer
from keras.layers import Input, Lambda
from keras.models import Model
from keras.callbacks import EarlyStopping
import keras.backend as K
from .util import PhoneTypeWriterDataIterator
# warning (obsolete)
from ....corpus.csj.process import SpectrogramImageArrayWithVADPhones
from ....ext.keras.callbacks import CsvLoggerCallback


def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    # y_pred = y_pred[:, 2:, :]
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)


class PhoneTypeWriterTrainerCSJ0001(PhoneTypeWriterTrainer):
    def __init__(self, phone_type_writer):
        super().__init__(phone_type_writer)

    def _build(self):
        # 学習用のモデル
        input_data, input_length = self.phone_type_writer.get_input_layer()
        y_pred = self.phone_type_writer.get_output_layer()
        labels = Input(name='the_labels', shape=(None, ), dtype='float32')
        label_length = Input(name='label_length', shape=(1, ), dtype='int64')
        loss_out = Lambda(
            ctc_lambda_func, output_shape=(1, ),
            name='ctc')([y_pred, labels, input_length, label_length])

        self._model = Model(
            inputs=[input_data, labels, input_length, label_length],
            outputs=loss_out)
        self._model.compile(
            loss={
                'ctc': lambda y_true, y_pred: y_pred
            }, optimizer='adam')
        # 学習データ，検証データのイテレータ
        train_sia = SpectrogramImageArrayWithVADPhones(
            cond='A.*[^1]$', max_wav_num_for_each_file=100)
        # cond='A01M0014', max_wav_num_for_each_file=100)
        train_sia.construct()
        self._train_iterator = PhoneTypeWriterDataIterator(
            train_sia,
            self.phone_type_writer.feature_extractor,
            batch_size=500)
        validation_sia = SpectrogramImageArrayWithVADPhones(
            cond='A.*1$', max_wav_num_for_each_file=10)
        # cond='A01M0015', max_wav_num_for_each_file=10)
        validation_sia.construct()
        self._validation_iterator = PhoneTypeWriterDataIterator(
            validation_sia,
            self.phone_type_writer.feature_extractor,
            batch_size=500)

    def train(self, **kwargs):
        cb_early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=0,
            verbose=1,
            mode='auto',
            restore_best_weights=True)
        cb_csv_logger = CsvLoggerCallback(self.get_csv_log_filename())
        
        self._build()
        self._model.fit_generator(
            self._train_iterator,
            steps_per_epoch=self._train_iterator.steps_per_epoch,
            epochs=20,
            validation_data=self._validation_iterator,
            validation_steps=self._validation_iterator.steps_per_epoch,
            callbacks=[cb_early_stopping, cb_csv_logger])


def train_with_autoencoder(phone_type_writer_version,
                           feature_extractor_version, autoencoder_version,
                           autoencoder_trainer_module_postfix,
                           autoencoder_trainer_class_postfix):
    # --- GPU ENVIRONMENT SETUP ---
    import tensorflow as tf
    from keras.backend import tensorflow_backend
    config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
    session = tf.Session(config=config)
    tensorflow_backend.set_session(session)
    # --- GPU ENVIRONMENT SETUP ---

    from . import base as ab
    feature_extractor = ab.construct_feature_extractor_with_autoencoder(
        feature_extractor_version, autoencoder_version,
        autoencoder_trainer_module_postfix, autoencoder_trainer_class_postfix)
    phone_type_writer = ab.construct_phone_type_writer(
        phone_type_writer_version, feature_extractor)
    trainer = PhoneTypeWriterTrainerCSJ0001(phone_type_writer)
    trainer.train()
    ab.save_phone_type_writer(trainer, upload=True)
    trainer.upload_csv_log()
