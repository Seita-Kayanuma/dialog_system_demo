from .base import PhoneTypeWriter, convert_id_to_phone, phone_list

from keras.layers import Input, Dense
from keras.layers.recurrent import LSTM
from keras.layers.wrappers import TimeDistributed
from keras.layers import Activation
from keras.models import Model
import keras.backend as K
import numpy as np


class PhoneTypeWriter0003(PhoneTypeWriter):
    def __init__(self, feature_extractor):
        super().__init__(feature_extractor)
        self._build()

    def _build(self):
        feature_dim = self.feature_extractor.get_feature_dim()

        # normal input path
        input_data = Input(
            name='the_input', shape=(None, feature_dim), dtype='float32')
        inner = TimeDistributed(Dense(128, activation='tanh',
                                      name='dense1'))(input_data)
        inner = TimeDistributed(Dense(64, activation='tanh',
                                      name='dense2'))(inner)
        inner = LSTM(
            64, return_sequences=True, activation='tanh', name='lstm1')(inner)
        inner = TimeDistributed(Dense(32, name='dense3', activation='tanh'))(inner)
        inner = TimeDistributed(Dense(len(phone_list), name='dense4'))(inner)
        y_pred = Activation('softmax', name='softmax')(inner)
        model = Model(inputs=[input_data], outputs=[y_pred])
        model.summary()
        
        # decoding path
        input_length = Input(name='input_length', shape=(1, ), dtype='int64')
        top_k_decoded, _ = K.ctc_decode(y_pred, input_length[:, 0])
        decoder = K.function([input_data, input_length], [top_k_decoded[0]])

        self._input_data = input_data
        self._input_length = input_length
        self._y_pred = y_pred
        self._decoder = decoder
        self._model = model

    def predict(self, x):
        x = self.feature_extractor.calc(x)
        x = x.reshape((1, ) + x.shape)
        x_length = np.array([x.shape[1]]).reshape((1, 1))
        phone_ids_pred = self._decoder([x, x_length])
        return convert_id_to_phone(phone_ids_pred[0][0].tolist())

    def predict_y_pred(self, x):
        x = self.feature_extractor.calc(x)
        x = x.reshape((1, ) + x.shape)
        y_pred = self._model.predict([x])
        return y_pred[0]

    def get_input_layer(self):
        return self._input_data, self._input_length

    def get_output_layer(self):
        return self._y_pred
    
    def save_model(self, filename):
        self._model.save_weights(filename)

    def load_model(self, filename):
        self._model.load_weights(filename)
