from .base import FacialExpressionRecognizer
from keras.layers import Input
from keras.layers import Dense
from keras.models import Model
from keras.utils import to_categorical
import keras.backend as K
from ....corpus.cohn_kanade import CKplus


class FacialExpressionRecognizer0002(FacialExpressionRecognizer):
    def __init__(self, feature_extractor=None):
        super().__init__(feature_extractor)
        fdim = feature_extractor.get_feature_dim()
        self.edim = len(CKplus.EMOTION_NAMES)

        input_vec = Input(shape=(fdim,))
        h = Dense(64, activation='relu')(input_vec)
        output_vec = Dense(self.edim, activation='softmax')(h)
        self.model = Model(input_vec, output_vec)
        self.initial_weights = self.model.get_weights()

    def fit_with_features(self, x, y, *args, validation_data=None, **kwargs):
        self.model.set_weights(self.initial_weights)
        self.model.compile(optimizer='rmsprop', loss='binary_crossentropy',
                           metrics=['accuracy'])
        y_cat = to_categorical(y)
        validation_data_cat = (validation_data[0],
                               to_categorical(validation_data[1], self.edim))
        self.model.fit(x, y_cat, validation_data=validation_data_cat,
                       epochs=100, batch_size=10)

    def predict_with_features(self, x):
        y_cat = self.model.predict(x)
        return K.eval(K.argmax(y_cat))

    def save_model(self, filename):
        self.model.save_weights(filename)

    def load_model(self, filename):
        self.model.load_weights(filename)
