# coding: utf-8
from keras.layers import Input
from keras.layers import Conv2D, Conv2DTranspose, Cropping2D
from keras.layers import Flatten, Dense, Reshape
from keras.layers import BatchNormalization, Layer
from keras.layers import Concatenate
from keras.layers import Activation
import keras.backend as K
from .base import SpectrogramImageAutoEncoder
import numpy as np

# 0007が基本．Floorの後にReLUの活性化を入れる．
# これが以前のChainerで作っていたものと等価なネットワーク．

class Floor(Layer):
    """
    各周波数ビンに対して固有の重みとバイアスによって，
    線形演算（weight * x + bias）をするもの．
    重みは正の値に限定するためにかける前に指数を取っている．
    重みとバイアスの初期値は0（重みの方は指数を取るので，実質1）となる．
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.input_dim = input_shape[1]
        self.weight = self.add_weight(
            name='weight',
            shape=(1, self.input_dim, 1, 1),
            initializer='zeros',
            trainable=True)
        self.bias = self.add_weight(
            name='bias',
            shape=(1, self.input_dim, 1, 1),
            initializer='zeros',
            trainable=True)
        super().build(input_shape)

    def call(self, x):
        return K.exp(self.weight) * x + self.bias

    def compute_output_shape(self, input_shape):
        return input_shape


class Unfloor(Layer):
    def __init__(self, floor_layer, **kwargs):
        super().__init__(**kwargs)
        self.weight = floor_layer.weight
        self.bias = floor_layer.bias

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, x):
        return (x - self.bias) / K.exp(self.weight)

    def compute_output_shape(self, input_shape):
        return input_shape


class L2Norm(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, x):
        l2norm = K.sqrt(K.sum(K.square(K.batch_flatten(x)), axis=1, keepdims=True)) \
            / np.prod(x.shape[1:]).value + 1e-5
        return l2norm

    def compute_output_shape(self, input_shape):
        return (input_shape[0], 1)


class Normalize(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shapes):
        pass

    def call(self, x):
        coef = K.reshape(x[1], (-1, 1, 1, 1))
        return x[0] / coef

    def compute_output_shape(self, input_shapes):
        return input_shapes[0]


class Denormalize(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shapes):
        pass

    def call(self, x):
        coef = K.reshape(x[1], (-1, 1, 1, 1))
        return x[0] * coef

    def compute_output_shape(self, input_shapes):
        return input_shapes[0]


class SpectrogramImageAutoEncoder0009(SpectrogramImageAutoEncoder):
    def __init__(self):
        super().__init__()

        input = Input(shape=(512, 10, 1), name='input')
        x = input
        # L2正規化
        l2 = L2Norm()(x)
        x = Normalize()([x, l2])
        # フロアリングとReLU
        floor_layer = Floor()
        x = floor_layer(x)
        x = Activation('relu')(x)
        # CNN-1
        x = Conv2D(16, (5, 5), activation='relu', padding='same', strides=(2, 2),
                   kernel_initializer='he_normal', bias_initializer='zeros')(x)
        x = BatchNormalization()(x)
        # CNN-2
        x = Conv2D(24, (7, 5), activation='relu', padding='same', strides=(3, 2),
                   kernel_initializer='he_normal', bias_initializer='zeros')(x)
        x = BatchNormalization()(x)
        # CNN-3
        x = Conv2D(36, (7, 5), activation='relu', padding='same', strides=(3, 2),
                   kernel_initializer='he_normal', bias_initializer='zeros')(x)
        x = BatchNormalization()(x)
        # CNN-4
        x = Conv2D(54, (7, 5), activation='relu', padding='same', strides=(3, 2),
                   kernel_initializer='he_normal', bias_initializer='zeros')(x)
        x = BatchNormalization()(x)
        # FC-1
        x = Flatten()(x)
        x = Dense(256, activation='relu',
                  kernel_initializer='he_normal', bias_initializer='zeros')(x)
        x = BatchNormalization()(x)
        # FC-2
        x = Dense(64, activation='relu',
                  kernel_initializer='he_normal', bias_initializer='zeros')(x)
        x = BatchNormalization()(x)
        x = Activation('tanh')(x)
        # ENCODED
        encoded = x
        # Inverse FC-2
        x = Dense(256, activation='relu',
                  kernel_initializer='he_normal', bias_initializer='zeros')(x)
        x = BatchNormalization()(x)
        # Inverse FC-1
        x = Dense(10 * 1 * 54, activation='relu',
                  kernel_initializer='he_normal', bias_initializer='zeros')(x)
        x = BatchNormalization()(x)
        # Inverse CNN-4
        x = Reshape((10, 1, 54))(x)
        x = Conv2DTranspose(36, (7, 5), activation='relu', padding='same',
                            strides=(3, 2),
                            kernel_initializer='he_normal', bias_initializer='zeros')(x)
        x = BatchNormalization()(x)
        # Inverse CNN-3
        x = Conv2DTranspose(24, (7, 5), activation='relu', padding='same',
                            strides=(3, 2),
                            kernel_initializer='he_normal', bias_initializer='zeros')(x)
        x = BatchNormalization()(x)
        # Inverse CNN
        x = Conv2DTranspose(16, (7, 5), activation='relu', padding='same',
                            strides=(3, 2),
                            kernel_initializer='he_normal', bias_initializer='zeros')(x)
        x = BatchNormalization()(x)
        # Inverse CNN-1
        x = Conv2DTranspose(1, (5, 5), padding='same',
                            strides=(2, 2),
                            kernel_initializer='he_normal', bias_initializer='zeros')(x)
        x = Cropping2D(cropping=(14, 3))(x)
        # unfloor
        x = Unfloor(floor_layer=floor_layer)(x)
        # denormalize
        x = Denormalize()([x, l2])
        # DECODED
        decoded = x

        self.build(input=input, encoded=encoded, l2=l2, decoded=decoded)
