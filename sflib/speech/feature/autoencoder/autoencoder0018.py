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

# 0011が基本（以前のものと等価）
# 全て32チャンネルで統一して，画像の圧縮という観点のみに注力する．
# これの成績が意外と悪くなかった．

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


class SpectrogramImageAutoEncoder0018(SpectrogramImageAutoEncoder):
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
        x = Conv2D(32, (5, 5), activation='relu', padding='same', strides=(2, 2),
                   kernel_initializer='he_normal', bias_initializer='zeros')(x)
        x = BatchNormalization()(x)
        # CNN-2
        x = Conv2D(32, (7, 5), activation='relu', padding='same', strides=(3, 2),
                   kernel_initializer='he_normal', bias_initializer='zeros')(x)
        x = BatchNormalization()(x)
        # CNN-3
        x = Conv2D(32, (7, 5), activation='relu', padding='same', strides=(3, 2),
                   kernel_initializer='he_normal', bias_initializer='zeros')(x)
        x = BatchNormalization()(x)
        # CNN-4
        x = Conv2D(32, (7, 5), activation='relu', padding='same', strides=(3, 2),
                   kernel_initializer='he_normal', bias_initializer='zeros')(x)
        x = BatchNormalization()(x)
        # CNN-5
        x = Conv2D(64, (7, 1), activation='relu', padding='same', strides=(3, 1),
                   kernel_initializer='he_normal', bias_initializer='zeros')(x)
        x = BatchNormalization()(x)
        # CNN-6
        x = Conv2D(128, (4, 1), activation='relu', padding='valid', strides=(1, 1),
                   kernel_initializer='he_normal', bias_initializer='zeros')(x)
        x = BatchNormalization()(x)
        # Reshape
        # x = Flatten()(x)
        x = Reshape((128,))(x)
        x = Activation('tanh')(x)
        # ENCODED
        encoded = x
        # Inverse Reshape
        x = Reshape((1, 1, 128))(x)
        # Inverse CNN-6
        x = Conv2DTranspose(64, (4, 1), activation='relu', padding='valid',
                            strides=(1, 1),
                            kernel_initializer='he_normal', bias_initializer='zeros')(x)
        x = BatchNormalization()(x)
        # Inverse CNN-5
        x = Conv2DTranspose(32, (7, 1), activation='relu', padding='same',
                            strides=(3, 1),
                            kernel_initializer='he_normal', bias_initializer='zeros')(x)
        x = Cropping2D(cropping=(1, 0))(x)
        x = BatchNormalization()(x)
        # Inverse CNN-4
        x = Conv2DTranspose(32, (7, 5), activation='relu', padding='same',
                            strides=(3, 2),
                            kernel_initializer='he_normal', bias_initializer='zeros')(x)
        x = BatchNormalization()(x)
        # Inverse CNN-3
        x = Conv2DTranspose(32, (7, 5), activation='relu', padding='same',
                            strides=(3, 2),
                            kernel_initializer='he_normal', bias_initializer='zeros')(x)
        x = BatchNormalization()(x)
        # Inverse CNN
        x = Conv2DTranspose(32, (7, 5), activation='relu', padding='same',
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
