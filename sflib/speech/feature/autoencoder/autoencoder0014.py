# coding: utf-8
from keras.layers import Input
from keras.layers import Conv2D, Conv2DTranspose, Cropping2D
from keras.layers import Flatten, Dense, Reshape
from keras.layers import BatchNormalization, Layer
from keras.layers import Concatenate
from keras.layers import Activation
from keras.layers import GlobalAveragePooling2D, Add
import keras.backend as K
from .base import SpectrogramImageAutoEncoder
import numpy as np

# ResNet風にする（入力を適宜Addしていく）
# チャネル数はそこそこ，ストライドも最小限に留めて多層化を行う．
# これは学習時間だけかかってよい結果も出なさそうだったのでボツとした．

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

def buildResNetBlock(x, num_channels, kernel_size, strides):
    x_in = x
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(num_channels, kernel_size, padding='same', strides=strides,
               kernel_initializer='he_normal', bias_initializer='zeros')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(num_channels, kernel_size, padding='same', strides=(1, 1),
               kernel_initializer='he_normal', bias_initializer='zeros')(x)
    x = BatchNormalization()(x)
    x_residual = Activation('relu')(x)
    x_shortcut = Conv2D(num_channels, kernel_size=(1, 1),
                        strides=strides, padding='same',
                        kernel_initializer='he_normal', bias_initializer='zeros')(x_in)
    x = Add()([x_residual, x_shortcut])
    return x
    
def buildResNetBlockDeconv(x, num_channels, kernel_size, strides):
    x_in = x
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2DTranspose(num_channels, kernel_size, padding='same', strides=strides,
                        kernel_initializer='he_normal', bias_initializer='zeros')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2DTranspose(num_channels, kernel_size, padding='same', strides=(1, 1),
                        kernel_initializer='he_normal', bias_initializer='zeros')(x)
    x = BatchNormalization()(x)
    x_residual = Activation('relu')(x)
    x_shortcut = Conv2DTranspose(num_channels, kernel_size=(1, 1),
                                 strides=strides, padding='same',
                                 kernel_initializer='he_normal', bias_initializer='zeros')(x_in)
    x = Add()([x_residual, x_shortcut])
    return x
    
class SpectrogramImageAutoEncoder0014(SpectrogramImageAutoEncoder):
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
        x = buildResNetBlock(x, 16, (3, 3), (2, 1))
        # CNN-2
        x = buildResNetBlock(x, 16, (3, 3), (2, 2))
        # CNN-3
        x = buildResNetBlock(x, 32, (3, 3), (2, 1))
        # CNN-4
        x = buildResNetBlock(x, 32, (3, 3), (2, 2))
        # CNN-5
        x = buildResNetBlock(x, 64, (3, 3), (2, 1))
        # CNN-6
        x = buildResNetBlock(x, 64, (3, 3), (2, 2))
        # CNN-7
        x = buildResNetBlock(x, 128, (3, 3), (2, 1))
        # CNN-8
        x = buildResNetBlock(x, 128, (3, 3), (2, 2))
        # CNN-9
        x = buildResNetBlock(x, 256, (3, 3), (2, 1))
        # FC-1
        x = Flatten()(x)
        x = Dense(128, activation='relu',
                  kernel_initializer='he_normal', bias_initializer='zeros')(x)
        x = BatchNormalization()(x)
        # FC-2
        x = Dense(64, 
                  kernel_initializer='he_normal', bias_initializer='zeros')(x)
        x = Activation('tanh')(x)
        # ENCODED
        encoded = x
        # Inverse FC-2
        x = Dense(128, activation='relu',
                  kernel_initializer='he_normal', bias_initializer='zeros')(x)
        x = BatchNormalization()(x)
        # Inverse FC-1
        x = Dense(1 * 1 * 256, activation='relu',
                  kernel_initializer='he_normal', bias_initializer='zeros')(x)
        x = BatchNormalization()(x)
        # Inverse CNN-9
        x = Reshape((1, 1, 256))(x)
        x = buildResNetBlockDeconv(x, 128, (3, 3), (2, 1))
        # Inverse CNN-8
        x = buildResNetBlockDeconv(x, 128, (3, 3), (2, 2))
        # Inverse CNN-7
        x = buildResNetBlockDeconv(x, 64, (3, 3), (2, 1))
        # Inverse CNN-6
        x = buildResNetBlockDeconv(x, 64, (3, 3), (2, 2))
        # Inverse CNN-5
        x = buildResNetBlockDeconv(x, 32, (3, 3), (2, 1))
        # Inverse CNN-4
        x = buildResNetBlockDeconv(x, 32, (3, 3), (2, 2))
        # Inverse CNN-3
        x = buildResNetBlockDeconv(x, 16, (3, 3), (2, 1))
        # Inverse CNN-2
        x = buildResNetBlockDeconv(x, 16, (3, 3), (2, 2))
        # Inverse CNN-1
        x = buildResNetBlockDeconv(x, 1, (3, 3), (2, 1))
        # Cropping
        x = Cropping2D(cropping=(0, 3))(x)
        # unfloor
        x = Unfloor(floor_layer=floor_layer)(x)
        # denormalize
        x = Denormalize()([x, l2])
        # DECODED
        decoded = x

        self.build(input=input, encoded=encoded, l2=l2, decoded=decoded)
