# coding: utf-8
from keras.layers import Input
from keras.layers import Conv2D, Conv2DTranspose
from keras.layers import Flatten, Dense, Reshape
from keras.layers import BatchNormalization
from .base import FaceAutoEncoder


class FaceAutoEncoder0002(FaceAutoEncoder):
    def __init__(self):
        super().__init__()
        # 入力画像のサイズ（チャネルが最後にくるのに注意）
        input_img = Input(shape=(96, 96, 1), name='input')
        x = input_img
        x = Conv2D(
            32, (3, 3), activation='relu', padding='same', strides=(2, 2))(x)
        x = BatchNormalization()(x)
        x = Conv2D(
            64, (3, 3), activation='relu', padding='same', strides=(2, 2))(x)
        x = BatchNormalization()(x)
        x = Conv2D(
            128, (3, 3), activation='relu', padding='same', strides=(2, 2))(x)
        x = BatchNormalization()(x)
        x = Conv2D(
            256, (3, 3), activation='relu', padding='same', strides=(2, 2))(x)
        x = BatchNormalization()(x)
        x = Flatten()(x)
        x = Dense(1028, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dense(256, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dense(32, activation='tanh', name='encoded_out')(x)
        encoded_out = x
        x = Dense(256, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dense(1028, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dense(6 * 6 * 256, activation='relu')(x)
        x = Reshape((6, 6, 256))(x)
        x = BatchNormalization()(x)
        x = Conv2DTranspose(
            256, (3, 3), activation='relu', padding='same', strides=(2, 2))(x)
        x = BatchNormalization()(x)
        x = Conv2DTranspose(
            128, (3, 3), activation='relu', padding='same', strides=(2, 2))(x)
        x = BatchNormalization()(x)
        x = Conv2DTranspose(
            64, (3, 3), activation='relu', padding='same', strides=(2, 2))(x)
        x = BatchNormalization()(x)
        x = Conv2DTranspose(
            32, (3, 3), activation='relu', padding='same', strides=(2, 2))(x)
        x = BatchNormalization()(x)
        x = Conv2D(
            1, (3, 3), activation='sigmoid', padding='same', name='decoded')(x)
        decoded = x

        self.build(
            input=input_img,
            encoded_out=encoded_out,
            encoded_in=encoded_out,
            decoded=decoded)
