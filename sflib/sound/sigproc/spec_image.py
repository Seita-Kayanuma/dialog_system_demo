# coding: utf-8
import numpy as np
from . import sigproc as sp


class SpectrogramImageGenerator:
    """
    音声データを与えて画像化するクラス．

    音声を順番に与えることで，
    必要なサンプル数が集まった時点で画像を生成してリストを返す．
    1枚分のデータをまとめて与えて計算させることも可能．
    """

    def __init__(self,
                 framesize=800,
                 frameshift=160,
                 fftsize=1024,
                 image_width=10,
                 image_height=None,
                 image_shift=5):
        """
        Parameters
        ----------
        framesize : int
            音声のフレームサイズ．デフォルト800（50ms）．
        frameshift : int
            音声のフレームシフト．デフォルト160（10ms）．
        fftsize : int
            FFTのサイズ．デフォルト1024．
        image_width : int
            画像の横幅（何個スペクトルを並べるか）．デフォルト10．
        image_height : int
            画像の縦幅．Noneの場合はfftsize/2になる.
        image_shift : int
            画像のシフト．デフォルト5．
        """
        self._framesize = framesize
        self._frameshift = frameshift
        self._fftsize = fftsize
        if image_height is None or image_height >= self._fftsize // 2:
            self._image_height = self._fftsize // 2
        else:
            self._image_height = image_height
        self._image_width = image_width
        self._image_shift = image_shift

        self._window = np.hanning(self._framesize)

        self._current_wave = np.int16([])
        self._current_image = np.zeros((self._image_height, 0))
        self._current_spec_count = 0

    @property
    def num_samples_per_image(self):
        """1枚の画像を作成するのに必要なサンプル数"""
        return self._framesize + self._frameshift * (self._image_width - 1)

    @property
    def num_samples_per_image_shift(self):
        return self._frameshift * self._image_shift

    @property
    def frame_size(self):
        return self._framesize

    @property
    def frame_shift(self):
        return self._frameshift

    @property
    def image_width(self):
        return self._image_width

    @property
    def image_shift(self):
        return self._image_shift

    @property
    def image_size(self):
        return (self._image_height, self._image_width)

    def reset(self):
        u"""状態をリセットする"""
        self._current_wave = np.int16([])
        self._current_image = np.zeros((self._image_height, 0))
        self._current_spec_count = 0

    def input_wave(self, x):
        u"""波形を入力して（計算可能であれば）画像を出力する
        
        Parameters
        ----------
        x: 波形データ np.array (dtype=np.int16)

        Returns
        -------
        生成された画像のリスト．空の場合もある．
        """
        self._current_wave = np.concatenate([self._current_wave, x])

        ret_image_list = []
        while len(self._current_wave) >= self._framesize:
            extracted_x = self._current_wave[0:self._framesize]
            # スペクトル計算
            spec = sp.calc_power_spectrum(extracted_x, self._fftsize,
                                          self._window)
            spec = spec[:self._image_height]
            # 画像の後ろにつける
            self._current_image = np.hstack([
                self._current_image,
                np.reshape(spec, (self._image_height, 1))
            ])
            # self._current_spec_count += 1
            # if self._current_spec_count % self._image_shift == 0:
            # 幅が画像1枚分になったら返却列に（コピーを）追加する
            if self._current_image.shape[1] >= self._image_width:
                ret_image_list.append(self._current_image.copy())
                # 画像シフト分だけ前の列を消す
                self._current_image = self._current_image[:, self.
                                                          _image_shift:]

            # 音声をシフト
            self._current_wave = self._current_wave[self._frameshift:]

        return ret_image_list


class SpectrogramImageArray:
    """音声データを与え画像列を生成するクラス．
    画像生成はその位置画像が要求された時に行う．"""

    def __init__(self, x, generator):
        """コンストラクタ

        Parameters
        ----------
        x: 音声データ（np.array, np.int16）
        generator: 画像生成器
        """
        self._x = x
        self._generator = generator

        # 生成できる画像の数を予め求めておく
        # 1枚の画像を生成するのに必要なサンプル数
        num_samples_per_image = self._generator.num_samples_per_image
        # 画像シフト分のサンプル数
        num_samples_per_image_shift = \
            self._generator.image_shift * self._generator.frame_shift
        # 持ってるデータのサンプル数
        num_samples = len(x)

        if num_samples < num_samples_per_image:
            # サンプル数が1枚の画像分に満たない場合は1枚だけ生成可能
            self._num_images = 1
        else:
            # それ以上の場合は，最初のフレームの分と
            # 残りは画像シフト分だけ生成可能
            self._num_images = \
                (num_samples - num_samples_per_image) \
                // num_samples_per_image_shift + 1

        self.__num_samples_per_image = num_samples_per_image
        self.__num_samples_per_image_shift = num_samples_per_image_shift

    def get_wav_data(self):
        return self._x
        
    def __len__(self):
        """生成できる画像数"""
        return self._num_images

    def __getitem__(self, index):
        """画像の取得"""
        # 範囲外だったらエラー
        if index >= self._num_images:
            raise IndexError()
        # 開始インデクスと終了インデクス
        s = self.__num_samples_per_image_shift * index
        e = s + self.__num_samples_per_image
        if e > len(self._x):
            e = len(self._x)
        # 切り出し
        sub_samples = self._x[s:e]
        # 足りない分は0を詰める
        n = self.__num_samples_per_image - len(sub_samples)
        if n > 0:
            # import ipdb; ipdb.set_trace()
            sub_samples = np.concatenate([sub_samples,
                                          np.zeros(n, np.int16)])
        # 画像の生成
        self._generator.reset()
        l_img = self._generator.input_wave(sub_samples)
        return l_img[0]
