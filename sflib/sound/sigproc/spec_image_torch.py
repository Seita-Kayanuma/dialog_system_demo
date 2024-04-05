# coding: utf-8
import numpy as np
import torch


class SpectrogramImageGeneratorTorch:
    """
    音声データを与えて画像化するクラス(PyTorchバージョン)．

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

        self._window = torch.hann_window(self._framesize)

        self._null_image = torch.zeros(
            (0, self._image_height, self._image_width), dtype=torch.float32)
        self._current_wave = torch.tensor([], dtype=torch.float32)
        self._current_image = torch.zeros((self._image_height, 0))
        self._current_spec_count = 0

    def to(self, device):
        self._null_image = self._null_image.to(device)
        self._current_wave = self._current_wave.to(device)
        self._current_image = self._current_image.to(device)

    @property
    def device(self):
        return self._null_image.device
        
    @property
    def num_samples_per_image(self):
        """1枚の画像を作成するのに必要なサンプル数"""
        # return self._framesize + self._frameshift * (self._image_width - 1)
        # torch.stftの仕様上，FFT Size分のサンプルが無いとFFTが実施できない
        return self._fftsize + self._frameshift * (self._image_width - 1)

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
        self._current_wave = self._current_wave[:0].clone().detach()
        self._current_image = self._current_image[:, :0].clone().detach()
        self._current_spec_count = 0

    def input_wave(self, x: np.array, images_required: bool = True):
        """波形を入力し，計算可能であればSTFTを計算する．
        また，要望があれば画像のバッチを生成して返却する．
        
        Parameters
        ----------
        x: 波形データ np.array (dtype=np.int16)
        images_required: Trueの場合は，即画像のバッチが返却される． 
           詳細は get_images メソッドを参照．
        
        Returns
        -------
        生成された画像のリスト．空の場合もある．
        """
        
        if len(x) > 0:
            x_tensor = torch.tensor(x, dtype=torch.float32).to(self.device)
            self._current_wave = torch.cat([self._current_wave, x_tensor])
        # STFT可能な部分を切り出す
        #  最低 fftsize (ex. 1024) だけ必要
        #  + frameshift (ex. 160) の倍数だけあれば，複数フレーム出力可能
        #  例えば， 1024 + 160 * 3 = 1024 + 480 = 10504 だけあれば4フレーム分STFTができる
        #  この時， 0 〜 160 * 3 = 480 までのサンプルは使用済となるので消去し，
        #  次回用には 480 以降のサンプルを残しておく．

        if len(self._current_wave) >= self._fftsize:
            num_frames = (len(self._current_wave) - self._fftsize) // self._frameshift + 1
            idx_end = self._fftsize + self._frameshift * num_frames
            x = self._current_wave[:idx_end].clone().detach()
            # FFTを計算する
            self._window = self._window.to(self.device)
            y = torch.stft(x, self._fftsize, hop_length=self._frameshift,
                           win_length=self._framesize, window=self._window,
                           center=False)
            # y の shape は，(fftsize//2 + 1, L, 2) 最後の2次元は複素数分
            # fftsize//2 + 1 の最後の要素は使わないので省く（本当は使った方がいい？）
            y = y[:self._image_height, :, :]
            # パワースペクトルに直す
            y = ((y / self._fftsize).pow(2.0).sum(dim=2).sqrt() + 1e-40).log10() * 20.0
            # 時間と周波数軸が逆なので直す（本当は時間が0次元目の方がいいかも...?）
            # y = y.t()
            # 画像の後ろにつける
            self._current_image = torch.cat([self._current_image, y], dim=1).clone().detach()
            # 残りの音声を保存
            self._current_wave = self._current_wave[(self._frameshift * num_frames):]

        if images_required:
            return self.get_images()

    @property
    def num_images_ready(self):
        current_width = self._current_image.shape[1]
        if current_width < self._image_width:
            return 0
        num_images = (current_width - self._image_width) // self._image_shift + 1
        return num_images
        
    def get_images(self, num_images=None):
        """画像のバッチが返せれば返せる分だけ返す．
        """
        num_images_ready = self.num_images_ready
        if num_images_ready == 0:
            return self._null_image
        if num_images is None or num_images > num_images_ready:
            num_images = num_images_ready
            
        image_list = []
        for i in range(num_images):
            idx_start = i * self._image_shift
            idx_end = idx_start + self._image_width
            image_list.append(self._current_image[:, idx_start:idx_end])
        batch = torch.stack(image_list, dim=0).detach()
        self._current_image = \
            self._current_image[:, (num_images * self._image_shift):]
        return batch
