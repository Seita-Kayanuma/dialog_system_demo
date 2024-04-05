# coding: utf-8
import numpy as np
import wave


class MutipleNoiseAdder:
    def __init__(self, noise_adder_list):
        self.__noise_adder_list = noise_adder_list

    def add_noise(self, x):
        rx = x
        for noise_adder in self.__noise_adder_list:
            rx = noise_adder.add_noise(rx)
        return rx


class NoiseAdder:
    """ノイズのファイルを予め読み込んで，音声に重畳するためのクラス．
    読み込まれたノイズデータは，（必要であれば）最初にシャッフルされたのち，
    連結されたような形で add_noise が呼ばれるたびに順番に重畳される．

    """

    def __init__(self, noise_file_list, shuffle=True):
        """
        Parameters
        ----------
        noise_file_list : list
            音声ファイル（ノイズ）のパスのリスト
        shuffle : bool
            音声ファイルの順序をシャッフルするか否か．
            これを True にした場合も，ファイル単位で
            シャッフルされることに注意されたい．
        """
        self._noise_data = []
        for filename in noise_file_list:
            f = wave.open(filename)
            data = f.readframes(f.getnframes())
            f.close()

            x = np.frombuffer(data, np.int16)
            x = x[:int(len(x) * 0.9)]
            self._noise_data.append(x)

        self._res_noise = np.array([], np.int16)
        self._current_noise_index = 0
        if shuffle is True:
            np.random.shuffle(self._noise_data)

    def add_noise(self, x):
        """
        ノイズを付与する．

        コンストラクタで読み込まれたノイズ波形データを
        順番に連結したものを波形レベルで単純に足しこむ．

        Parameters
        ----------
        x : numpy.ndarray
            ノイズを重畳する（クリーンなな）音声波形
        
        Returns
        -------
        numpy.ndarray
            xと同じサイズの音声波形．
            ノイズが付与されている．
        """
        nx = self._res_noise
        while len(nx) < len(x):
            nx = np.concatenate(
                [nx, self._noise_data[self._current_noise_index]])
            self._current_noise_index = \
                (self._current_noise_index + 1) % len(self._noise_data)
        rx = x + nx[:len(x)]
        self._res_noise = nx[len(x):]
        return rx


class IntermittentNoiseAdder(NoiseAdder):
    """間欠的なノイズ付与器
    """

    def __init__(self,
                 noise_file_list,
                 min_interval=16000,
                 max_interval=16000 * 5,
                 shuffle=True):
        """
        Parameters
        ----------
        noise_file_list : list
            音声ファイル（ノイズ）のパスのリスト
        interval: int
            間隔（サンプル）．デフォルトは 16000 （1秒）
        shuffle : bool
            音声ファイルの順序をシャッフルするか否か．
            これを True にした場合も，ファイル単位で
            シャッフルされることに注意されたい．
        """
        self._noise_data = []
        self._min_interval = min_interval
        self._max_interval = max_interval
        self._shuffle = shuffle
        for filename in noise_file_list:
            f = wave.open(filename)
            data = f.readframes(f.getnframes())
            f.close()
            x = np.frombuffer(data, np.int16)
            self._noise_data.append(x)
        self._res_noise = np.array([], np.int16)
        self._res_silence = 0
        self._current_noise_index = 0
        if shuffle is True:
            np.random.shuffle(self._noise_data)

    def add_noise(self, x):
        """
        ノイズを付与する．

        コンストラクタで読み込まれたノイズ波形データを
        順番に連結したものを波形レベルで単純に足しこむ．

        Parameters
        ----------
        x : numpy.ndarray
            ノイズを重畳する（クリーンなな）音声波形
        
        Returns
        -------
        numpy.ndarray
            xと同じサイズの音声波形．
            ノイズが付与されている．
        """
        rx = x.copy()
        nx = self._res_noise
        p = 0  # 処理済みのサンプル数
        # ノイズが残っていたらまずそれを載せる
        if len(x) > len(nx):
            if len(nx) > 0:
                rx[:len(nx)] += nx
                self._res_noise = np.array([], np.int16)
                p = len(nx)
            self._res_silence = np.random.randint(self._min_interval,
                                                  self._max_interval)
            self._current_noise_index += 1
            if self._current_noise_index == len(self._noise_data):
                self._current_noise_index = 0
                np.random.shuffle(self._noise_data)
        else:
            rx += nx[:len(x)]
            self._res_noise = nx[len(x):]
            return rx
        while p < len(x):
            if len(x) - p > self._res_silence:
                p += self._res_silence
                self._res_silence = 0
            else:
                self._res_silence -= len(x) - p
                return rx
            nx = self._noise_data[self._current_noise_index]
            if len(x) - p > len(nx):
                rx[p:(p + len(nx))] += nx
                self._res_noise = np.array([], np.int16)
                p += len(nx)
                self._res_silence = np.random.randint(self._min_interval,
                                                      self._max_interval)
                self._current_noise_index += 1
                if self._current_noise_index == len(self._noise_data):
                    self._current_noise_index = 0
                    np.random.shuffle(self._noise_data)
            else:
                rx[p:] = nx[:(len(x) - p)]
                self._res_noise = nx[(len(x) - p):]
                return rx
