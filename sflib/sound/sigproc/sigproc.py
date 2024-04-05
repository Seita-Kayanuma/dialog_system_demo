# coding: utf-8
import numpy as np
import copy
import wave


def calc_power_spectrum(x, fftsize=1024, window=None):
    """
    パワースペクトログラムを計算する．

    Parameters
    ----------
    x : numpy.ndarray
        要素の型はint16．
    fftsize : int
        FFTのサイズ．xの長さより大きくなければならない．
    window : numpy.ndarray
        窓関数．xと同じ長さでなければならない．
        与えられたなかった場合はハニング窓が利用される．
    
    Returns
    -------
    numpy.ndarray
        fftsizeの半分の長さのパワースペクトルであり，
        折り返し分を含まない．
        また，フロアリングと対数をかける処理，
        具体的には :math:`x' = 20 \log_{10} (x+10^{-40})` 
        を行い出力する．
    """
    if window is None:
        window = np.hanning(len(x))
    wx = x * window
    data = np.hstack([wx, np.zeros(fftsize - len(wx))])
    y = np.fft.rfft(data) / fftsize
    psd = np.abs(y) + 1e-40
    psd = 20.0 * np.log10(psd)
    return psd[:(fftsize // 2)]
