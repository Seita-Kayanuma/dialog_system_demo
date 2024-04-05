import numpy as np
from .base import PhoneTypeWriterFeatureExtractor
from librosa.feature import mfcc, rmse, delta


class PhoneTypeWriterFeatureExtractorMFCC0001(PhoneTypeWriterFeatureExtractor):
    def __init__(self):
        pass

    def get_filename_base(self):
        s = super().get_filename_base()
        return s

    def calc(self, x):
        x = x / (2**15)
        # MFCCs
        m = mfcc(x, sr=16000, dct_type=3, n_mfcc=16, n_fft=400, hop_length=160)
        dm = delta(m, width=5)
        ddm = delta(dm, width=5)
        # Powers
        p = rmse(y=x, frame_length=400, hop_length=160)
        dp = delta(p, width=5)
        ddp = delta(dp, width=5)
        x = np.concatenate([m.T, dm.T, ddm.T, dp.T, ddp.T], axis=1)
        return x

    def get_feature_dim(self):
        # return self._autoencoder.encoded_dim + 1
        return 16 * 3 + 2
