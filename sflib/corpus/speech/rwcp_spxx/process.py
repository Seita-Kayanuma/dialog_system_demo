# coding: utf-8
from . import RWCP_SPXX
from .util import dat2vad, dat2vadphones
from ...sound.safia import apply_safia
from os import path
import os
import wave
import numpy as np
from sflib.sound.sigproc.spec_image \
    import SpectrogramImageArray, SpectrogramImageGenerator


class VADInfo:
    VAD_INFO_DIR_NAME = 'vad'

    def __init__(self, rwcp=None):
        if rwcp is None:
            self._rwcp = RWCP_SPXX()
        else:
            self._rwcp = rwcp
        self.__vad_dir_path = path.join(self._rwcp.get_data_path(),
                                        self.__class__.VAD_INFO_DIR_NAME)
        if not path.exists(self.__vad_dir_path):
            os.makedirs(self.__vad_dir_path, mode=0o755, exist_ok=True)

    def get_vad_info_path(self, id):
        return path.join(self.__vad_dir_path, id + ".vad.txt")

    def get_vad_info(self, id):
        filename = self.get_vad_info_path(id)
        if not path.exists(filename):
            info = dat2vad(self._rwcp.get_dat_path(id))
            self._write_vad(info, filename)
        else:
            info = self._read_vad(filename)
        return info

    def _write_vad(self, info, filename):
        with open(filename, "w") as f:
            for key in ('VAD-L', 'VAD-R', 'UTT-L', 'UTT-R'):
                for s, e, d in info[key]:
                    f.write("%s,%d,%d,%d\n" % (key, s, e, d))

    def _read_vad(self, filename):
        vad_l = []
        vad_r = []
        utt_l = []
        utt_r = []
        with open(filename, "r") as f:
            while True:
                line = f.readline()
                if line is None or len(line) == 0:
                    break
                key, s, e, d = line.rstrip().split(',')
                if key == 'VAD-L':
                    vad_l.append((int(s), int(e), int(d)))
                elif key == 'VAD-R':
                    vad_r.append((int(s), int(e), int(d)))
                elif key == 'UTT-L':
                    utt_l.append((int(s), int(e), int(d)))
                elif key == 'UTT-R':
                    utt_r.append((int(s), int(e), int(d)))
        return {'VAD-L': vad_l, 'VAD-R': vad_r, 'UTT-L': utt_l, 'UTT-R': utt_r}


class WavDataWithSafia:
    WAV_SAFIA_DIR_NAME = 'wav_safia'

    def __init__(self, rwcp=None):
        if rwcp is None:
            self._rwcp = RWCP_SPXX()
        else:
            self._rwcp = rwcp
        self.__wav_safia_dir_path = path.join(
            self._rwcp.get_data_path(), self.__class__.WAV_SAFIA_DIR_NAME)
        if not path.exists(self.__wav_safia_dir_path):
            os.makedirs(self.__wav_safia_dir_path, mode=0o755, exist_ok=True)

    def get_wav_path(self, id):
        return path.join(self.__wav_safia_dir_path, id + '.wav')

    def get_wav_data(self, id):
        filename = self.get_wav_path(id)
        if not path.exists(filename):
            # オリジナル音声データの読み込み
            wf = wave.open(self._rwcp.get_wav_path(id), 'r')
            channels = wf.getnchannels()
            data = wf.readframes(wf.getnframes())
            wf.close()
            # SAFIAをかける
            x = np.frombuffer(data, 'int16')
            x = x.reshape(-1, channels).T
            x_safia = apply_safia(x)
            data_safia = x_safia.T.ravel().tobytes()
            # 書き込み
            wf = wave.open(filename, 'w')
            wf.setnchannels(channels)
            wf.setframerate(16000)
            wf.setsampwidth(2)
            wf.writeframes(data_safia)
            wf.close()
        else:
            wf = wave.open(filename, 'r')
            channels = wf.getnchannels()
            data = wf.readframes(wf.getnframes())
            wf.close()

            x = np.frombuffer(data, 'int16')
            x = x.reshape(-1, channels).T
            x_safia = x

        return x_safia


class WavDataWithVAD:
    def __init__(self, rwcp=None):
        if rwcp is None:
            self._rwcp = RWCP_SPXX()
        else:
            self._rwcp = rwcp
        self._vi = VADInfo(self._rwcp)
        self._ws = WavDataWithSafia(self._rwcp)

    def get_wav_data_list(self, id, tag='VAD-L'):
        # VAD情報を読み込む
        vad = self._vi.get_vad_info(id)
        vad_info = vad[tag]

        # 音声データを取得
        wav = self._ws.get_wav_data(id)

        # tagからチャネルを判断
        if tag[-2:] == '-L':
            channel = 0
        elif tag[-2:] == '-R':
            channel = 1
        else:
            raise RuntimeError(
                "cannot determine the channel from the tag '%s'" % tag)
        wav = wav[channel]

        # 分割しつつ返す
        results = []
        rate = 16000  # TODO 16kHz限定
        for s, e, d in vad_info:
            si = (s * rate) // 1000
            ei = (e * rate) // 1000
            if ei - si > 0:
                results.append(wav[si:ei].copy())
        return results


class SpectrogramImageArrayWithVAD:
    def __init__(self,
                 rwcp=None,
                 tag='UTT-L',
                 cond='T',
                 max_wav_num_for_each_file=None,
                 shuffle=True):
        if rwcp is None:
            self._rwcp = RWCP_SPXX()
        else:
            self._rwcp = rwcp
        self._wav_data = WavDataWithVAD(self._rwcp)
        self._tag = tag
        self._cond = cond
        self._max_wav_num_for_each_file = max_wav_num_for_each_file
        self._shuffle = shuffle

    def construct(self):
        id_list = self._rwcp.get_id_list(self._cond)
        print("%d ids found." % len(id_list))

        gen = SpectrogramImageGenerator()
        total_image_count = 0
        sa_list = []
        image_index_base_list = []
        for i, id in enumerate(id_list):
            print(i, id)
            wavs = self._wav_data.get_wav_data_list(id, self._tag)
            if self._max_wav_num_for_each_file is not None and \
               self._max_wav_num_for_each_file < len(wavs):
                if self._shuffle is True:
                    np.random.shuffle(wavs)
                wavs = wavs[:self._max_wav_num_for_each_file]
            for wav in wavs:
                sa = SpectrogramImageArray(wav, gen)
                sa_list.append(sa)
                image_index_base_list.append(total_image_count)
                total_image_count += len(sa)
        self._sa_list = sa_list
        self._index_base_list = np.array(image_index_base_list, 'int')
        self._total_image_count = total_image_count

    def __len__(self):
        return self._total_image_count

    def __getitem__(self, key):
        if isinstance(key, int):
            return self.get_item_by_index(key)
        elif isinstance(key, slice):
            r = [
                self.get_item_by_index(i)
                for i in range(*(key.indices(len(self))))
            ]
            return tuple(r)
        else:
            raise ValueError()

    def get_item_by_index(self, i):
        ii = i - self._index_base_list
        sa_index = np.where(ii >= 0)[0][-1]
        local_index = ii[sa_index]
        return self._sa_list[sa_index][local_index]


class VadPhonesInfo:
    VAD_PHONES_INFO_DIR_NAME = "vad_phones"

    def __init__(self, rwcp=None):
        if rwcp is None:
            self._rwcp = RWCP_SPXX()
        else:
            self._rwcp = rwcp
        self.__vad_phones_dir_path = path.join(
            self._rwcp.get_data_path(),
            self.__class__.VAD_PHONES_INFO_DIR_NAME)
        if not path.exists(self.__vad_phones_dir_path):
            os.makedirs(self.__vad_phones_dir_path, mode=0o755, exist_ok=True)

    def get_vad_phones_info_path(self, id):
        return path.join(self.__vad_phones_dir_path, id + ".vad_phones.txt")

    def get_vad_phones_info(self, id):
        filename = self.get_vad_phones_info_path(id)
        if not path.exists(filename):
            try:
                info = dat2vadphones(self._rwcp.get_dat_path(id))
            except RuntimeError as e:
                print("%s" % self._rwcp.get_dat_path(id))
                raise e
            self._write_vad_phones(info, filename)
        else:
            info = self._read_vad_phones(filename)
        return info

    def _write_vad_phones(self, info, filename):
        with open(filename, "w") as f:
            for key, vads in info.items():
                for s, e, d, phones in vads:
                    f.write(
                        "%s,%d,%d,%d,%s\n" % (key, s, e, d, ' '.join(phones)))

    def _read_vad_phones(self, filename):
        info = {'VAD-L': [], 'VAD-R': [], 'UTT-L': [], 'UTT-R': []}
        with open(filename, "r") as f:
            while True:
                line = f.readline()
                if line is None or len(line) == 0:
                    break
                key, s, e, d, phones = line.rstrip().split(',')
                info[key].append((int(s), int(e), int(d), phones.split(' ')))
        return info


class WavDataWithVADPhones:
    def __init__(self, rwcp=None):
        if rwcp is None:
            self._rwcp = RWCP_SPXX()
        else:
            self._rwcp = rwcp
        self._vpi = VadPhonesInfo(self._rwcp)

    def get_wav_data_and_phones_list(self, id, tag='VAD-L'):
        # VAD情報を読み込む
        info = self._vpi.get_vad_phones_info(id)

        vads = info[tag]
        channel = 0
        if tag[-2:] == '-R':
            channel = 1

        # 音声ファイルを読み込んで np.int16 形式にする
        wav_file_path = self._rwcp.get_wav_path(id)
        wf = wave.open(wav_file_path, 'r')
        wav = wf.readframes(wf.getnframes())
        rate = wf.getframerate()
        wf.close()
        wav = np.frombuffer(wav, 'int16')
        wav = wav.reshape(-1, 2).T
        wav = wav[channel]

        # 分割しつつ返す
        results = []
        for s, e, d, phones in vads:
            si = (s * rate) // 1000
            ei = (e * rate) // 1000
            if ei - si > 0:
                results.append((wav[si:ei].copy(), phones))
        return results


class SpectrogramImageArrayWithVADPhones:
    def __init__(self,
                 rwcp=None,
                 tag='VAD-L',
                 cond='T',
                 max_wav_num_for_each_file=None,
                 shuffle=True,
                 image_shift=2,
                 noise_adder=None):
        if rwcp is None:
            self._rwcp = RWCP_SPXX()
        else:
            self._rwcp = rwcp
        self._wav_phones_data = WavDataWithVADPhones(self._rwcp)
        if isinstance(tag, list):
            self._tags = tag
        else:
            self._tags = [tag]
        self._cond = cond
        self._max_wav_num_for_each_file = max_wav_num_for_each_file
        self._shuffle = shuffle
        self._gen = SpectrogramImageGenerator(image_shift=image_shift)
        self._noise_adder = noise_adder

    def construct(self):
        id_list = self._rwcp.get_id_list(self._cond)
        print("%d ids found." % len(id_list))

        item_list = []
        for i, id in enumerate(id_list):

            # 10フレーム分は入力があるようにする
            wav_len_thresh = self._gen.num_samples_per_image + \
                             self._gen.num_samples_per_image_shift * 9

            # print("GEN: %d, %d" % \
            #       (self._gen.num_samples_per_image, self._gen.num_samples_per_image_shift))
            def get_estimated_input_len(wav_length):
                return (wav_length - self._gen.num_samples_per_image) \
                    // self._gen.num_samples_per_image_shift + 1

            data = []
            for tag in self._tags:
                data.extend(
                    self._wav_phones_data.get_wav_data_and_phones_list(
                        id, tag))

            if self._max_wav_num_for_each_file is not None and \
               self._max_wav_num_for_each_file < len(data):
                if self._shuffle is True:
                    np.random.shuffle(data)
            this_item_list = []
            for wav, phones in data:
                # print("%d, %d" % (len(wav), get_estimated_input_len(len(wav))))
                if len(wav) < wav_len_thresh:
                    continue
                if len(phones) < 1:
                    continue
                input_len = get_estimated_input_len(len(wav))
                if input_len < len(phones):
                    print("INPUT LENGTH(%d) IS LESS THAN PHONES(%d)" %
                          (input_len, len(phones)))
                    continue
                this_item_list.append((wav, phones))

            if self._max_wav_num_for_each_file is not None and \
               self._max_wav_num_for_each_file < len(data):
                this_item_list = this_item_list[:self.
                                                _max_wav_num_for_each_file]

            print(i, id, len(this_item_list))
            item_list.extend(this_item_list)

        self._item_list = item_list

    def __len__(self):
        return len(self._item_list)

    def __getitem__(self, key):
        if isinstance(key, int):
            return self.get_item_by_index(key)
        elif isinstance(key, slice):
            r = [
                self.get_item_by_index(i)
                for i in range(*(key.indices(len(self))))
            ]
            return tuple(r)
        else:
            raise ValueError()

    def get_item_by_index(self, i):
        wav, phones = self._item_list[i]

        # 必要に応じてゼロ詰めをする
        num_samples = len(wav)
        num_samples -= self._gen.num_samples_per_image
        if num_samples < 0:
            wav = np.concatenate([wav, np.zeros(-num_samples, np.int16)])
        else:
            residual_num_samples = num_samples % self._gen.num_samples_per_image_shift
            if residual_num_samples > 0:
                n = self._gen.num_samples_per_image_shift - residual_num_samples
                wav = np.concatenate([wav, np.zeros(n, np.int16)])
        if self._noise_adder is not None:
            wav = self._noise_adder.add_noise(wav)
        self._gen.reset()
        images = np.stack(self._gen.input_wave(wav))
        # print("len(wav)=%d, images.shape=%d" % (len(wav), images.shape[0]))

        return (images, phones)
