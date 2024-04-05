# coding: utf-8
# ターン認定の2020年2月バージョン
# Utteranceは廃止．
# Turnの認定の際，最初にある短い発話相当のものは1度に限り無視する．
# また，相手との相互作用によって決めていた部分を廃止．

# 内部的なデータ構造は以下のようにする．
# 時刻は全て音声ファイルの開始時刻を0とするミリ秒単位の時刻．
#
# VAD
#   (開始時刻, 終了時刻) のタプルの列がチャネルごとにある．
#   vad[0][0][0] は，最初のチャネルの最初の区間の開始時刻を表す．
#
# TURN
#   (開始時刻, 終了時刻, 発話種類) のタプルの列がチャネルごとにある．
#   発話種類は，'T'がターン，'S'が短い発話．
#
# 全て，開始時間で昇順にソートされており，同じチャネル内では区間同士に
# 重複はないものとする．
import numpy as np
from os import path
from ...data.elan.data import EafInfo
from ...data.elan.io import write_to_eaf
import wave


def vad2trn(vad,
            THETA_SHORT_PAUSE=1000,
            THETA_KEEP_TURN=1500,
            THETA_SHORT_UTTERANCE=1000):
    """VAD情報をTURN情報に変換する．
    """
    result = []
    for vad_c in vad:
        # print("----")
        res = []
        ts = None  # 直前発話の開始時刻
        te = None  # 直前発話の終了時刻
        r = False  # 短い発話の棄却が済んでいるか
        for tns, tne in vad_c:
            # print(ts, te, r)
            if ts is None:
                ts, te = tns, tne
            else:
                if r is False:
                    if tne - ts <= THETA_SHORT_UTTERANCE:
                        # 統合しても短い発話止まりだったら継続
                        te = tne
                    elif te - ts <= THETA_SHORT_UTTERANCE:
                        # 短い発話はそのまま認定
                        # ただし，短い間の場合は棄却フラグを立てる 
                        res.append((ts, te, 'S'))
                        # if tns - te <= THETA_KEEP_TURN:
                        if tns - te <= THETA_SHORT_PAUSE:
                            r = True
                        ts, te = tns, tne
                    elif tns - te <= THETA_KEEP_TURN:
                        te = tne
                    else:
                        res.append((ts, te, 'T'))
                        ts, te = tns, tne
                else:
                    if tne - ts <= THETA_SHORT_UTTERANCE:
                        # 統合しても短い発話止まりだったら継続
                        te = tne
                    elif tns - te <= THETA_KEEP_TURN:
                        te = tne
                    else:
                        if te - ts <= THETA_SHORT_UTTERANCE:
                            res.append((ts, te, 'S'))
                        else:
                            res.append((ts, te, 'T'))
                        ts, te = tns, tne
                        r = False
        if ts is not None:
            if te - ts <= THETA_SHORT_UTTERANCE:
                res.append((ts, te, 'S'))
            else:
                res.append((ts, te, 'T'))
        result.append(res)
    return result


class DurationInfoV2:
    """
    一つの音声ファイルに対応する各種区間情報を保持するクラス．
    """

    def __init__(self, data, wav_filename):
        self._vad_info = data['VAD']
        self._turn_info = data['TURN']
        self._wav_filename = wav_filename
        self._wav = None

    @property
    def wav_filename(self):
        return self._wav_filename

    @property
    def wav(self):
        if self._wav is None:
            self._read_wav()
        return self._wav

    @property
    def vad(self):
        return self._vad_info

    @property
    def turn(self):
        return self._turn_info

    def clear_cache(self):
        self._wav = None

    def to_eaf(self):
        """
        ELANのEAF情報へ変換する
        """
        eaf_info = EafInfo()
        for ch in range(2):
            for data_type, data_list in (('TURN', self.turn[ch]),
                                         ('VAD', self.vad[ch])):
                tier_name = "{}-{}".format(data_type, ch)
                for values in data_list:
                    start = values[0]
                    end = values[1]
                    if len(values) > 2:
                        anon = values[2]
                    else:
                        anon = ''
                    eaf_info.append_annotation(tier_name, start, end, anon)
        return eaf_info

    def _read_wav(self):
        wf = wave.open(self._wav_filename, 'r')
        channels = wf.getnchannels()
        data = wf.readframes(wf.getnframes())
        x = np.frombuffer(data, dtype=np.int16)
        x = x.reshape(-1, channels).T
        self._wav = x
        wf.close()

    def __repr__(self):
        return "DurationInfoV2(VAD=(%d, %d), TURN=(%d, %d))" \
            % (len(self.vad[0]), len(self.vad[1]),
               len(self.turn[0]), len(self.turn[1]),)


class DurationInfoV2Manager:
    """いくつかの種類（VAD，UTTERANCE，TURN）の音声区間情報を管理するクラス．
    """

    def __init__(self):
        pass

    def get_duration_info_dir_path(self):
        """
        区間情報を保存するディレクトリのパス（フルパス）を取得する．
        継承クラスで実装する必要がある．
        また，このメソッドが呼ばれた以降は当該ディレクトリは存在する
        ものとして実行される（したがって存在しなければ作成する必要がある）．
        """
        raise NotImplementedError()

    def get_wav_filename(self, id):
        pass

    def get_vad_info(self, id):
        """
        IDに対応するVAD区間情報を取得する．
        継承クラスで実装する必要がある．
        VAD区間情報は，音声チャネル毎に(開始時間, 終了時間)のリストを持つ．
        現状ステレオ限定なので，例えば
        info[0][0][0]はチャネル0の最初の発話の開始時間，
        info[1][0][1]はチャネル1の最初の発話の終了時間
        を表すようなデータを返却する必要がある．
        """
        pass

    def build_duration_info(self, id):
        """
        IDに対応する区間情報を構築する．
        """
        vad_info = self.get_vad_info(id)
        turn_info = vad2trn(vad_info)
        info = {'VAD': vad_info, 'TURN': turn_info}
        return DurationInfoV2(info, self.get_wav_filename(id))

    def get_duration_info_path(self, id):
        """
        IDに対応する区間情報ファイルのパスを取得する
        """
        return path.join(self.get_duration_info_dir_path(), id + '.dur.txt')

    def get_duration_info(self, id):
        filename = self.get_duration_info_path(id)
        if not path.exists(filename):
            info = self.build_duration_info(id)
            self._write_duration(info, filename)
        else:
            info = self._read_duration(filename, self.get_wav_filename(id))
        return info

    def get_duration_info_eaf_dir_path(self):
        """
        区間情報を保存するディレクトリのパス（フルパス）を取得する．
        継承クラスで実装する必要がある．
        また，このメソッドが呼ばれた以降は当該ディレクトリは存在する
        ものとして実行される（したがって存在しなければ作成する必要がある）．
        """
        raise NotImplementedError()

    def get_duration_info_eaf_path(self, id):
        """
        IDに対応するEAFファイルの名前を取得する
        """
        return path.join(self.get_duration_info_eaf_dir_path(), id + '.eaf')

    def get_media_info_for_eaf(self, id):
        """
        IDに対応する音声ファイルの位置情報を返す．
        下記のもののタプルである必要がある
        絶対パスURL ... file:/somewhere/in/the/disk/foo.wav
        MIME type ... audio/wav でいいはず
        相対パスURL ... ../wav_safia/foo.wav など（eafファイルから見た相対位置）
        """
        pass

    def write_duration_info_eaf(self, id):
        eaf_info = self.get_duration_info(id).to_eaf()
        media_info = self.get_media_info_for_eaf(id)
        eaf_info.append_media(*media_info)
        write_to_eaf(eaf_info, self.get_duration_info_eaf_path(id))

    def _write_duration(self, info: DurationInfoV2, filename):
        with open(filename, 'w') as f:
            for duration_type, data in (('VAD', info.vad), 
                                        ('TURN', info.turn)):
                f.write(duration_type + "\n")
                for ch, values_list in enumerate(data):
                    f.write("CH %d BEGIN\n" % (ch, ))
                    for values in values_list:
                        f.write(','.join([str(x) for x in values]) + "\n")
                    f.write("CH %d END\n" % (ch, ))

    def _read_duration(self, filename, wav_filename):
        with open(filename, 'r') as f:
            result = []
            for duration_type in ('VAD', 'TURN'):
                if f.readline().rstrip() != duration_type:
                    raise RuntimeError('type not matched')
                result_for_type = []
                for ch in range(2):
                    if f.readline()[:2] != 'CH':
                        raise RuntimeError('channel info is not given')
                    result_for_channel = []
                    while True:
                        line = f.readline()
                        if line[:2] == 'CH':
                            break
                        values = line.rstrip().split(',')
                        # 最初の2つはintに変換しておく
                        values[0] = int(values[0])
                        values[1] = int(values[1])
                        result_for_channel.append(values)
                    result_for_type.append(result_for_channel)
                result.append(result_for_type)
        data = {'VAD': result[0], 'TURN': result[1]}
        return DurationInfoV2(data, wav_filename)
