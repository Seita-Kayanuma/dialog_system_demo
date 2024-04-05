# coding: utf-8
from ... import config
from os import path
import wave
import numpy as np
import os
import subprocess as sp
import re
import shutil


class NictVADConfig:
    # パワー閾値のデフォルト値
    DEFAULT_POWER_THRESHOLD = 8.0

    # 最大ギャップ長のデフォルト値
    DEFAULT_MAXIMUM_GAP_LENGTH = 30

    # 最短発話長のデフォルト値
    DEFAULT_MINIMUM_UTTERENCE_LENGTH = 20

    # プレロール長のデフォルト値
    DEFAULT_PREROLL_LENGTH = 30

    # アフターロール長のデフォルト値
    DEFAULT_AFTERROLL_LENGTH = 30

    def __init__(self):
        self._power_threshold = self.__class__.DEFAULT_POWER_THRESHOLD
        self._maximum_gap_length = self.__class__.DEFAULT_MAXIMUM_GAP_LENGTH
        self._minimum_utterance_length = \
            self.__class__.DEFAULT_MINIMUM_UTTERENCE_LENGTH
        self._preroll_length = self.__class__.DEFAULT_PREROLL_LENGTH
        self._afterroll_length = self.__class__.DEFAULT_AFTERROLL_LENGTH

    @property
    def power_threshold(self):
        return self._power_threshold

    @power_threshold.setter
    def power_threshold(self, value):
        self._power_threshold = value

    @property
    def maximum_gap_length(self):
        return self._maximum_gap_length

    @maximum_gap_length.setter
    def maximum_gap_length(self, value):
        self._maximum_gap_length = value

    @property
    def minimum_utterance_length(self):
        return self._minimum_utterance_length

    @minimum_utterance_length.setter
    def minimum_utterance_length(self, value):
        self._minimum_utterance_length = value

    @property
    def preroll_length(self):
        return self._preroll_length

    @preroll_length.setter
    def preroll_length(self, value):
        self._preroll_length = value

    @property
    def afterroll_length(self):
        return self._afterroll_length

    @afterroll_length.setter
    def afterroll_length(self, value):
        self._afterroll_length = value

    def write(self, filename):
        """設定ファイルに書き出す
        """
        template = """\
#NICTmmse config
	NICTmmse:FilterBankOrder=24
	NICTmmse:FrameLength=20.0
	NICTmmse:FrameShift=10.0
	NICTmmse:SamplingFrequency=16000
	NICTmmse:CutoffHighFrequency=8000.0
	NICTmmse:CutoffLowFrequency=0
	NICTmmse:Preemphasis=0.98
	NICTmmse:CleanSpeechGMM=../model/gmm512_MMSE.20091201
	NICTmmse:InitialNoiseLength=100.0
#NICTmfcc config
	NICTmfcc:FilterBankOrder=20
	NICTmfcc:FrameLength=20.0
	NICTmfcc:FrameShift=10.0
	NICTmfcc:ZeroPadLength=10.0
	NICTmfcc:SamplingFrequency=16000
	NICTmfcc:CutoffHighFrequency=8000.0
	NICTmfcc:CutoffLowFrequency=0.0
	NICTmfcc:Preemphasis=0.98
	NICTmfcc:CepstrumOrder=12
	NICTmfcc:DeltaCepstrumWindow=2
	NICTmfcc:Parameter=pow+cep+dpow+dcep
#NICTcms config
	NICTcms:CepstrumOrder=12
	NICTcms:Parameter=pow+cep+dpow+dcep
	NICTcms:CMSType=online
	NICTcms:AttRate=0.9
	NICTcms:InitMean=../model/meanfile26
	NICTcms:Delay=90

#NICTvad config
	NICTvad:Vad=on
	NICTvad:FrameLength=20
	NICTvad:FrameShift=10.0
	NICTvad:SamplingFrequency=16000
	NICTvad:CepstrumOrder=12
	NICTvad:InputParameter=pow+cep+dpow+dcep
	NICTvad:OutputParameter=cep+dpow+dcep
	NICTvad:PowParameter=pow
	NICTvad:InitialPowerLength=100.0
	NICTvad:PowerThreshold={power_threshold:.1f}

	NICTvad:GMMParameter=cep+dpow+dcep
	NICTvad:NoisyGMM=../model/lab
	NICTvad:NoiseGMM=../model/sil
	NICTvad:PosteriorThreshold=0.2

	NICTvad:ConsecutiveLength=3
	NICTvad:MaximumGapLength={maximum_gap_length:d}
	NICTvad:MinimumUtteranceLength={minimum_utterance_length:d}
	NICTvad:PrerollLength={preroll_length:d}
	NICTvad:AfterrollLength={afterroll_length:d}
	NICTvad:SendSpeechOnly=no
	NICTvad:Tof_eq_StartMark=off
"""
        text = template.format(
            power_threshold=self._power_threshold,
            maximum_gap_length=self._maximum_gap_length,
            minimum_utterance_length=self._minimum_utterance_length,
            preroll_length=self._preroll_length,
            afterroll_length=self._afterroll_length,
        )
        with open(filename, 'w') as f:
            f.write(text)


class NictVAD:
    # このクラス用のデータファイルディレクトリ
    DEFAULT_PATH = path.join(config.get_package_data_dir(__package__))

    # プログラムディレクトリのパス
    VAD_SCRIPT_PATH = path.join('20160115', 'vad_script')

    def __init__(self, conf=None):
        if not conf:
            conf = NictVADConfig()
            # デフォルト値を設定する
            conf.power_threshold = 6.0
            conf.maximum_gap_length = 20
            conf.minimum_utterance_length = 10
            conf.preroll_length = 1
            conf.afterroll_length = 1
            # conf.preroll_length = 30
            # conf.afterroll_length = 30
        self.__config = conf
        self.__script_path = path.join(self.__class__.DEFAULT_PATH,
                                       self.__class__.VAD_SCRIPT_PATH)

    def run(self, filename):
        """VADを実行する．
        
        Args:
          filename: wavファイルのパス

        Returns:
          チャネルごとの (区間開始時間[ms]，区間終了時間[ms]) のリスト．
          結果を r とすると，
          r[0][0][0] が，最初のチャネルの最初の区間の開始時間
          r[0][0][1] が，最初のチャネルの最初の区間の終了時間
          を表す．
        """
        # 音声データを読み込む
        wf = wave.open(filename, 'r')
        channels = wf.getnchannels()
        data = wf.readframes(wf.getnframes())
        wf.close()

        # チャネル毎にwavファイルを書き出す
        tmp_dir_name = path.join(self.__script_path, 'tmp')
        if not path.exists(tmp_dir_name):
            os.mkdir(tmp_dir_name)
        flist_filename = path.join(tmp_dir_name, 'tmp.lst')
        with open(flist_filename, 'w') as f:
            x = np.frombuffer(data, 'int16')
            x = x.reshape(-1, channels).T
            for i in range(channels):
                wavfilename = path.join('tmp', 'tmp{:02d}.wav'.format(i))
                f.write(wavfilename + os.linesep)

                wavfilepath = path.join(self.__script_path, wavfilename)
                data_ = x[i, :].ravel().tobytes()
                wf = wave.open(wavfilepath, 'w')
                wf.setnchannels(1)
                wf.setframerate(16000)
                wf.setsampwidth(2)
                wf.writeframes(data_)
                wf.close()

        # 設定ファイルを書き出す
        # config_file_path = path.join(self.__script_path, 'config_tmp')
        config_file_path = path.join(self.__script_path, 'config_front')
        self.__config.write(config_file_path)

        # 実行する
        os.chdir(self.__script_path)
        p = sp.Popen([
            path.join(self.__script_path, 'vad_flist.bash'), '-C',
            config_file_path, flist_filename
            # path.join(self.__script_path, 'vad_flist.bash'), flist_filename
        ],
                     stdout=sp.PIPE)
        (outs, errs) = p.communicate()
        shutil.rmtree(tmp_dir_name)

        # print(outs)

        # 出力を処理する
        result = []
        vad_list = None
        for line in outs.decode('utf-8').split(os.linesep):
            m = re.search(r'tmp..\.wav', line)
            if m:
                if vad_list:
                    result.append(vad_list)
                vad_list = []
                continue
            if line[:6] == '  #VAD':
                _, _, _, _, start, _, end = line.rstrip().split(' ')
                vad_list.append((
                    int(float(start)),
                    int(float(end)),
                ))
        if vad_list:
            result.append(vad_list)

        return result


nict_vad = NictVAD()


def run_vad(filename):
    return nict_vad.run(filename)
