# coding: utf-8
from ...lang.yomi.mecab_yomi import get_yomi
from ...lang.yomi.voca import yomi2voca
import re

patt_num = re.compile('[0-9]+')

def dat2vad(filename, max_gap=700):
    """
    RWCP-SPxxのdatファイルからVAD情報に直す．

    datファイルは以下のような形になっている．
    
    0001 
    1A
    B
    980
    1620
    いらっしゃいませ。
    iraqshaimase.
    #
    0002
    5A
    B
    1830
    2420
    えー
    e-
    #
    0003
    2A
    B
    2420
    3930
    それでは，
    soredewa,
    #

    (1) 発話番号 ... 0001
    (2) フラグ（今回は使わない） ... 1A
    (3) 話者（Aがチャネル1，Bがチャネル2） ... B
    (4) 発話の開始時刻（ミリ秒） ... 980
    (5) 発話の終了時刻（ミリ秒） ... 1620
    (6) 発話書き起こし（仮名漢字） ... いらっしゃいませ。
    (7) 発話書き起こし（ローマ字表記） ... iraqshaimase.
    (8) ターミネータ ... #
    """
    vad_dict = {'VAD-L': [], 'VAD-R': []}
    with open(filename, 'rb') as f:
        while True:
            # (1) 発話番号（使わない）
            line = f.readline().decode('utf-8')
            # 読み込めなければ終了
            if not line:
                break
            # (2) フラグ（使わない）
            line = f.readline()
            # (3) 話者（AならR，BならL）
            line = f.readline().decode('utf-8')
            speaker = line[0]
            if speaker == 'B':
                key = 'VAD-L'
            elif speaker == 'A':
                key = 'VAD-R'
            else:
                raise RuntimeError("Unknown Speaker %s" % speaker)
            # print(line, speaker)
            # (4) 発話開始時間（改行文字を消さなくても整数には直せるらしい）
            start_time = int(f.readline().decode('utf-8'))
            # (5) 発話終了時間
            end_time = int(f.readline().decode('utf-8'))
            # (6) 発話書き起こし（仮名漢字） （使わない）
            line = f.readline()
            # (7) 発話書き起こし（ローマ字表記） （使わない）
            line = f.readline()
            # (8) ターミネータ （使わない）
            line = f.readline()

            vad_dict[key].append([start_time, end_time, end_time - start_time])
        utt_dict = {'UTT-L': [], 'UTT-R': []}
    for channel in 'L', 'R':
        vads = vad_dict['VAD-' + channel]

        start = None
        end = None
        for vad in vads:
            if start is None:
                start = vad[0]
                end = vad[1]
                continue
            if vad[0] - end < max_gap:
                end = vad[1]
                continue
            dur = end - start
            utt_dict['UTT-' + channel].append([start, end, dur])
            start = vad[0]
            end = vad[1]
            dur = end - start
    vad_dict.update(utt_dict)
    return vad_dict


def dat2vadphones(filename, max_gap=700):
    vad_dict = {'VAD-L': [], 'VAD-R': []}
    with open(filename, 'rb') as f:
        while True:
            # (1) 発話番号（使わない）
            line = f.readline().decode('utf-8')
            # 読み込めなければ終了
            if not line:
                break
            # (2) フラグ（使わない）
            line = f.readline()
            # (3) 話者（AならR，BならL）
            line = f.readline().decode('utf-8')
            speaker = line[0]
            if speaker == 'B':
                key = 'VAD-L'
            elif speaker == 'A':
                key = 'VAD-R'
            else:
                raise RuntimeError("Unknown Speaker %s" % speaker)
            # print(line, speaker)
            # (4) 発話開始時間（改行文字を消さなくても整数には直せるらしい）
            start_time = int(patt_num.search(f.readline().decode('utf-8'))[0])
            # (5) 発話終了時間
            end_time = int(patt_num.search(f.readline().decode('utf-8'))[0])
            # (6) 発話書き起こし（仮名漢字） 
            content = f.readline().decode('utf-8').rstrip()
            # (7) 発話書き起こし（ローマ字表記） （使わない）
            line = f.readline()
            # (8) ターミネータ （使わない）
            line = f.readline()

            yomi = get_yomi(content)
            voca = yomi2voca(yomi).split(' ')
            vad_dict[key].append([start_time, end_time, end_time - start_time, voca])
        utt_dict = {'UTT-L': [], 'UTT-R': []}
    for channel in 'L', 'R':
        vads = vad_dict['VAD-' + channel]

        start = None
        end = None
        voca = []
        for vad in vads:
            if start is None:
                start = vad[0]
                end = vad[1]
                voca += vad[3]
                continue
            if vad[0] - end < max_gap:
                end = vad[1]
                voca += vad[3]
                continue
            dur = end - start
            utt_dict['UTT-' + channel].append([start, end, dur, voca])
            start = vad[0]
            end = vad[1]
            dur = end - start
            voca = [] + vad[3]
    vad_dict.update(utt_dict)
    return vad_dict
