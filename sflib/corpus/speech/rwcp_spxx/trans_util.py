# coding: utf-8
from ...speech.trans import TransInfo
from ....lang.yomi.voca import yomi2voca
from ....lang.yomi.roma import roma2kana
import re

patt_num = re.compile('[0-9]+')


def yomi2pron(s):
    s_in = s
    s = s_in[:]
    if s is None:
        return None
    try:
        phones = yomi2voca(s)
    except RuntimeError as e:
        print("INPUT:", s_in)
        print("STRIPPED:", s)
        raise e
    return phones


def read_dat_file(filename, max_gap=700):
    """
    RWCPのdatファイルを読み込んで，転記情報に変換する
    """
    # RWCPは最大2chまで
    info = [[], []]
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
                channel = 0
            elif speaker == 'A':
                channel = 1
            else:
                raise RuntimeError("Unknown Speaker %s" % speaker)
            # print(line, speaker)
            # (4) 発話開始時間（改行文字を消さなくても整数には直せるらしい）
            start_time = int(patt_num.search(f.readline().decode('utf-8'))[0])
            # (5) 発話終了時間
            end_time = int(patt_num.search(f.readline().decode('utf-8'))[0])
            # (6) 発話書き起こし（仮名漢字）
            content = f.readline().decode('utf-8').rstrip()
            # (7) 発話書き起こし（ローマ字表記）
            roma = f.readline().decode('utf-8').rstrip()
            # (8) ターミネータ （使わない）
            line = f.readline()

            content = content.replace(',', '、')
            kana = roma2kana(roma)
            pron = yomi2pron(kana)
            info[channel].append(
                TransInfo(start_time, end_time, content, pron))
    # 統合を行う
    unified_info = [[], []]
    for ch, data_list in enumerate(info):
        if len(data_list) < 2:
            unified_info[ch] = info[ch]
            continue
        t = data_list[0]
        for data in data_list[1:]:
            if data.start - t.end < max_gap:
                t = TransInfo(t.start, data.end,
                              t.trans + '、' + data.trans,
                              t.pron + ' sp ' + data.pron)
            else:
                trans = t.trans.replace('，', '、')
                trans = trans.replace('。、', '。')
                trans = trans.replace('、、', '、')
                pron = 'sp ' + t.pron + ' sp'
                pron = pron.replace('sp sp', 'sp')
                t = TransInfo(t.start - max_gap,
                              t.end + max_gap,
                              trans, pron)
                unified_info[ch].append(t)
                t = data
        trans = t.trans.replace('，', '、')
        trans = trans.replace('。、', '。')
        trans = trans.replace('、、', '、')
        pron = 'sp ' + t.pron + ' sp'
        pron = pron.replace('sp sp', 'sp')
        t = TransInfo(t.start - max_gap,
                      t.end + max_gap,
                      trans, pron)
        unified_info[ch].append(t)
    info = unified_info
    
    return info
