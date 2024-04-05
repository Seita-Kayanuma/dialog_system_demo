# coding: utf-8
import re
from sflib.corpus.speech.trans import TransInfo
from sflib.lang.yomi.voca import yomi2voca

__patt_vad = re.compile(r'\d+ (\d+\.\d+)\-(\d+\.\d+) (.)\:(<.+>)?$')
__patt_content = re.compile(r'^(.+)& (.+)$')


def format_string(s):
    s_in = s
    if '×' in s:
        return None
    # (?)などを消す
    s = re.sub(r'\(.\)', '', s)
    # # (X (F アイウエオ;カキクケコ)) → アイウエオ にする
    # s = re.sub(r'(?:\(.+? )+(.+?)(?:;.+?)?(?:\)+)', r'\1', s)
    while True:
        so = s
        s = re.sub(r'\(.+? ([^(]+?)(?:;[^(]+?)*?\)', r'\1', so)
        if so == s:
            break
    # ; だけ単独で存在するケースがあるので消す
    s = re.sub(r'(;|,)', r'', s)
    # 発話にまたがってしまった場合のケース
    s = re.sub(r'(?:\(.+? )+(.+?)$', r'\1', s)
    s = re.sub(r'^(.+?)\)+', r'\1', s)
    s = re.sub(r'\)+', r'', s)
    # <息>などを消す
    s = re.sub(r'<[^>]+>', '', s)

    s = re.sub(r' +', '', s)

    if len(s) == 0:
        return None

    return s


def yomi2pron(s):
    s_in = s
    s = format_string(s)
    if s is None:
        return None
    try:
        phones = yomi2voca(s)
    except RuntimeError as e:
        print("INPUT:", s_in)
        print("STRIPPED:", s)
        raise e
    return phones


def read_trn_file(filename, max_gap=300):
    """
    CSJのTRNファイルを読み込んで，転記情報に変換する
    """
    # CSJは最大2chまで
    info = [[], []]

    ch = 0
    start = None
    end = None
    trans = None
    pron = None

    with open(filename, 'rb') as f:
        while True:
            line = f.readline().decode('Shift-JIS')
            if not line:
                break
            line = line.rstrip()
            # print(line)

            # % で始まる部分はコメントなのでとばす（頭にしかない）
            if line[0] == '%':
                continue

            # VAD情報が書いてある行か判定
            m = __patt_vad.match(line)
            if m is not None:
                # ここまでの結果があったら確定
                if start is not None:
                    trans = format_string(trans)
                    pron = yomi2pron(pron)
                    # 音素列に正しく変換できたら追加する
                    if trans is not None and pron is not None:
                        info[ch].append(TransInfo(start, end, trans, pron))
                    start = None
                # 雑音だったら無視
                if m.lastindex > 3:
                    continue
                # チャネルの判定
                if m.group(3) == 'L':
                    ch = 0
                elif m.group(3) == 'R':
                    ch = 1
                else:
                    continue
                start = int(float(m.group(1)) * 1000)
                end = int(float(m.group(2)) * 1000)
                trans = ''
                pron = ''
                continue

            # 発言内容をためていく
            if start is None:
                continue

            m = __patt_content.match(line)
            # import ipdb; ipdb.set_trace()
            if m is None or m.lastindex != 2:
                continue
            trans += m.group(1)
            pron += m.group(2).rstrip()
    if start is not None:
        trans = format_string(trans)
        pron = yomi2pron(pron)
        # 音素列に正しく変換できたら追加する
        if trans is not None and pron is not None:
            info[ch].append(TransInfo(start, end, trans, pron))
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
