# coding: utf-8
import sys
import re


def tran2vad(filename, max_gap=700):
    """
    CSJのTRNファイルからVAD情報に直す．

    TRNファイルは以下のような形になっている．
    
    0303 00658.711-00658.947 L:(D2 は)
    0304 00659.212-00662.381 L:が(F えー)高水準で相関が認められました
    0305 00665.796-00668.979 L:で次に(F えー)結果をまとめたいと思います
    0306 00665.817-00665.957 L:<雑音>
    0307 00671.500-00671.884 L:よいしょ
    0308 00676.116-00676.447 L:よいしょ
    0309 00677.822-00678.172 L:<咳>
    0310 00679.268-00682.928 L:(F え)まず長調の原型とその短調の変形を
    """
    vad_dict = {'VAD-L': [], 'VAD-R': []}
    with open(filename, 'rb') as f:
        line = f.readline().decode('Shift-JIS')
        while True:
            line = f.readline().decode('Shift-JIS')
            if not line:
                break
            _, t, c = line.rstrip().split(' ', 2)
            # print t, c
            s, e = [int(float(x) * 1000.0) for x in t.split('-')]
            d = e - s
            spk, uc = c.split(':', 1)
            if '<' in uc:
                continue
            # print spk, s, e, d, uc
            key = 'VAD-' + spk
            vad_dict[key].append([s, e, d])

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


def str2phones(s):
    s_in = s
    if '×'in s:
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
    # カタカナから音素列に直す
    if len(s) == 0:
        return None
    try:
        phones = yomi2voca(s)
    except RuntimeError as e:
        print ("INPUT:", s_in)
        print ("STRIPPED:", s)
        raise e
    return phones.split(' ')


def tran2vadphones(filename, tag='L'):
    info = []
    current = None
    patt_vad = re.compile(r'\d+ (\d+\.\d+)\-(\d+\.\d+) (.)\:(<.+>)?$')
    patt_content = re.compile(r'^.+& (.+)$')

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
            m = patt_vad.match(line)
            if m is not None:
                # ここまでの結果があったら確定
                if current is not None:
                    phones = str2phones(current[3])
                    # 音素列に正しく変換できたら追加する
                    if phones is not None:
                        current[3] = phones
                        info.append(current)
                    current = None
                # 雑音だったら無視
                if m.lastindex > 3:
                    continue
                # 対象のタグじゃなければ無視
                if m.group(3) != tag:
                    continue
                start = int(float(m.group(1)) * 1000)
                end = int(float(m.group(2)) * 1000)
                duration = end - start
                current = [start, end, duration, '']
                continue

            # 発言内容をためていく
            if current is None:
                continue

            m = patt_content.match(line)
            # import ipdb; ipdb.set_trace()
            if m is None or m.lastindex != 1:
                continue
            s = m.group(1).rstrip()
            current[3] += s
    if current is not None:
        current[3] = str2phones(current[3])
        if current[3] is not None:
            info.append(current)

    return info


#
yomi2voca_list = [
    # 3文字以上からなる変換規則
    ["ウ゛ァ", " b a"],
    ["ウ゛ィ", " b i"],
    ["ウ゛ェ", " b e"],
    ["ウ゛ォ", " b o"],
    ["ウ゛ュ", " by u"],

    # 2文字からなる変換規則
    ["ゥ゛", " b u"],
    ["アァ", " a a"],
    ["イィ", " i i"],
    ["イェ", " i e"],
    ["イャ", " y a"],
    ["ウゥ", " u:"],
    ["エェ", " e e"],
    ["オォ", " o:"],
    ["カァ", " k a:"],
    ["キィ", " k i:"],
    ["クゥ", " k u:"],
    ["クャ", " ky a"],
    ["クュ", " ky u"],
    ["クョ", " ky o"],
    ["ケェ", " k e:"],
    ["コォ", " k o:"],
    ["ガァ", " g a:"],
    ["ギィ", " g i:"],
    ["グゥ", " g u:"],
    ["グャ", " gy a"],
    ["グュ", " gy u"],
    ["グョ", " gy o"],
    ["ゲェ", " g e:"],
    ["ゴォ", " g o:"],
    ["サァ", " s a:"],
    ["シィ", " sh i:"],
    ["スゥ", " s u:"],
    ["スャ", " sh a"],
    ["スュ", " sh u"],
    ["スョ", " sh o"],
    ["セェ", " s e:"],
    ["ソォ", " s o:"],
    ["ザァ", " z a:"],
    ["ジィ", " j i:"],
    ["ズゥ", " z u:"],
    ["ズャ", " zy a"],
    ["ズュ", " zy u"],
    ["ズョ", " zy o"],
    ["ゼェ", " z e:"],
    ["ゾォ", " z o:"],
    ["タァ", " t a:"],
    ["チィ", " ch i:"],
    ["ツァ", " ts a"],
    ["ツィ", " ts i"],
    ["ツゥ", " ts u:"],
    ["ツャ", " ch a"],
    ["ツュ", " ch u"],
    ["ツョ", " ch o"],
    ["ツェ", " ts e"],
    ["ツォ", " ts o"],
    ["テェ", " t e:"],
    ["トォ", " t o:"],
    ["ダァ", " d a:"],
    ["ヂィ", " j i:"],
    ["ヅゥ", " d u:"],
    ["ヅャ", " zy a"],
    ["ヅュ", " zy u"],
    ["ヅョ", " zy o"],
    ["デェ", " d e:"],
    ["ドォ", " d o:"],
    ["ナァ", " n a:"],
    ["ニィ", " n i:"],
    ["ヌゥ", " n u:"],
    ["ヌャ", " ny a"],
    ["ヌュ", " ny u"],
    ["ヌョ", " ny o"],
    ["ネェ", " n e:"],
    ["ノォ", " n o:"],
    ["ハァ", " h a:"],
    ["ヒィ", " h i:"],
    ["フゥ", " f u:"],
    ["フャ", " hy a"],
    ["フュ", " hy u"],
    ["フョ", " hy o"],
    ["ヘェ", " h e:"],
    ["ホォ", " h o:"],
    ["バァ", " b a:"],
    ["ビィ", " b i:"],
    ["ブゥ", " b u:"],
    ["フャ", " hy a"],
    ["ブュ", " by u"],
    ["フョ", " hy o"],
    ["ベェ", " b e:"],
    ["ボォ", " b o:"],
    ["パァ", " p a:"],
    ["ピィ", " p i:"],
    ["プゥ", " p u:"],
    ["プャ", " py a"],
    ["プュ", " py u"],
    ["プョ", " py o"],
    ["ペェ", " p e:"],
    ["ポォ", " p o:"],
    ["マァ", " m a:"],
    ["ミィ", " m i:"],
    ["ムゥ", " m u:"],
    ["ムャ", " my a"],
    ["ムュ", " my u"],
    ["ムョ", " my o"],
    ["メェ", " m e:"],
    ["モォ", " m o:"],
    ["ヤァ", " y a:"],
    ["ユゥ", " y u:"],
    ["ユャ", " y a:"],
    ["ユュ", " y u:"],
    ["ユョ", " y o:"],
    ["ヨォ", " y o:"],
    ["ラァ", " r a:"],
    ["リィ", " r i:"],
    ["ルゥ", " r u:"],
    ["ルャ", " ry a"],
    ["ルュ", " ry u"],
    ["ルョ", " ry o"],
    ["レェ", " r e:"],
    ["ロォ", " r o:"],
    ["ワァ", " w a:"],
    ["ヲォ", " o:"],
    ["ウ゛", " b u"],
    ["ディ", " d i"],
    ["デェ", " d e:"],
    ["デャ", " dy a"],
    ["デュ", " dy u"],
    ["デョ", " dy o"],
    ["ティ", " t i"],
    ["テェ", " t e:"],
    ["テャ", " ty a"],
    ["テュ", " ty u"],
    ["テョ", " ty o"],
    ["スィ", " s i"],
    ["ズァ", " z u a"],
    ["ズィ", " z i"],
    ["ズゥ", " z u"],
    ["ズャ", " zy a"],
    ["ズュ", " zy u"],
    ["ズョ", " zy o"],
    ["ズェ", " z e"],
    ["ズォ", " z o"],
    ["キャ", " ky a"],
    ["キュ", " ky u"],
    ["キョ", " ky o"],
    ["シャ", " sh a"],
    ["シュ", " sh u"],
    ["シェ", " sh e"],
    ["ショ", " sh o"],
    ["チャ", " ch a"],
    ["チュ", " ch u"],
    ["チェ", " ch e"],
    ["チョ", " ch o"],
    ["トゥ", " t u"],
    ["トャ", " ty a"],
    ["トュ", " ty u"],
    ["トョ", " ty o"],
    ["ドァ", " d o a"],
    ["ドゥ", " d u"],
    ["ドャ", " dy a"],
    ["ドュ", " dy u"],
    ["ドョ", " dy o"],
    ["ドォ", " d o:"],
    ["ニャ", " ny a"],
    ["ニュ", " ny u"],
    ["ニョ", " ny o"],
    ["ヒャ", " hy a"],
    ["ヒュ", " hy u"],
    ["ヒョ", " hy o"],
    ["ミャ", " my a"],
    ["ミュ", " my u"],
    ["ミョ", " my o"],
    ["リャ", " ry a"],
    ["リュ", " ry u"],
    ["リョ", " ry o"],
    ["ギャ", " gy a"],
    ["ギュ", " gy u"],
    ["ギョ", " gy o"],
    ["ヂェ", " j e"],
    ["ヂャ", " j a"],
    ["ヂュ", " j u"],
    ["ヂョ", " j o"],
    ["ジェ", " j e"],
    ["ジャ", " j a"],
    ["ジュ", " j u"],
    ["ジョ", " j o"],
    ["ビャ", " by a"],
    ["ビュ", " by u"],
    ["ビョ", " by o"],
    ["ピャ", " py a"],
    ["ピュ", " py u"],
    ["ピョ", " py o"],
    ["ウァ", " u a"],
    ["ウィ", " w i"],
    ["ウェ", " w e"],
    ["ウォ", " w o"],
    ["ファ", " f a"],
    ["フィ", " f i"],
    ["フゥ", " f u"],
    ["フャ", " hy a"],
    ["フュ", " hy u"],
    ["フョ", " hy o"],
    ["フェ", " f e"],
    ["フォ", " f o"],
    ["ヴァ", " b a"],
    ["ヴィ", " b i"],
    ["ヴェ", " b e"],
    ["ヴォ", " b o"],
    ["ヴュ", " by u"],

    # 1音からなる変換規則
    ["ア", " a"],
    ["イ", " i"],
    ["ウ", " u"],
    ["エ", " e"],
    ["オ", " o"],
    ["カ", " k a"],
    ["キ", " k i"],
    ["ク", " k u"],
    ["ケ", " k e"],
    ["コ", " k o"],
    ["サ", " s a"],
    ["シ", " sh i"],
    ["ス", " s u"],
    ["セ", " s e"],
    ["ソ", " s o"],
    ["タ", " t a"],
    ["チ", " ch i"],
    ["ツ", " ts u"],
    ["テ", " t e"],
    ["ト", " t o"],
    ["ナ", " n a"],
    ["ニ", " n i"],
    ["ヌ", " n u"],
    ["ネ", " n e"],
    ["ノ", " n o"],
    ["ハ", " h a"],
    ["ヒ", " h i"],
    ["フ", " f u"],
    ["ヘ", " h e"],
    ["ホ", " h o"],
    ["マ", " m a"],
    ["ミ", " m i"],
    ["ム", " m u"],
    ["メ", " m e"],
    ["モ", " m o"],
    ["ラ", " r a"],
    ["リ", " r i"],
    ["ル", " r u"],
    ["レ", " r e"],
    ["ロ", " r o"],
    ["ガ", " g a"],
    ["ギ", " g i"],
    ["グ", " g u"],
    ["ゲ", " g e"],
    ["ゴ", " g o"],
    ["ザ", " z a"],
    ["ジ", " j i"],
    ["ズ", " z u"],
    ["ゼ", " z e"],
    ["ゾ", " z o"],
    ["ダ", " d a"],
    ["ヂ", " j i"],
    ["ヅ", " z u"],
    ["デ", " d e"],
    ["ド", " d o"],
    ["バ", " b a"],
    ["ビ", " b i"],
    ["ブ", " b u"],
    ["ベ", " b e"],
    ["ボ", " b o"],
    ["パ", " p a"],
    ["ピ", " p i"],
    ["プ", " p u"],
    ["ペ", " p e"],
    ["ポ", " p o"],
    ["ヤ", " y a"],
    ["ユ", " y u"],
    ["ヨ", " y o"],
    ["ワ", " w a"],
    ["ヴ", " b u"],
    ["ヰ", " i"],
    ["ヱ", " e"],
    ["ン", " N"],
    ["ッ", " q"],
    ["ー", ":"],

    # ここまでに処理されない ァィゥェォ はそのまま大文字扱い
    ["ァ", " a"],
    ["ィ", " i"],
    ["ゥ", " u"],
    ["ェ", " e"],
    ["ォ", " o"],
    ["ヮ", " w a"],
    ["ォ", " o"],

    # その他特別なルール
    ["ヲ", " o"],
    ["、", " sp"]
]

# yomi2voca_list = [[u(x[0]), x[1]] for x in yomi2voca_list]


def yomi2voca(input):
    """
    ひらがな文を音素列に変換する
    """
    # output = u''
    # for c in input:
    #    output += yomi2voca_dict[c]
    output = input
    for arg in yomi2voca_list:
        # import ipdb; ipdb.set_trace()
        output = output.replace(*arg)

    # 先頭のスペースを削除
    output = output.strip()

    # 変換できたかチェック
    # import ipdb; ipdb.set_trace()
    if not re.match(r"^[ a-zN:]+$", output):
        raise RuntimeError('cannot convert')

    return output
