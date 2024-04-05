# coding: utf-8
#####################################
# Juliusで音素アライメントを算出する
#####################################
import MeCab
import jctconv
import mojimoji
import re
import subprocess as sb
import six

def u(str):
    if six.PY2:
        return str.decode('utf-8')
    return str

# 漢字かな交じり文をカタカナに直すため
mecab_tagger = MeCab.Tagger('--node-format=%pS%f[8] --eos-format=''\\n'' --unk-format=%M')

#
def arabic2kanji(input):
    """アラビック数字を漢数字に直す．
    先頭の"-"は，特別に「マイナス」とする
    """
    # マイナスがついているかどうか判定
    flag_minus = False
    if input[0] == '-':
        flag_minus = True
        input.strip()
    # カンマがあるかもしれないので消す
    input = input.replace(',', '')

    def convert_unit(unit):
        """4桁を1ユニットとして
        ◯千◯百◯十◯
        という表現を得る
        """
        base = ['零', '一', 'ニ', '三', '四',
                '五', '六', '七', '八', '九']
        base = [u(s) for s in base]
        # unitが一桁でその値が0なら空を返す
        if len(unit) and unit[0] == '0':
            return ''
        # unitが'0000'なら，''を返して終了
        # （こういうケースは途中の何も無い桁のときにしか起こり得ない）
        if unit == '0000':
            return ''
        result = ''
        try:
            # 反対から見ていって埋める
            seq = reversed(unit)
            # 一の桁
            value = next(seq)
            if value == '0':
                # 0の場合は何も出さない
                pass
            else:
                # その他の場合は対応する数字をそのまま出す
                result += base[int(value)]
            # 十の桁
            value = next(seq)
            if value == '0':
                # 0の場合は何も出さない
                pass
            elif value == '1':
                # 1の場合は'十'だけ出す
                result += u('十')
            else:
                # その他の場合は'十'と対応する数字を出す
                result += u('十') + base[int(value)]
            # 百の桁
            value = next(seq)
            if value == '0':
                # 0の場合は何も出さない
                pass
            elif value == '1':
                # 1の場合は'百'だけ出す
                result += u('百')
            else:
                # その他の場合は'百'と対応する数字を出す
                result += u('百') + base[int(value)]
            # 千の桁
            value = next(seq)
            if value == '0':
                # 0の場合は何も出さない
                pass
            else:
                # その他の場合は'百'と対応する数字を出す
                result += u('千') + base[int(value)]
        except:
            pass

        # 全体を反転させて返す
        return result[::-1]

    # 小数点前と小数点後に分ける
    number_list = input.split('.')
    number_before = '' # 小数点前
    number_after  = '' # 小数点後
    if len(number_list) > 2:
        # 小数点が二つ以上あったらおかしいので例外
        raise RuntimeError('unexpected number format')
    if len(number_list) > 1:
        number_after = number_list[1]
    number_before = number_list[0]

    # 小数点前の数字について
    unit_string = ['', '万', '億', '兆', '京', '垓']
    unit_string = [u(s) for s in unit_string]
    result = ''
    for current_unit_string in unit_string:
        # 下4桁を取得
        unit_string = number_before[-4:]
        # 漢数字に変換
        unit_result = convert_unit(unit_string)
        # 変換結果が空でなければ桁を入れる
        if len(unit_result) > 0:
            unit_result += current_unit_string
        # 反転して結果に追加
        result += unit_result[::-1]
        # 下4桁を削除
        number_before = number_before[:-4]

        # 空になったら終了
        if len(number_before) == 0:
            break

    # 反転する
    result = result[::-1]

    # 空の場合は'零'にする
    if len(result) == 0:
        result = u('零')

    # 小数点以下がある場合は追加作業をする
    if len(number_after) > 0:
        result += u('点')
        base = ['零', '一', 'ニ', '三', '四',
                '五', '六', '七', '八', '九']
        base = [u(s) for s in base]
        for v in number_after:
            result += base[int(v)]

    # マイナスがついている場合は「マイナス」をつける
    if flag_minus:
        result = u('マイナス') + result

    return result

alpha2yomi_list = [
    ['[ａＡ]', 'えー'],
    ['[ｂＢ]', 'びー'],
    ['[ｃＣ]', 'しー'],
    ['[ｄＤ]', 'でぃー'],
    ['[ｅＥ]', 'いー'],
    ['[ｆＦ]', 'えふ'],
    ['[ｇＧ]', 'じー'],
    ['[ｈＨ]', 'えっち'],
    ['[ｉＩ]', 'あい'],
    ['[ｊＪ]', 'じぇー'],
    ['[ｋＫ]', 'けー'],
    ['[ｌＬ]', 'える'],
    ['[ｍＭ]', 'えむ'],
    ['[ｎＮ]', 'えぬ'],
    ['[ｏＯ]', 'おー'],
    ['[ｐＰ]', 'ぴー'],
    ['[ｑＱ]', 'きゅー'],
    ['[ｒＲ]', 'あーる'],
    ['[ｓＳ]', 'えす'],
    ['[ｔＴ]', 'てぃー'],
    ['[ｕＵ]', 'ゆー'],
    ['[ｖＶ]', 'ぶい'],
    ['[ｗＷ]', 'だぶりゅー'],
    ['[ｘＸ]', 'えっくす'],
    ['[ｙＹ]', 'わい'],
    ['[ｚＺ]', 'ぜっと'],
    ]
alpha2yomi_list = [[re.compile(u(x[0])), u(x[1])] for x in alpha2yomi_list]

pat_to_sub = re.compile(u('[　，．。]'))

pat_number = re.compile(r'(\-?[0-9,]+(\.?[0-9]+))')

def get_yomi(input):
    """
    漢字仮名交じり文を，ひらがな文に変換する

    input: unicode文字列
    """
    # 数値は半角キャラクタで与えられるようなのでさっさと置換する
    def convert(mo):
        return arabic2kanji(mo.group(0))
    input = pat_number.sub(convert, input)

    # 半角文字列を全角文字列に変換する
    input_zen = mojimoji.han_to_zen(input)

    # MeCabで読みをつける
    if six.PY2:
        input_yomi = mecab_tagger.parse(input_zen.encode('utf-8')).decode('utf-8')
    else:
        input_yomi = mecab_tagger.parse(input_zen)
    input_yomi = input_yomi.rstrip()

    # スペースなど余計なものは除く
    input_yomi = pat_to_sub.sub('', input_yomi)

    # 変換しきれていないアルファベットは一文字ごとの読みに変える
    for pat, repl in alpha2yomi_list:
        input_yomi = pat.sub(repl, input_yomi)
        # print input_yomi

    # 平仮名に直す
    input_hira = jctconv.kata2hira(input_yomi)

    return input_hira

yomi2voca_list = [
# 3文字以上からなる変換規則
    ["う゛ぁ", " b a"],
    ["う゛ぃ", " b i"],
    ["う゛ぇ", " b e"],
    ["う゛ぉ", " b o"],
    ["う゛ゅ", " by u"],

# 2文字からなる変換規則
    ["ぅ゛", " b u"],

    ["あぁ", " a a"],
    ["いぃ", " i i"],
    ["いぇ", " i e"],
    ["いゃ", " y a"],
    ["うぅ", " u:"],
    ["えぇ", " e e"],
    ["おぉ", " o:"],
    ["かぁ", " k a:"],
    ["きぃ", " k i:"],
    ["くぅ", " k u:"],
    ["くゃ", " ky a"],
    ["くゅ", " ky u"],
    ["くょ", " ky o"],
    ["けぇ", " k e:"],
    ["こぉ", " k o:"],
    ["がぁ", " g a:"],
    ["ぎぃ", " g i:"],
    ["ぐぅ", " g u:"],
    ["ぐゃ", " gy a"],
    ["ぐゅ", " gy u"],
    ["ぐょ", " gy o"],
    ["げぇ", " g e:"],
    ["ごぉ", " g o:"],
    ["さぁ", " s a:"],
    ["しぃ", " sh i:"],
    ["すぅ", " s u:"],
    ["すゃ", " sh a"],
    ["すゅ", " sh u"],
    ["すょ", " sh o"],
    ["せぇ", " s e:"],
    ["そぉ", " s o:"],
    ["ざぁ", " z a:"],
    ["じぃ", " j i:"],
    ["ずぅ", " z u:"],
    ["ずゃ", " zy a"],
    ["ずゅ", " zy u"],
    ["ずょ", " zy o"],
    ["ぜぇ", " z e:"],
    ["ぞぉ", " z o:"],
    ["たぁ", " t a:"],
    ["ちぃ", " ch i:"],
    ["つぁ", " ts a"],
    ["つぃ", " ts i"],
    ["つぅ", " ts u:"],
    ["つゃ", " ch a"],
    ["つゅ", " ch u"],
    ["つょ", " ch o"],
    ["つぇ", " ts e"],
    ["つぉ", " ts o"],
    ["てぇ", " t e:"],
    ["とぉ", " t o:"],
    ["だぁ", " d a:"],
    ["ぢぃ", " j i:"],
    ["づぅ", " d u:"],
    ["づゃ", " zy a"],
    ["づゅ", " zy u"],
    ["づょ", " zy o"],
    ["でぇ", " d e:"],
    ["どぉ", " d o:"],
    ["なぁ", " n a:"],
    ["にぃ", " n i:"],
    ["ぬぅ", " n u:"],
    ["ぬゃ", " ny a"],
    ["ぬゅ", " ny u"],
    ["ぬょ", " ny o"],
    ["ねぇ", " n e:"],
    ["のぉ", " n o:"],
    ["はぁ", " h a:"],
    ["ひぃ", " h i:"],
    ["ふぅ", " f u:"],
    ["ふゃ", " hy a"],
    ["ふゅ", " hy u"],
    ["ふょ", " hy o"],
    ["へぇ", " h e:"],
    ["ほぉ", " h o:"],
    ["ばぁ", " b a:"],
    ["びぃ", " b i:"],
    ["ぶぅ", " b u:"],
    ["ふゃ", " hy a"],
    ["ぶゅ", " by u"],
    ["ふょ", " hy o"],
    ["べぇ", " b e:"],
    ["ぼぉ", " b o:"],
    ["ぱぁ", " p a:"],
    ["ぴぃ", " p i:"],
    ["ぷぅ", " p u:"],
    ["ぷゃ", " py a"],
    ["ぷゅ", " py u"],
    ["ぷょ", " py o"],
    ["ぺぇ", " p e:"],
    ["ぽぉ", " p o:"],
    ["まぁ", " m a:"],
    ["みぃ", " m i:"],
    ["むぅ", " m u:"],
    ["むゃ", " my a"],
    ["むゅ", " my u"],
    ["むょ", " my o"],
    ["めぇ", " m e:"],
    ["もぉ", " m o:"],
    ["やぁ", " y a:"],
    ["ゆぅ", " y u:"],
    ["ゆゃ", " y a:"],
    ["ゆゅ", " y u:"],
    ["ゆょ", " y o:"],
    ["よぉ", " y o:"],
    ["らぁ", " r a:"],
    ["りぃ", " r i:"],
    ["るぅ", " r u:"],
    ["るゃ", " ry a"],
    ["るゅ", " ry u"],
    ["るょ", " ry o"],
    ["れぇ", " r e:"],
    ["ろぉ", " r o:"],
    ["わぁ", " w a:"],
    ["をぉ", " o:"],

    ["う゛", " b u"],
    ["でぃ", " d i"],
    ["でぇ", " d e:"],
    ["でゃ", " dy a"],
    ["でゅ", " dy u"],
    ["でょ", " dy o"],
    ["てぃ", " t i"],
    ["てぇ", " t e:"],
    ["てゃ", " ty a"],
    ["てゅ", " ty u"],
    ["てょ", " ty o"],
    ["すぃ", " s i"],
    ["ずぁ", " z u a"],
    ["ずぃ", " z i"],
    ["ずぅ", " z u"],
    ["ずゃ", " zy a"],
    ["ずゅ", " zy u"],
    ["ずょ", " zy o"],
    ["ずぇ", " z e"],
    ["ずぉ", " z o"],
    ["きゃ", " ky a"],
    ["きゅ", " ky u"],
    ["きょ", " ky o"],
    ["しゃ", " sh a"],
    ["しゅ", " sh u"],
    ["しぇ", " sh e"],
    ["しょ", " sh o"],
    ["ちゃ", " ch a"],
    ["ちゅ", " ch u"],
    ["ちぇ", " ch e"],
    ["ちょ", " ch o"],
    ["とぅ", " t u"],
    ["とゃ", " ty a"],
    ["とゅ", " ty u"],
    ["とょ", " ty o"],
    ["どぁ", " d o a"],
    ["どぅ", " d u"],
    ["どゃ", " dy a"],
    ["どゅ", " dy u"],
    ["どょ", " dy o"],
    ["どぉ", " d o:"],
    ["にゃ", " ny a"],
    ["にゅ", " ny u"],
    ["にょ", " ny o"],
    ["ひゃ", " hy a"],
    ["ひゅ", " hy u"],
    ["ひょ", " hy o"],
    ["みゃ", " my a"],
    ["みゅ", " my u"],
    ["みょ", " my o"],
    ["りゃ", " ry a"],
    ["りゅ", " ry u"],
    ["りょ", " ry o"],
    ["ぎゃ", " gy a"],
    ["ぎゅ", " gy u"],
    ["ぎょ", " gy o"],
    ["ぢぇ", " j e"],
    ["ぢゃ", " j a"],
    ["ぢゅ", " j u"],
    ["ぢょ", " j o"],
    ["じぇ", " j e"],
    ["じゃ", " j a"],
    ["じゅ", " j u"],
    ["じょ", " j o"],
    ["びゃ", " by a"],
    ["びゅ", " by u"],
    ["びょ", " by o"],
    ["ぴゃ", " py a"],
    ["ぴゅ", " py u"],
    ["ぴょ", " py o"],
    ["うぁ", " u a"],
    ["うぃ", " w i"],
    ["うぇ", " w e"],
    ["うぉ", " w o"],
    ["ふぁ", " f a"],
    ["ふぃ", " f i"],
    ["ふぅ", " f u"],
    ["ふゃ", " hy a"],
    ["ふゅ", " hy u"],
    ["ふょ", " hy o"],
    ["ふぇ", " f e"],
    ["ふぉ", " f o"],

# 1音からなる変換規則
    ["あ", " a"],
    ["い", " i"],
    ["う", " u"],
    ["え", " e"],
    ["お", " o"],
    ["か", " k a"],
    ["き", " k i"],
    ["く", " k u"],
    ["け", " k e"],
    ["こ", " k o"],
    ["さ", " s a"],
    ["し", " sh i"],
    ["す", " s u"],
    ["せ", " s e"],
    ["そ", " s o"],
    ["た", " t a"],
    ["ち", " ch i"],
    ["つ", " ts u"],
    ["て", " t e"],
    ["と", " t o"],
    ["な", " n a"],
    ["に", " n i"],
    ["ぬ", " n u"],
    ["ね", " n e"],
    ["の", " n o"],
    ["は", " h a"],
    ["ひ", " h i"],
    ["ふ", " f u"],
    ["へ", " h e"],
    ["ほ", " h o"],
    ["ま", " m a"],
    ["み", " m i"],
    ["む", " m u"],
    ["め", " m e"],
    ["も", " m o"],
    ["ら", " r a"],
    ["り", " r i"],
    ["る", " r u"],
    ["れ", " r e"],
    ["ろ", " r o"],
    ["が", " g a"],
    ["ぎ", " g i"],
    ["ぐ", " g u"],
    ["げ", " g e"],
    ["ご", " g o"],
    ["ざ", " z a"],
    ["じ", " j i"],
    ["ず", " z u"],
    ["ぜ", " z e"],
    ["ぞ", " z o"],
    ["だ", " d a"],
    ["ぢ", " j i"],
    ["づ", " z u"],
    ["で", " d e"],
    ["ど", " d o"],
    ["ば", " b a"],
    ["び", " b i"],
    ["ぶ", " b u"],
    ["べ", " b e"],
    ["ぼ", " b o"],
    ["ぱ", " p a"],
    ["ぴ", " p i"],
    ["ぷ", " p u"],
    ["ぺ", " p e"],
    ["ぽ", " p o"],
    ["や", " y a"],
    ["ゆ", " y u"],
    ["よ", " y o"],
    ["わ", " w a"],
    ["ゐ", " i"],
    ["ゑ", " e"],
    ["ん", " N"],
    ["っ", " q"],
    ["ー", ":"],

# ここまでに処理されてない ぁぃぅぇぉ はそのまま大文字扱い
    ["ぁ", " a"],
    ["ぃ", " i"],
    ["ぅ", " u"],
    ["ぇ", " e"],
    ["ぉ", " o"],
    ["ゎ", " w a"],
    ["ぉ", " o"],

#その他特別なルール
    ["を", " o"],
    ["、", " sp"]
]
yomi2voca_list = [[u(x[0]), x[1]] for x in yomi2voca_list]

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

_grammar_template = """GRAMMAR
0 2 1 0 1
1 1 2 0 0
2 0 3 0 0
3 -1 -1 1 0
DFAEND
0 [w_0] silB
1 [w_1] %s
2 [w_2] silE
DICEND
"""

def get_grammar_string(input):
    """
    Juliusに入力するための文法表記を取得する
    """
    yomi = get_yomi(input)
    voca = yomi2voca(yomi)

    return _grammar_template % voca

class PhonemeAligner:
    def __init__(self,
                 julius_bin_path, plugin_dir_path,
                 phone_model_path,
                 dummy_dfa_path, dummy_dict_path):
        import os
        DEVNULL = open(os.devnull, 'wb')

        self.process = sb.Popen(
            [julius_bin_path,
             '-plugindir', plugin_dir_path,
             '-h',         phone_model_path,
             '-dfa',       dummy_dfa_path,
             '-v',         dummy_dict_path,
             '-palign',
             '-input',     'alignment'],
            stdin=sb.PIPE, stdout=sb.PIPE,
            stderr=DEVNULL,
            bufsize=0, universal_newlines=False)
        self._pat = re.compile(r'\[ *(\d+) *(\d+)\] *(\-?\d+\.\d+) *([\w:]+)')
        self.__read_until_read_waveform_input()

    def __read_until_read_waveform_input(self):
        while True:
            line = self.process.stdout.readline()
            # print "stdout:", line,
            if re.search(r'read waveform input', str(line)):
                break

    def get_alignment(self, sentence, data):
        # 文法を入力
        grammar_string = get_grammar_string(sentence)
        self.process.stdin.write(six.b(grammar_string))

        # 音声データのヘッダを入力
        self.process.stdin.write(six.b("WAV\n"))
        self.process.stdin.write(six.b("%d\n" % (int(len(data) / 2),)))
        self.process.stdin.write(data)

        # begin forced alignment まで読み飛ばし
        while True:
            line = self.process.stdout.readline().decode('utf-8')
            # import ipdb; ipdb.set_trace()
            if re.search(r'begin forced alignment', line):
                break

        # end forced alignment が出てくるまで，matchをして
        # 値を取り出してくる
        endtime_list = []
        phone_list = []
        while True:
            line = self.process.stdout.readline().decode('utf-8')
            if re.search(r'end forced alignment', line):
                break

            m = self._pat.match(line)
            if m is not None:
                g = m.groups()
                endtime_list.append(int(g[1]) * 10)
                phone_list.append(g[3])

        self.__read_until_read_waveform_input()

        return endtime_list, phone_list

if __name__ == '__main__':
    mecab_tagger = MeCab.Tagger('--node-format=%pS%f[8] --eos-format=''\\n'' --unk-format=%M')
    # mecab_tagger = MeCab.Tagger('--node-format=%pS%m+%f[0]\\s --eos-format=''\\n'' --unk-format=%M')
    input = u("私の名前は藤江です")
    
    def convert(mo):
        return arabic2kanji(mo.group(0))
    input = pat_number.sub(convert, input)

    # 半角文字列を全角文字列に変換する
    input_zen = mojimoji.han_to_zen(input)

    if six.PY2:
        input_yomi = mecab_tagger.parse(input_zen.encode('utf-8')).decode('utf-8')
    else:
        input_yomi = mecab_tagger.parse(input_zen)
    
    print(input_yomi)
    
    # print arabic2kanji('123')
    # print arabic2kanji('123123')
    # print arabic2kanji('123123123')
    # print arabic2kanji('123123123123')
    # print arabic2kanji('0')
    # print arabic2kanji('1')
    # print arabic2kanji('10')
    # print arabic2kanji('100')
    # print arabic2kanji('1000')
    # print arabic2kanji('10000')
    # print arabic2kanji('100000')
    # print arabic2kanji('1000000')
    # print arabic2kanji('10000000')
    # print arabic2kanji('100000000')
    # print arabic2kanji('123.123')
    # print arabic2kanji('-123.123')

    # # print mecab_tagger.parse('ｐｅｐｅr').rstrip()
    yomi = get_yomi(u('JALの172.34cmのpepper は対話型ロボットです。'))
    print(yomi)
    # voca = yomi2voca(yomi)
    # print voca
