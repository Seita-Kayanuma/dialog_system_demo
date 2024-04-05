# coding: utf-8
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
mecab_tagger = MeCab.Tagger('--node-format=%pS%f[8] --eos-format='
                            '\\n'
                            ' --unk-format=%M')


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
        base = ['零', '一', 'ニ', '三', '四', '五', '六', '七', '八', '九']
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
    number_before = ''  # 小数点前
    number_after = ''  # 小数点後
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
        base = ['零', '一', 'ニ', '三', '四', '五', '六', '七', '八', '九']
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
    ['[〇]', 'まる'],
]
alpha2yomi_list = [[re.compile(u(x[0])), u(x[1])] for x in alpha2yomi_list]

pat_to_sub = re.compile(u('[　，．。「」]'))

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

    # スペースなど余計なものは除く
    input_zen = pat_to_sub.sub('', input_zen)

    # MeCabで読みをつける
    if six.PY2:
        input_yomi = mecab_tagger.parse(
            input_zen.encode('utf-8')).decode('utf-8')
    else:
        input_yomi = mecab_tagger.parse(input_zen)
    input_yomi = input_yomi.rstrip()


    # 変換しきれていないアルファベットは一文字ごとの読みに変える
    for pat, repl in alpha2yomi_list:
        input_yomi = pat.sub(repl, input_yomi)
        # print input_yomi

    # カタカナに直す
    # input_hira = jctconv.kata2hira(input_yomi)
    input_kata = jctconv.hira2kata(input_yomi)
    
    # return input_hira
    # return input_yomi
    
    # print (input_zen, input_kata)
    
    return input_kata
