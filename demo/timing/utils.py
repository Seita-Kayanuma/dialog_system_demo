import json
import MeCab
import difflib
from dotmap import DotMap


def load_config(f_path):
    with open(f_path, 'r') as f:
        config_json = json.load(f)
        return DotMap(config_json)



def get_similar_word(keywords, text):
    best_match = None
    best_idx = None
    highest_ratio = 0.0
    for idx, keyword in enumerate(keywords):
        for i in range(len(text) - len(keyword) + 1):
            substring = text[i:i+len(keyword)]
            ratio = difflib.SequenceMatcher(None, substring, keyword).ratio()
            if ratio > highest_ratio:
                highest_ratio = ratio
                best_match = keyword
                best_idx = idx
    return best_idx


_kana_list = None
# <sp> | 抜き
_kanas = """ア イ ウ エ オ カ キ ク ケ コ ガ ギ グ ゲ ゴ サ シ ス セ ソ
ザ ジ ズ ゼ ゾ タ チ ツ テ ト ダ デ ド ナ ニ ヌ ネ ノ ハ ヒ フ ヘ ホ
バ ビ ブ ベ ボ パ ピ プ ペ ポ マ ミ ム メ モ ラ リ ル レ ロ ヤ ユ ヨ
ワ ヲ ン ウィ ウェ ウォ キャ キュ キョ ギャ ギュ ギョ シャ シュ ショ
ジャ ジュ ジョ チャ チュ チョ ディ ドゥ デュ ニャ ニュ ニョ ヒャ ヒュ ヒョ
ビャ ビュ ビョ ピャ ピュ ピョ ミャ ミュ ミョ リャ リュ リョ イェ クヮ
グヮ シェ ジェ ティ トゥ チェ ツァ ツィ ツェ ツォ ヒェ ファ フィ フェ フォ フュ
テュ ブィ ニェ ミェ スィ ズィ ヴァ ヴィ ヴ ヴェ ヴォ ー ッ"""
_kana_list = [x.replace(' ', '') for x in _kanas.replace('\n', ' ').split(' ')]
_kana_list = sorted(_kana_list, key=len, reverse=True)


# 音声認識結果 -> モーラ
def split_pron_to_mora(pron: str):
    """発音形をモーラに分割する
    Args:
        pron: 発音形
    
    Returns:
        モーラのリスト
    """
    moras = []
    while len(pron) > 0:
        flag = False
        for kana in _kana_list:
            if pron.startswith(kana):
                moras.append(kana)
                pron = pron[len(kana):]
                flag = True
                break
        if not flag:
            # 警告を表示する
            # print('Warning: Unknown kana: {}'.format(pron))
            # pronの先頭文字を削除する
            pron = pron[1:]
    return ' '.join(moras), len(moras)


noises = ['|', '<sp>']
def denoise_kana(kanas: list):
    denoise_kanas = []
    for kana in kanas:
        kana = kana.split(' ')
        kana = [k.replace('+F', '').replace('+D', '') for k in kana if k not in noises]
        denoise_kanas.append(' '.join(kana))
    return denoise_kanas