import os
from pykakasi import kakasi
from glob import glob

def convert_to_kana(text):
    """テキストをひらがなに変換する関数"""
    kakasi_instance = kakasi()
    kakasi_instance.setMode("J", "K")  # 漢字からカタカナへ
    kakasi_instance.setMode("H", "K")  # ひらがなからカタカナへ
    kakasi_instance.setMode("r", "Hepburn")  # ローマ字はそのまま
    converter = kakasi_instance.getConverter()
    return converter.do(text)

def get_latest_summary_file(directory):
    """指定したディレクトリ内の最新のファイルを取得する"""
    files = glob(os.path.join(directory, '*.txt'))
    if not files:
        raise FileNotFoundError("要約ファイルが見つかりませんでした。")
    return max(files, key=os.path.getctime)

def process_to_kana(input_file, output_file):
    """ファイルをひらがなに変換して保存する関数"""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(input_file, 'r', encoding='utf-8') as f:
        input_text = f.read()

    kana_text = convert_to_kana(input_text)

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(kana_text)

    print(f"カナに変換した結果をファイルに保存しました: {output_file}")
