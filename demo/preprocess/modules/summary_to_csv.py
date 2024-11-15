import os
import re
import csv
from datetime import datetime

def get_latest_summary_file(directory):
    """指定したディレクトリ内の最新のファイルを取得する"""
    files = [f for f in os.listdir(directory) if f.endswith('.txt')]
    if not files:
        raise FileNotFoundError("要約ファイルが見つかりませんでした。")
    return max(files, key=lambda x: os.path.getmtime(os.path.join(directory, x)))

def parse_dialogue(text):
    """テキストを解析して、各文を抽出する。"""
    dialogues = []
    sentences = re.split(r'(?<=[。！？])\s*', text.strip())
    for sentence in sentences:
        if sentence:
            dialogues.append((sentence.strip(), '1'))
    return dialogues

def process_to_csv(input_file, output_csv_file=None):
    """ファイルをCSVに変換して保存する関数"""
    if not output_csv_file:
        output_csv_dir = 'data/csv'
        os.makedirs(output_csv_dir, exist_ok=True)
        base_name = os.path.basename(input_file).replace('.txt', '.csv')
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_csv_file = os.path.join(output_csv_dir, f'converted_{timestamp}_{base_name}')
    
    with open(input_file, 'r', encoding='utf-8') as f:
        input_text = f.read()

    dialogues = parse_dialogue(input_text)

    with open(output_csv_file, 'w', encoding='utf-8', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['text', 'system'])
        for text, system in dialogues:
            writer.writerow([text, system])

    print(f"CSVファイルを保存しました: {output_csv_file}")
