import os
import csv
import datetime

def process_text_file(input_path, output_csv_path):
    # 最新のファイルを取得
    if os.path.isdir(input_path):
        files = [f for f in os.listdir(input_path) if f.endswith('.txt')]
        if not files:
            print("No text files found in the directory.")
            return
        latest_file = max(files, key=lambda x: os.path.getmtime(os.path.join(input_path, x)))
        input_file_path = os.path.join(input_path, latest_file)
    else:
        input_file_path = input_path

    # ファイルを開き、内容を「。」で分割
    with open(input_file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    sentences = content.split('。')

    # CSVファイルの作成
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    with open(output_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['text', 'len', 'system']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence:
                writer.writerow({
                    'text': sentence,
                    'len': len(sentence),
                    'system': 1
                })
    
    print(f"CSV file has been created: {output_csv_path}")

# # 使用例
# input_path = 'demo/preprocess/data/summary_kana_kakasi'  # テキストファイルが入っているディレクトリ
# output_csv_path = 'output_directory/output_sentences.csv'  # 出力CSVファイル名
# process_text_file(input_path, output_csv_path)
