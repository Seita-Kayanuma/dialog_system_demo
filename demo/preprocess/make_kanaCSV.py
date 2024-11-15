from modules.kana_converter import process_to_kana, get_latest_summary_file
from modules.summary_to_csv import process_to_csv
from modules.summary_generator import generate_summary
from modules.text_spliter import process_text_file
from datetime import datetime

#demo/preprocess/make_kanaCSV.py       

"""
# コードの説明
このスクリプトは、Python の subprocess モジュールを使用して、別の Python スクリプトを実行する関数 run_script を定義しています。
ChatGPTを用いて要約を作り、その後カナに変換を行う。その後、。で文章を切り、CSVの形式になるように変換する。

# ディレクトリ構造
このスクリプトは、以下のようなディレクトリ構造を前提としています。

project/
│
├── main.py  # 実行スクリプト
├── modules/  # モジュールをまとめたディレクトリ
│   ├── __init__.py  # モジュールパッケージとして認識させる空ファイル
│   ├── kana_converter.py
│   ├── summary_to_csv.py
│   └── summary_generator.py  

modules/ フォルダ内にモジュールをまとめ、main.py からインポートや実行ができるように構成されています。
"""

def main():
    # 入力パス
    condition_file = 'demo/preprocess/prompt/condition.txt'
    surgery_file = 'demo/preprocess/prompt/surgery.txt'
    
    # 現在時刻に基づいて出力ファイル名を生成
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename_prefix = f"output_{current_time}"
    print(output_filename_prefix) #必ず1行目にプリント(shの都合)
    
    # 出力パス
    summary_path = f'demo/preprocess/data/summary/{output_filename_prefix}.txt'  # 要約
    scinario_path = f'demo/preprocess/data/scinario_kanji/{output_filename_prefix}.csv'  # 漢字シナリオ
    output_path = f"demo/preprocess/data/summary_kana_kakasi/{output_filename_prefix}.txt"  # kakasiでカナに変換したもの
    csv_path = f"demo/preprocess/data/scinario/{output_filename_prefix}/data_.csv"
    
    # 要約を生成
    generate_summary(condition_file, surgery_file, summary_path)  # 引数：プロンプトの条件テキスト, 手術説明書のテキスト, 出力ディレクトリ
    # text, system のCSV形式で保存 ex) 今日は,1
    process_to_csv(summary_path, scinario_path)  # 引数：入力ファイル、出力ファイル
    
    # kakasiを利用してカナに変換
    process_to_kana(summary_path, output_path)  # 引数：入力ファイル、出力ファイル
    process_text_file(output_path, csv_path)
   
if __name__ == "__main__":
    main()

