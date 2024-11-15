# コードについて
-data
    chatGPTで作成したシナリオなどが入っている
-prompt
    chatGPTでシナリオを作成するときのプロンプトの格納場所

concat_silent_dif.py　萱沼作成、何用か忘れた
concat_silent_same.py 萱沼作成、何用か忘れた
concat_wav.py 谷口さん作成


run_tts_espnet.py
    convert_summary_to_csv.pyで作ったcsvを元に音声合成を行う。

make_kanaCSV.py
    summary_generator.py: OpenAIのAPIを使って要約を生成し、指定のディレクトリに保存する。
    kana_converter.py: 生成された要約をカナに変換する。
    summary_to_csv.py: 要約テキストを解析し、CSV形式に変換して保存する。
    text_spliter.py: カナ変換後のテキストを句点「。」で分割し、CSVに保存する。


demo/preprocess/run/run.sh　でデータ作成