# -*- coding: utf-8 -*-
import getpass
import os
from typing import List, Optional
from pydantic import BaseModel, Field
from openai import OpenAI
import csv
import json
import datetime

# python demo/preprocess/structured_outputs.py 
# Q&Aを作成
surgery_manual_path = 'demo/preprocess/prompt/surgery.txt'
script_path = 'demo/preprocess/data/summary/output_20241109_180145.txt'
qa_path = 'demo/preprocess/prompt/qa_example.txt'



def load_text_from_file(file_path):
    """ファイルからテキストを読み込む関数"""
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

# APIキーの設定
if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = getpass.getpass("OpenAI API Key:")

# 基本的なレスポンスの構造定義
class ResponseStep(BaseModel):
    steps: List[str]
    expected_question: List[str]
    answer: List[str]

# OpenAIクライアントのインスタンス作成
client = OpenAI()

surgery_manual = load_text_from_file(surgery_manual_path)
script = load_text_from_file(script_path)
qa_example = load_text_from_file(qa_path)

# Prompt construction
prompt = f"""
あなたは医者です。手術の説明書の内容を読んで、患者に説明している状況です。この時、患者さんから想定される質問とその質問に対する回答を、20組考えてください。[手術説明書]の内容を参考にしてください。患者さんが不安を持たず、理解しやすい形での説明を目指してください。

# 解説部分

- 手術の詳細を説明する際、専門用語を避け、患者が理解しやすい簡単な言葉を用いてください。
- 患者の気持ちを考慮し、安心感を与えた内容にしてください。不必要に不安を煽らないようにしてください。
- 想定される質問には具体的な例や状況を織り交ぜて適切に答え、患者が決断しやすいようにサポートしてください。

# 出力形式（CSV）

"質問番号","質問","回答"
"質問1","患者さんからの想定される質問1","質問1に対する医師としての回答"
"質問2","患者さんからの想定される質問2","質問2に対する医師としての回答"
...
"質問20","患者さんからの想定される質問20","質問20に対する医師としての回答"

# [手術説明書]
{surgery_manual}


# Examples
{qa_example}


# Notes

- 患者さんが不安を抱える場面では、質問にストレートに答えるだけでなく、安心感を与えるための追加説明を加えてください。
- 手術説明書の内容を正確に読み取り、その内容に基づいて具体的でわかりやすい説明を心がけてください。
- 専門的な内容を話す際は、簡潔にし、必要に応じて図や比喩を使って説明を工夫してください。
- 患者さんが理解しやすく、安心できるコミュニケーションを心がけてください。
"""

# リクエストの送信
response = client.beta.chat.completions.parse(
    model="gpt-4o-mini",
    temperature=0,
    messages=[
        {
            "role": "system",
            "content": prompt,
        }
    ],
    response_format=ResponseStep,
)


# 保存ファイル名を毎回変えるためのタイムスタンプ生成
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
csv_output_path = f'demo/preprocess/data/Q&A/patient_questions_and_answers_{timestamp}.csv'

# 出力ディレクトリが存在しない場合は作成する
output_dir = os.path.dirname(csv_output_path)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# CSVファイルに結果を書き込む関数
def save_responses_to_csv(file_path: str, responses):
    with open(file_path, 'w', newline='', encoding='utf-8') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["質問", "回答"])
        
        # レスポンスのcontentをパースしてJSONオブジェクトとして扱う
        content = json.loads(responses.choices[0].message.content)
        for question, answer in zip(content['expected_question'], content['answer']):
            csv_writer.writerow([question, answer])

# レスポンスをCSVに書き込む
if response:
    save_responses_to_csv(csv_output_path, response)
    print(f"質問と回答のペアが {csv_output_path} に保存されました。")
else:
    print("レスポンスの生成に失敗しました。")


