import openai
import os
import tiktoken  # tiktokenのインポート
from datetime import datetime

# python demo/preprocess/summary.py

MODEL_NAME = 'gpt-4o-mini'  # gpt-3.5-turbo-16k, gpt-4o-mini
MAX_MODEL_TOKENS = 16384  # モデルの最大トークン数

# 環境変数からAPIキーを取得し、クライアントを作成
api_key = os.getenv("OPENAI_API_KEY")
if api_key is None:
    raise ValueError("環境変数 'OPENAI_API_KEY' が設定されていません。")
client = openai.OpenAI(api_key=api_key)

def load_text_from_file(file_path):
    """ファイルからテキストを読み込む関数"""
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    return text

def count_tokens(text, model_name=MODEL_NAME):
    """テキストのトークン数を計算する関数"""
    encoding = tiktoken.encoding_for_model(model_name)
    num_tokens = len(encoding.encode(text))
    return num_tokens

def chat_with_gpt(prompt):
    """GPTモデルとチャットする関数"""
    # プロンプトのトークン数を計算
    prompt_tokens = count_tokens(prompt)
    print(f"入力トークン数: {prompt_tokens}")

    # 応答に利用可能なトークン数を計算
    available_tokens = MAX_MODEL_TOKENS - prompt_tokens

    if available_tokens <= 0:
        raise ValueError("プロンプトが長すぎます。プロンプトのトークン数を減らしてください。")

    # `max_tokens`を設定
    max_response_tokens = available_tokens
    print(f"応答に利用可能なトークン数: {max_response_tokens}")

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": prompt}
        ],
        max_tokens=max_response_tokens,
        temperature=0.7,
        top_p=0.9,
        n=1,
        presence_penalty=0.6,
        frequency_penalty=0.0,
        logit_bias=None,
        user="kayanuma"
    )
    return response.choices[0].message.content.strip()

def create_prompt(conditions_file, instructions_file):
    """プロンプトを作成する関数"""
    base_prompt = '''あなたは手術説明のための対話システムです。以下の説明書の内容を説明しなければなりません。相手は医療知識がない患者さんです。説明書中の情報の必要十分な説明を行いなさい。口語での１つの文書で回答しなさい。
条件：
{conditions}

[説明書]
{instructions}
'''

    # ファイルの存在チェックと読み込み
    if not os.path.exists(conditions_file):
        raise FileNotFoundError(f"指定されたファイルが見つかりません: {conditions_file}")
    if not os.path.exists(instructions_file):
        raise FileNotFoundError(f"指定されたファイルが見つかりません: {instructions_file}")

    conditions = load_text_from_file(conditions_file)
    instructions = load_text_from_file(instructions_file)

    # プロンプトのフォーマット
    prompt = base_prompt.format(conditions=conditions, instructions=instructions)
    return prompt

def main():
    """メイン実行関数"""
    # ファイルパスの設定
    condition_file = 'demo/preprocess/prompt/condition.txt'
    surgery_file = 'demo/preprocess/prompt/surgery.txt'
    output_dir = 'demo/preprocess/data/summary'  # 出力ディレクトリ

    # ディレクトリが存在しない場合は作成
    os.makedirs(output_dir, exist_ok=True)

    # 現在の日時を取得してファイル名に使用
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = os.path.join(output_dir, f'summary_{timestamp}.txt')

    # プロンプトを作成
    prompt = create_prompt(condition_file, surgery_file)

    # GPTモデルに問い合わせ
    reply = chat_with_gpt(prompt)

    # 応答をファイルに保存
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(reply)

    print(f"応答をファイルに保存しました: {output_file}")

if __name__ == '__main__':
    main()
