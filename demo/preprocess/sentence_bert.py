import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import pandas as pd
import os

class SentenceLukeJapanese:
    def __init__(self, model_name_or_path, device=None):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModel.from_pretrained(model_name_or_path)
        self.model.eval()

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model.to(self.device)

    def _mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]  # 最初の要素がトークンの埋め込み
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    @torch.no_grad()
    def encode(self, sentences, batch_size=8):
        all_embeddings = []
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i + batch_size]
            encoded_input = self.tokenizer(batch, padding=True, truncation=True, return_tensors='pt').to(self.device)
            model_output = self.model(**encoded_input)
            sentence_embeddings = self._mean_pooling(model_output, encoded_input['attention_mask']).cpu()
            all_embeddings.append(sentence_embeddings)
        return torch.cat(all_embeddings, dim=0)

# 汎用的なエンコードと保存関数
def encode_and_save(sentences, model, output_path):
    embeddings = model.encode(sentences)
    with open(output_path, 'wb') as f:
        pickle.dump((embeddings, sentences), f)
    print(f"Sentence embeddings and sentences saved to '{output_path}'")

# 汎用的なロード関数
def load_embeddings(file_path):
    with open(file_path, 'rb') as f:
        embeddings, sentences = pickle.load(f)
    return embeddings, sentences

# 類似度計算関数
def calculate_similarity(embeddings, new_embedding):
    similarity_scores = cosine_similarity(new_embedding, embeddings)
    return similarity_scores

# 最も類似する文章を取得する関数
def find_most_similar_sentence(sentences, new_sentence_embedding, existing_embeddings):
    similarity_scores = calculate_similarity(existing_embeddings, new_sentence_embedding)
    max_index = np.argmax(similarity_scores)
    return sentences[max_index], similarity_scores[0, max_index], max_index

# 使用例
if __name__ == "__main__":
    MODEL_NAME = "sonoisa/sentence-luke-japanese-base-lite"
    model = SentenceLukeJapanese(MODEL_NAME)

    # CSVファイルからデータを読み込む
    file_path = 'demo/preprocess/data/Q&A/sample.csv'
    data = pd.read_csv(file_path)
    sentences = data['質問'].tolist()  # 質問列をエンコード

    # 保存ファイルのパスをCSVファイル名に基づいて設定
    output_path = os.path.splitext(file_path)[0] + '_embeddings.pkl'

    # エンコードと保存
    encode_and_save(sentences, model, output_path=output_path)

    # ベクトルと文章のロード
    loaded_embeddings, loaded_sentences = load_embeddings(output_path)

    # # 新しい文章「腎臓ってなんですか？」をエンコード
    # new_sentence = ["腎臓ってなんですか？"]
    # new_embedding = model.encode(new_sentence)

    # # 最も類似する文章を特定してプリント
    # most_similar_sentence, similarity_score, max_index = find_most_similar_sentence(loaded_sentences, new_embedding, loaded_embeddings)
    # print(f"Most similar sentence: {most_similar_sentence}")
    # print(f"Cosine similarity score: {similarity_score}")
    # print(f"Line number: {max_index + 1}")
