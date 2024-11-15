import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import pandas as pd
import os


# 標準入力から使うやつ

def load_embeddings(file_path):
    with open(file_path, 'rb') as f:
        embeddings, sentences = pickle.load(f)
    return embeddings, sentences

def calculate_similarity(embeddings, new_embedding):
    similarity_scores = cosine_similarity(new_embedding, embeddings)
    return similarity_scores

def find_most_similar_sentence(sentences, new_sentence_embedding, existing_embeddings):
    similarity_scores = calculate_similarity(existing_embeddings, new_sentence_embedding)
    max_index = np.argmax(similarity_scores)
    return sentences[max_index], similarity_scores[0, max_index], max_index

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

if __name__ == "__main__":
    # モデルの読み込み
    MODEL_NAME = "sonoisa/sentence-luke-japanese-base-lite"
    model = SentenceLukeJapanese(MODEL_NAME)

    # 保存されているベクトルと文章を読み込む
    output_path = 'demo/preprocess/data/Q&A/sample_embeddings.pkl'  # ここに保存先のファイルパスを指定
    loaded_embeddings, loaded_sentences = load_embeddings(output_path)

    # 標準入力から新しい文章を取得
    new_sentence = input("新しい文章を入力してください: ")
    new_embedding = model.encode([new_sentence])

    # 最も類似する文章を特定してプリント
    most_similar_sentence, similarity_score, max_index = find_most_similar_sentence(loaded_sentences, new_embedding, loaded_embeddings)
    print(f"最も類似している文章: {most_similar_sentence}")
    print(f"コサイン類似度: {similarity_score}")
    print(f"行番号: {max_index + 1}")
