import difflib

# python sample/get_similar_word.py

def main():
    text = "アーエーダトオモイマス"
    keywords = ["エーキャンパス", "ビーキャンパス", "シーキャンパス"]
    best_match = None
    highest_ratio = 0.0
    for keyword in keywords:
        for i in range(len(text) - len(keyword) + 1):
            substring = text[i:i+len(keyword)]
            ratio = difflib.SequenceMatcher(None, substring, keyword).ratio()
            if ratio > highest_ratio:
                highest_ratio = ratio
                best_match = keyword
    if best_match:
        print("最も近い単語:", best_match)
    else:
        print("近い単語が見つかりませんでした")
    

if __name__ == "__main__":
    main()