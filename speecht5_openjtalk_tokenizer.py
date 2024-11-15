import json
import logging
import os
from pathlib import Path
import re
from transformers import SpeechT5Tokenizer
from transformers.models.speecht5.tokenization_speecht5 import (
    PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES,
)
from itertools import chain
from typing import List, Optional, Tuple


logger = logging.getLogger(__name__)

NP_CHARCTERS = " !\"#$%&'()=~|`{+*}<>?_-^\\@[;:],./　！”＃＄％＆’（）＝～｜｀｛＋＊｝＜＞？＿ー＾￥＠「；：」、。・`"


def _g2p_with_np(text: str, np_lsit: str) -> List[str]:
    from pyopenjtalk import g2p

    np_pattern = re.compile(f"([{re.escape(np_lsit)}])")

    return list(
        chain.from_iterable(
            [
                (text,) if text in np_lsit else g2p(text, kana=False, join=False)
                for text in np_pattern.split(text)
                if len(text) > 0
            ]
        )
    )


VOCAB_FILES_NAMES = {
    "vocab_file": "vocab.json",
}

PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "esnya/japanese_speecht5_tts": "https://huggingface.co/esnya/japanese_speecht5_tts/resolve/main/vocab.json",
    },
}


class SpeechT5OpenjtalkTokenizer(SpeechT5Tokenizer):
    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        vocab_file,
        bos_token: str = "<s>",
        eos_token: str = "</s>",
        unk_token: str = "<unk>",
        pad_token: str = "<pad>",
        non_phenome_characters: str = NP_CHARCTERS,
        **kwargs,
    ):
        try:
            super().__init__(
                vocab_file=None,
                bos_token=bos_token,
                eos_token=eos_token,
                unk_token=unk_token,
                pad_token=pad_token,
                **kwargs,
            )
        except TypeError:
            pass

        self.non_phenome_characters = non_phenome_characters
        self.vocab_file = vocab_file

        self._load_vocab()

    def _load_vocab(self):
        if isinstance(self.vocab_file, str) and self.vocab_file.endswith(".json"):
            with open(self.vocab_file, encoding="utf-8") as f:
                self.label2id = json.load(f)
            self.id2label = {v: k for k, v in self.label2id.items()}

    @property
    def bos_token_id(self):
        return super().bos_token_id

    @property
    def vocab_size(self):
        return len(self.label2id)

    def get_vocab(self):
        return self.label2id

    def __getstate__(self):
        state = super().__getstate__()
        del state["sp_model"]
        return state

    def __setstate__(self, d):
        self.__dict__ = d
        self._load_vocab()

    def save_vocabulary(
        self, save_directory: str, filename_prefix: Optional[str] = None
    ):
        if filename_prefix is None:
            filename_prefix = ".json"

        save_path = Path(save_directory)
        if not save_path.is_dir():
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return

        vocab_path = Path(save_directory) / Path(f"vocab{filename_prefix}")
        vocab_path.parent.mkdir(parents=True, exist_ok=True)
        with open(vocab_path, "w", encoding="utf-8") as f:
            json.dump(self.label2id, f, ensure_ascii=False, indent=2)

        return (str(vocab_path),)

    def _tokenize(self, text: str) -> List[str]:
        return _g2p_with_np(text, self.non_phenome_characters)

    def _convert_token_to_id(self, token):
        return self.label2id.get(token, self.label2id.get(self.unk_token))

    def _convert_id_to_token(self, index):
        return self.id2label.get(index, self.unk_token)
