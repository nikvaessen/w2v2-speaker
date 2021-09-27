################################################################################
#
# Tokenizer for the wav2vec2 network.
#
# Author(s): Nik Vaessen
################################################################################

from typing import Dict, List

from attr import dataclass
from transformers.models.wav2vec2 import Wav2Vec2CTCTokenizer

import torch as t

from src.tokenizer.base import BaseTokenizer

################################################################################
# wrapper around huggingfacae tokenizer


@dataclass
class Wav2vec2TokenizerConfig:
    tokenizer_huggingface_id: str


class Wav2vec2Tokenizer(BaseTokenizer):
    def __init__(self, cfg: Wav2vec2TokenizerConfig):
        self.tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(
            cfg.tokenizer_huggingface_id
        )

    def encode_string(self, string: str) -> t.Tensor:
        return t.IntTensor(self.tokenizer(string).input_ids)

    def decode_tensor(self, token_tensor: t.Tensor) -> str:
        assert len(token_tensor.shape) == 1

        decoded_str = self.tokenizer.decode(token_tensor)

        return decoded_str

    def vocabulary_dictionary(self) -> Dict[str, int]:
        return self.tokenizer.get_vocab()

    def vocabulary_size(self) -> int:
        return self.tokenizer.vocab_size

    def special_tokens_dictionary(self) -> Dict[str, int]:
        return self.tokenizer.special_tokens_map

    def blank_token_id(self) -> int:
        return 0
