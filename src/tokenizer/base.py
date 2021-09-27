################################################################################
#
# base API for a tokenizer
#
# Author(s): Nik Vaessen
################################################################################

from abc import abstractmethod
from typing import Dict

import torch as t

################################################################################
# base API


class BaseTokenizer:
    @abstractmethod
    def encode_string(self, string: str) -> t.Tensor:
        pass

    @abstractmethod
    def decode_tensor(self, token_tensor: t.Tensor):
        pass

    @abstractmethod
    def vocabulary_dictionary(self) -> Dict[str, int]:
        pass

    @abstractmethod
    def vocabulary_size(self) -> int:
        pass

    @abstractmethod
    def special_tokens_dictionary(self) -> Dict[str, int]:
        pass

    @abstractmethod
    def blank_token_id(self) -> int:
        pass
