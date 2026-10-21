"""GPT-2 BPE tokenizer loaded offline from ``assets/gpt2.tiktoken``.

The C++ twin feeds the very same file, regex and special token to
``tokenizers::Tiktoken`` so both sides produce identical token ids.
"""

from __future__ import annotations

from pathlib import Path

import tiktoken
from tiktoken.load import load_tiktoken_bpe

from torchtwins.config import REPO_ROOT

DEFAULT_TOKENIZER_PATH = REPO_ROOT / "assets" / "gpt2.tiktoken"
GPT2_PATTERN = r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
END_OF_TEXT = "<|endoftext|>"


# @twin Tokenizer
class Tokenizer:
    def __init__(self, path: Path = DEFAULT_TOKENIZER_PATH) -> None:
        ranks = load_tiktoken_bpe(str(path))
        self.end_of_text_id = len(ranks)
        self._encoding = tiktoken.Encoding(
            name="gpt2",
            pat_str=GPT2_PATTERN,
            mergeable_ranks=ranks,
            special_tokens={END_OF_TEXT: self.end_of_text_id},
        )

    def encode(self, text: str) -> list[int]:
        return self._encoding.encode(text, allowed_special={END_OF_TEXT})

    def decode(self, ids: list[int]) -> str:
        return self._encoding.decode(ids)
