"""Tokenizer 封装：文本与 token 序列之间的确定性转换。

约定（与压缩主流程的契约）：

- ``encode`` 不添加任何特殊 token（``add_special_tokens=False``），
  token 流即待编码的符号流；
- 上下文引导 token（BOS）由 :class:`~taichi_compress.model.LLMPredictor`
  内部处理，不进入符号流；
- ``decode`` 保留特殊 token（``skip_special_tokens=False``），
  byte-level BPE 分词器下 encode → decode 严格往返（有测试保证）。
"""

from __future__ import annotations

from typing import Sequence

from transformers import AutoTokenizer

__all__ = ["Tokenizer"]


class Tokenizer:
    """HuggingFace AutoTokenizer 的确定性封装。"""

    def __init__(self, model_id: str) -> None:
        """加载 tokenizer。

        Args:
            model_id: HuggingFace 模型标识，必须与 LLMPredictor 使用的一致
        """
        self._model_id = model_id
        self._tokenizer = AutoTokenizer.from_pretrained(model_id)

    @property
    def model_id(self) -> str:
        """HuggingFace 模型标识。"""
        return self._model_id

    @property
    def vocab_size(self) -> int:
        """tokenizer 词表大小（不含模型 embedding 的填充位）。"""
        return len(self._tokenizer)

    @property
    def bos_token_id(self) -> int | None:
        """起始 token id（可能为 None，由 LLMPredictor 决定回退策略）。"""
        return self._tokenizer.bos_token_id

    @property
    def eos_token_id(self) -> int | None:
        """结束 token id。"""
        return self._tokenizer.eos_token_id

    def encode(self, text: str) -> list[int]:
        """文本 → token id 序列（不添加特殊 token，空文本 → 空序列）。"""
        return self._tokenizer.encode(text, add_special_tokens=False)

    def decode(self, token_ids: Sequence[int]) -> str:
        """token id 序列 → 文本（保留特殊 token，保证严格往返）。"""
        return self._tokenizer.decode(list(token_ids), skip_special_tokens=False)

    def is_valid_token_id(self, token_id: int) -> bool:
        """token id 是否在 tokenizer 词表范围内。"""
        return 0 <= token_id < self.vocab_size
