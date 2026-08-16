"""太极压缩 (TaiChi Compression) —— LLM 概率预测 + 算术编码的无损文本压缩。

太极生两仪：预测为"阴"（LLM 根据上文给出下一 token 的概率分布），
编码为"阳"（算术编码把概率转化为比特流）。
"""

from __future__ import annotations

from .arithmetic import (
    DEFAULT_PRECISION,
    ArithmeticDecoder,
    ArithmeticEncoder,
    build_cdf,
    decode,
    encode,
)
from .utils import BitReader, BitWriter

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "DEFAULT_PRECISION",
    "ArithmeticEncoder",
    "ArithmeticDecoder",
    "build_cdf",
    "encode",
    "decode",
    "BitWriter",
    "BitReader",
]
