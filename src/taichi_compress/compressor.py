"""压缩器主流程 —— 阴阳合璧（压缩侧）。

流程：读取文本 → tokenizer 分词 → 逐 token 由 LLM 预测概率分布 →
算术编码 → 输出含容器头的压缩文件。

容器格式（版本 1，逐字节确定）::

    偏移 0..5    magic = b"TAICHI"
    偏移 6       格式版本 = 1
    偏移 7       CDF 量子精度（1~30 比特）
    varint       logit 量化精度（0 表示不量化）
    varint       原始 UTF-8 字节数
    varint       token 总数
    varint+bytes 模型标识（长度前缀的 UTF-8）
    4 字节       原始字节的 CRC32（解压侧完整性校验）
    其后         算术编码比特流（MSB-first，末尾补 0 对齐字节）

限制：整文件单段压缩（KV Cache 连续增长），受模型上下文长度约束
（Qwen2.5-0.5B 为 32768 token）。大文件分块处理属于 Phase 2。
"""

from __future__ import annotations

import math
import time
import zlib
from dataclasses import dataclass
from pathlib import Path

from .arithmetic import (
    DEFAULT_PRECISION,
    MAX_PRECISION,
    MIN_PRECISION,
    ArithmeticEncoder,
    build_cdf,
)
from .model import DEFAULT_MODEL_ID, LLMPredictor, PredictorConfig
from .utils import BitWriter, read_varint, write_varint

__all__ = [
    "MAGIC",
    "CONTAINER_VERSION",
    "ContainerHeader",
    "CompressionStats",
    "check_context_limit",
    "parse_header",
    "read_text_file",
    "compress_text",
    "compress_file",
]

MAGIC = b"TAICHI"
CONTAINER_VERSION = 1
_HEADER_FIXED = len(MAGIC) + 2  # magic + version + precision


@dataclass(frozen=True)
class ContainerHeader:
    """压缩容器头：解压所需的全部元数据。"""

    precision: int  # CDF 量子精度（比特）
    logit_scale: int | None  # logit 量化精度（None 表示未量化）
    original_size: int  # 原始 UTF-8 字节数
    num_tokens: int  # token 总数
    model_id: str  # 模型标识（解压侧必须一致）
    crc32: int  # 原始字节的 CRC32
    version: int = CONTAINER_VERSION

    def to_bytes(self) -> bytes:
        """序列化为容器头字节串（长度可变）。"""
        buf = bytearray()
        buf += MAGIC
        buf.append(self.version)
        buf.append(self.precision)
        write_varint(buf, self.logit_scale or 0)
        write_varint(buf, self.original_size)
        write_varint(buf, self.num_tokens)
        model_bytes = self.model_id.encode("utf-8")
        write_varint(buf, len(model_bytes))
        buf += model_bytes
        buf += self.crc32.to_bytes(4, "big")
        return bytes(buf)


def parse_header(data: bytes) -> tuple[ContainerHeader, int]:
    """解析容器头。

    Returns:
        (容器头, 算术比特流起始偏移)

    Raises:
        ValueError: magic / 版本 / 精度非法，或文件头被截断
    """
    if len(data) < _HEADER_FIXED + 4 or data[: len(MAGIC)] != MAGIC:
        raise ValueError("不是太极压缩文件（magic 不符或数据过短）")
    version = data[len(MAGIC)]
    if version != CONTAINER_VERSION:
        raise ValueError(f"不支持的容器版本: {version}")
    precision = data[len(MAGIC) + 1]
    if not MIN_PRECISION <= precision <= MAX_PRECISION:
        raise ValueError(f"CDF 精度非法: {precision}")
    pos = _HEADER_FIXED
    scale, pos = read_varint(data, pos)
    original_size, pos = read_varint(data, pos)
    num_tokens, pos = read_varint(data, pos)
    model_len, pos = read_varint(data, pos)
    if pos + model_len + 4 > len(data):
        raise ValueError("容器头被截断")
    try:
        model_id = data[pos : pos + model_len].decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("模型标识解码失败") from exc
    pos += model_len
    crc = int.from_bytes(data[pos : pos + 4], "big")
    header = ContainerHeader(
        precision=precision,
        logit_scale=scale or None,
        original_size=original_size,
        num_tokens=num_tokens,
        model_id=model_id,
        crc32=crc,
        version=version,
    )
    return header, pos + 4


@dataclass(frozen=True)
class CompressionStats:
    """压缩结果统计（基准测试与 CLI 共用）。"""

    original_size: int  # 原始 UTF-8 字节数
    compressed_size: int  # 压缩文件字节数（含容器头）
    token_count: int
    ideal_bits: float  # -Σ log2 P(token)，理论编码下界（不含容器头）
    duration_seconds: float  # 压缩耗时（不含模型加载）

    @property
    def ratio(self) -> float:
        """压缩比 original / compressed。"""
        return self.original_size / self.compressed_size if self.compressed_size else 0.0

    @property
    def bits_per_byte(self) -> float:
        """每原始字节的压缩比特数（bpb）；原始为空时为 0。"""
        if not self.original_size:
            return 0.0
        return self.compressed_size * 8 / self.original_size

    @property
    def bits_per_token(self) -> float:
        """每 token 的压缩比特数；无 token 时为 0。"""
        return self.compressed_size * 8 / self.token_count if self.token_count else 0.0

    @property
    def speed_bytes_per_second(self) -> float:
        """压缩吞吐（原始字节 / 秒）。"""
        if self.duration_seconds <= 0:
            return 0.0
        return self.original_size / self.duration_seconds


def check_context_limit(predictor: LLMPredictor, num_tokens: int) -> None:
    """校验 token 数（加内部引导 token）不超过模型上下文上限。"""
    limit = predictor.max_context_tokens
    if limit and num_tokens + 1 > limit:
        raise ValueError(
            f"token 数 {num_tokens}（含引导 token）超过模型上下文上限 {limit}；"
            "大文件分块压缩属于 Phase 2"
        )


def compress_text(
    text: str,
    predictor: LLMPredictor | None = None,
    *,
    precision: int = DEFAULT_PRECISION,
    model_id: str = DEFAULT_MODEL_ID,
    device: str | None = None,
) -> bytes:
    """压缩一段文本，返回容器字节串（容器头 + 算术编码比特流）。

    Args:
        text: 待压缩文本（任意 UTF-8 内容，可为空）
        predictor: 已加载的预测器；None 时按 model_id / device 现场加载
        precision: CDF 量子精度，1~30 比特
        model_id / device: 仅在 predictor 为 None 时生效

    Returns:
        压缩字节串
    """
    return _compress(text, predictor, precision, model_id, device)[0]


def compress_file(
    input_path: str | Path,
    output_path: str | Path,
    predictor: LLMPredictor | None = None,
    *,
    precision: int = DEFAULT_PRECISION,
    model_id: str = DEFAULT_MODEL_ID,
    device: str | None = None,
) -> CompressionStats:
    """压缩文本文件（输入必须是合法 UTF-8），写出 .tc 容器并返回统计。"""
    text = read_text_file(input_path)
    data, stats = _compress(text, predictor, precision, model_id, device)
    Path(output_path).write_bytes(data)
    return stats


def _compress(
    text: str,
    predictor: LLMPredictor | None,
    precision: int,
    model_id: str,
    device: str | None,
) -> tuple[bytes, CompressionStats]:
    """压缩主循环：逐 token「LLM 预测 → 算术编码」。"""
    if predictor is None:
        predictor = LLMPredictor(PredictorConfig(model_id=model_id, device=device))
    started = time.perf_counter()
    raw = text.encode("utf-8")
    tokens = predictor.tokenizer.encode(text)
    check_context_limit(predictor, len(tokens))
    predictor.reset()  # 复用的预测器必须回到全新状态

    writer = BitWriter()
    encoder = ArithmeticEncoder(writer, precision)  # 构造时校验 precision
    context: list[int] = []
    ideal_bits = 0.0
    for token in tokens:
        probs = predictor.predict_next_token_probabilities(context)
        encoder.encode_symbol(token, build_cdf(probs, precision))
        ideal_bits -= math.log2(max(float(probs[token]), 1e-300))
        context.append(token)
    encoder.finish()

    header = ContainerHeader(
        precision=precision,
        logit_scale=predictor.logit_scale,
        original_size=len(raw),
        num_tokens=len(tokens),
        model_id=predictor.model_id,
        crc32=zlib.crc32(raw),
    )
    data = header.to_bytes() + writer.to_bytes()
    stats = CompressionStats(
        original_size=len(raw),
        compressed_size=len(data),
        token_count=len(tokens),
        ideal_bits=ideal_bits,
        duration_seconds=time.perf_counter() - started,
    )
    return data, stats


def read_text_file(path: str | Path) -> str:
    """读取文本文件，强制 UTF-8（Phase 1 仅支持纯文本）。"""
    raw = Path(path).read_bytes()
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{path} 不是合法的 UTF-8 文本（Phase 1 仅支持纯文本）") from exc
