"""解压器主流程 —— 压缩的镜像。

流程：解析容器头 → 按头部元数据加载（或校验已给的）同一模型 →
逐 token「LLM 预测 → 算术解码」→ tokenizer 还原文本 → CRC32 校验。
校验失败说明模型、配置或比特流与压缩侧不一致，绝不输出错误结果。
"""

from __future__ import annotations

import time
import zlib
from dataclasses import dataclass
from pathlib import Path

from .arithmetic import ArithmeticDecoder, build_cdf
from .compressor import ContainerHeader, check_context_limit, parse_header
from .model import LLMPredictor, PredictorConfig
from .utils import BitReader

__all__ = ["DecompressionStats", "decompress_text", "decompress_file"]


@dataclass(frozen=True)
class DecompressionStats:
    """解压结果统计（基准测试与 CLI 共用）。"""

    compressed_size: int  # 压缩文件字节数
    original_size: int  # 还原的 UTF-8 字节数
    token_count: int
    duration_seconds: float  # 解压耗时（不含模型加载）

    @property
    def ratio(self) -> float:
        """压缩比 original / compressed。"""
        return self.original_size / self.compressed_size if self.compressed_size else 0.0

    @property
    def speed_bytes_per_second(self) -> float:
        """解压吞吐（还原字节 / 秒）。"""
        if self.duration_seconds <= 0:
            return 0.0
        return self.original_size / self.duration_seconds


def decompress_text(
    data: bytes,
    predictor: LLMPredictor | None = None,
    *,
    device: str | None = None,
) -> str:
    """解压容器字节串，返回还原文本。

    模型标识以容器头为准：未提供 predictor 时按头部 model_id / logit_scale
    加载；提供时校验其与头部一致（防止拿错模型解出静默错误的数据，
    CRC32 兜底校验最终仍会拦截）。

    Args:
        data: ``compress_text`` / ``compress_file`` 产生的字节串
        predictor: 已加载的预测器；None 时按容器头现场加载
        device: 仅在 predictor 为 None 时生效

    Returns:
        还原的文本

    Raises:
        ValueError: 容器头非法、模型/配置不匹配、比特流损坏或校验失败
    """
    return _decompress(data, predictor, device)[0]


def decompress_file(
    input_path: str | Path,
    output_path: str | Path,
    predictor: LLMPredictor | None = None,
    *,
    device: str | None = None,
) -> DecompressionStats:
    """解压 .tc 文件，写出还原的 UTF-8 文本并返回统计。"""
    data = Path(input_path).read_bytes()
    text, stats = _decompress(data, predictor, device)
    Path(output_path).write_bytes(text.encode("utf-8"))
    return stats


def _decompress(
    data: bytes,
    predictor: LLMPredictor | None,
    device: str | None,
) -> tuple[str, DecompressionStats]:
    """解压主循环：逐 token「LLM 预测 → 算术解码」，镜像压缩侧调用序列。"""
    header, body_start = parse_header(data)
    if predictor is None:
        predictor = LLMPredictor(
            PredictorConfig(
                model_id=header.model_id,
                device=device,
                logit_scale=header.logit_scale,
            )
        )
    else:
        _verify_predictor(predictor, header)

    started = time.perf_counter()
    check_context_limit(predictor, header.num_tokens)
    predictor.reset()  # 复用的预测器必须回到全新状态

    decoder = ArithmeticDecoder(BitReader(data[body_start:]), header.precision)
    tokens: list[int] = []
    for _ in range(header.num_tokens):
        probs = predictor.predict_next_token_probabilities(tokens)
        tokens.append(decoder.decode_symbol(build_cdf(probs, header.precision)))
    text = predictor.tokenizer.decode(tokens)
    raw = text.encode("utf-8")
    if len(raw) != header.original_size or zlib.crc32(raw) != header.crc32:
        raise ValueError(
            "解压校验失败：还原文本与压缩侧不一致"
            "（模型、量化配置或比特流可能损坏/被篡改）"
        )
    stats = DecompressionStats(
        compressed_size=len(data),
        original_size=len(raw),
        token_count=header.num_tokens,
        duration_seconds=time.perf_counter() - started,
    )
    return text, stats


def _verify_predictor(predictor: LLMPredictor, header: ContainerHeader) -> None:
    """校验外部传入的预测器与容器头声明的配置一致。"""
    if predictor.model_id != header.model_id:
        raise ValueError(
            f"预测器模型 {predictor.model_id!r} 与文件头 {header.model_id!r} 不一致"
        )
    if predictor.logit_scale != header.logit_scale:
        raise ValueError(
            f"logit 量化精度不一致：预测器 {predictor.logit_scale}，"
            f"文件头 {header.logit_scale}"
        )
