"""单文件基准测试：太极压缩 vs 传统压缩算法（gzip -9 / xz -9 / zstd -19）。

指标：压缩后大小、压缩比、bpb（bit/byte）、压缩/解压速度、往返正确性。
太极压缩的计时不含模型加载；内存以进程峰值 RSS 计（模型权重占主导）。

任务 6 将基于 enwik8 等标准语料做完整评估，本模块提供可复用的对比原语
与结果表格渲染（``run_benchmark`` / ``format_benchmark_table``）。
"""

from __future__ import annotations

import gzip
import lzma
import resource
import sys
import time
import unicodedata
from dataclasses import dataclass
from typing import Callable

from .arithmetic import DEFAULT_PRECISION

__all__ = [
    "AlgorithmResult",
    "benchmark_classic",
    "benchmark_taichi",
    "run_benchmark",
    "sample_blocks",
    "evaluate_corpus_classic",
    "evaluate_corpus_taichi",
    "evaluate_corpus",
    "format_benchmark_table",
    "peak_rss_mib",
]


@dataclass(frozen=True)
class AlgorithmResult:
    """单个算法在单次基准上的结果。"""

    name: str  # 算法名（表格展示用）
    original_size: int  # 原始字节数
    compressed_size: int  # 压缩后字节数（不可用时为 0）
    compress_seconds: float | None  # 压缩耗时；None 表示该算法不可用
    decompress_seconds: float | None  # 解压耗时
    roundtrip_ok: bool | None  # 往返校验结果；None 表示未验证/不可用
    note: str = ""  # 备注（如不可用原因）

    @property
    def available(self) -> bool:
        """该算法是否实际参与了基准。"""
        return self.compress_seconds is not None

    @property
    def ratio(self) -> float:
        """压缩比 original / compressed。"""
        if not self.compressed_size:
            return 0.0
        return self.original_size / self.compressed_size

    @property
    def bits_per_byte(self) -> float:
        """每原始字节的压缩比特数（bpb）。"""
        if not self.original_size:
            return 0.0
        return self.compressed_size * 8 / self.original_size

    @property
    def compress_speed(self) -> float:
        """压缩吞吐（字节/秒）。"""
        if not self.compress_seconds:
            return 0.0
        return self.original_size / self.compress_seconds

    @property
    def decompress_speed(self) -> float:
        """解压吞吐（字节/秒）。"""
        if not self.decompress_seconds:
            return 0.0
        return self.original_size / self.decompress_seconds


def benchmark_classic(data: bytes) -> list[AlgorithmResult]:
    """对 gzip -9 / xz -9 / zstd -19 运行压缩-解压基准。

    zstd 优先使用 zstandard 包，其次 Python 3.14+ 的 compression.zstd，
    两者皆无则标记为不可用（不报错，便于在精简环境中运行）。
    """
    results = [
        _run(name, data, compress_fn, decompress_fn)
        for name, compress_fn, decompress_fn in _classic_codecs()
    ]
    if _zstd_codec() is None:
        results.append(
            AlgorithmResult("zstd -19", len(data), 0, None, None, None, note="zstandard 未安装")
        )
    return results


def benchmark_taichi(
    text: str,
    predictor: "object | None" = None,
    *,
    model_id: str | None = None,
    device: str | None = None,
    dtype: str | None = None,
    precision: int = DEFAULT_PRECISION,
) -> AlgorithmResult:
    """太极压缩基准：压缩 + 解压 + 往返校验（计时不含模型加载）。

    Args:
        text: 待压缩文本
        predictor: 已加载的 LLMPredictor；None 时按 model_id / device / dtype 现场加载
        model_id / device / dtype: 仅在 predictor 为 None 时生效
        precision: CDF 量子精度
    """
    from .model import DEFAULT_MODEL_ID, LLMPredictor, PredictorConfig  # 延迟导入：本模块其余功能不依赖 torch

    if predictor is None:
        predictor = LLMPredictor(
            PredictorConfig(
                model_id=model_id or DEFAULT_MODEL_ID, device=device, dtype=dtype
            )
        )
    raw = text.encode("utf-8")
    t0 = time.perf_counter()
    from .compressor import compress_text

    data = compress_text(text, predictor, precision=precision)
    t1 = time.perf_counter()
    from .decompressor import decompress_text

    restored = decompress_text(data, predictor)
    t2 = time.perf_counter()
    return AlgorithmResult(
        name="taichi",
        original_size=len(raw),
        compressed_size=len(data),
        compress_seconds=t1 - t0,
        decompress_seconds=t2 - t1,
        roundtrip_ok=restored == text,
    )


def run_benchmark(
    text: str,
    predictor: "object | None" = None,
    *,
    model_id: str | None = None,
    device: str | None = None,
    dtype: str | None = None,
    precision: int = DEFAULT_PRECISION,
) -> list[AlgorithmResult]:
    """运行完整对比（太极压缩 + 传统算法），返回结果列表（太极在前）。"""
    taichi = benchmark_taichi(
        text, predictor, model_id=model_id, device=device, dtype=dtype, precision=precision
    )
    return [taichi] + benchmark_classic(text.encode("utf-8"))


def peak_rss_mib() -> float:
    """进程峰值 RSS（MiB）。macOS 的 ru_maxrss 单位是字节，Linux 是 KiB。"""
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    kib = raw / 1024 if sys.platform == "darwin" else float(raw)
    return kib / 1024


def _run(
    name: str,
    data: bytes,
    compress_fn: Callable[[bytes], bytes],
    decompress_fn: Callable[[bytes, int], bytes],
) -> AlgorithmResult:
    """运行一次压缩-解压并计时校验。"""
    t0 = time.perf_counter()
    compressed = compress_fn(data)
    t1 = time.perf_counter()
    restored = decompress_fn(compressed, len(data))
    t2 = time.perf_counter()
    return AlgorithmResult(
        name=name,
        original_size=len(data),
        compressed_size=len(compressed),
        compress_seconds=t1 - t0,
        decompress_seconds=t2 - t1,
        roundtrip_ok=restored == data,
    )


def _zstd_codec() -> tuple[Callable[[bytes], bytes], Callable[[bytes, int], bytes]] | None:
    """返回 zstd -19 的压缩/解压函数对；不可用时返回 None。"""
    try:
        import zstandard

        return (
            lambda d: zstandard.ZstdCompressor(level=19).compress(d),
            lambda d, n: zstandard.ZstdDecompressor().decompress(d, max_output_size=n),
        )
    except ImportError:  # pragma: no cover - 环境相关
        pass
    try:  # Python 3.14+ 标准库
        from compression import zstd

        return (
            lambda d: zstd.compress(d, level=19),
            lambda d, _: zstd.decompress(d),
        )
    except ImportError:  # pragma: no cover
        return None


def _classic_codecs() -> list[tuple[str, Callable[[bytes], bytes], Callable[[bytes, int], bytes]]]:
    """可用传统算法的 (名称, 压缩函数, 解压函数) 列表。"""
    codecs = [
        ("gzip -9", lambda d: gzip.compress(d, 9), lambda d, _: gzip.decompress(d)),
        ("xz -9", lambda d: lzma.compress(d, preset=9), lambda d, _: lzma.decompress(d)),
    ]
    zstd = _zstd_codec()
    if zstd is not None:
        codecs.append(("zstd -19", zstd[0], zstd[1]))
    return codecs


# —— 语料级评测（分段采样 + 逐块独立压缩，任务 6） ——


def sample_blocks(text: str, num_blocks: int, block_chars: int) -> list[str]:
    """把文本等分为 num_blocks 段，取每段开头的 block_chars 个字符作为采样块。

    覆盖全文、确定性（无随机性）；block_chars 超过段长时相邻块会重叠。
    大语料（如 enwik8）上逐块独立压缩，模拟分块处理并保证各算法同口径。

    Raises:
        ValueError: 参数非正，或文本不足以分段
    """
    if num_blocks <= 0 or block_chars <= 0:
        raise ValueError("num_blocks 与 block_chars 必须为正整数")
    segment = len(text) // num_blocks
    if segment < 1:
        raise ValueError(f"文本长度 {len(text)} 不足以分成 {num_blocks} 段")
    return [text[i * segment : i * segment + block_chars] for i in range(num_blocks)]


def evaluate_corpus_classic(
    blocks: list[str],
    name: str,
    compress_fn: Callable[[bytes], bytes],
    decompress_fn: Callable[[bytes, int], bytes],
) -> AlgorithmResult:
    """对采样块逐块运行传统算法压缩-解压并汇总（与 taichi 同口径）。"""
    total_original = 0
    total_compressed = 0
    compress_seconds = 0.0
    decompress_seconds = 0.0
    roundtrip_ok = True
    for block in blocks:
        raw = block.encode("utf-8")
        t0 = time.perf_counter()
        compressed = compress_fn(raw)
        t1 = time.perf_counter()
        restored = decompress_fn(compressed, len(raw))
        t2 = time.perf_counter()
        total_original += len(raw)
        total_compressed += len(compressed)
        compress_seconds += t1 - t0
        decompress_seconds += t2 - t1
        roundtrip_ok = roundtrip_ok and restored == raw
    return AlgorithmResult(
        name=name,
        original_size=total_original,
        compressed_size=total_compressed,
        compress_seconds=compress_seconds,
        decompress_seconds=decompress_seconds,
        roundtrip_ok=roundtrip_ok,
        note=f"{len(blocks)} 个采样块逐块独立压缩",
    )


def evaluate_corpus_taichi(
    blocks: list[str],
    predictor: "object | None" = None,
    *,
    model_id: str | None = None,
    device: str | None = None,
    dtype: str | None = None,
    precision: int = DEFAULT_PRECISION,
    progress: Callable[[int, int, int, int], None] | None = None,
) -> AlgorithmResult:
    """对采样块逐块运行太极压缩-解压并汇总（计时不含模型加载）。

    Args:
        blocks: 采样文本块
        predictor: 已加载的 LLMPredictor；None 时按 model_id / device / dtype 现场加载
        model_id / device / dtype: 仅在 predictor 为 None 时生效
        precision: CDF 量子精度
        progress: 进度回调 progress(done, total, 原始字节, 压缩字节)
    """
    from .compressor import compress_text
    from .decompressor import decompress_text
    from .model import DEFAULT_MODEL_ID, LLMPredictor, PredictorConfig

    if predictor is None:
        predictor = LLMPredictor(
            PredictorConfig(
                model_id=model_id or DEFAULT_MODEL_ID, device=device, dtype=dtype
            )
        )
    total_original = 0
    total_compressed = 0
    compress_seconds = 0.0
    decompress_seconds = 0.0
    roundtrip_ok = True
    for i, block in enumerate(blocks, start=1):
        raw = block.encode("utf-8")
        t0 = time.perf_counter()
        data = compress_text(block, predictor, precision=precision)
        t1 = time.perf_counter()
        restored = decompress_text(data, predictor)
        t2 = time.perf_counter()
        total_original += len(raw)
        total_compressed += len(data)
        compress_seconds += t1 - t0
        decompress_seconds += t2 - t1
        roundtrip_ok = roundtrip_ok and restored == block
        if progress is not None:
            progress(i, len(blocks), len(raw), len(data))
    return AlgorithmResult(
        name="taichi",
        original_size=total_original,
        compressed_size=total_compressed,
        compress_seconds=compress_seconds,
        decompress_seconds=decompress_seconds,
        roundtrip_ok=roundtrip_ok,
        note=f"{len(blocks)} 个采样块逐块独立压缩",
    )


def evaluate_corpus(
    blocks: list[str],
    predictor: "object | None" = None,
    *,
    model_id: str | None = None,
    device: str | None = None,
    dtype: str | None = None,
    precision: int = DEFAULT_PRECISION,
    progress: Callable[[int, int, int, int], None] | None = None,
    include_taichi: bool = True,
) -> list[AlgorithmResult]:
    """在同一组采样块上运行全部算法（taichi 在前），保证公平对比。

    ``include_taichi=False`` 时只跑传统算法，不加载模型（快速预览模式）。
    """
    results: list[AlgorithmResult] = []
    if include_taichi:
        results.append(
            evaluate_corpus_taichi(
                blocks,
                predictor,
                model_id=model_id,
                device=device,
                dtype=dtype,
                precision=precision,
                progress=progress,
            )
        )
    for name, compress_fn, decompress_fn in _classic_codecs():
        results.append(evaluate_corpus_classic(blocks, name, compress_fn, decompress_fn))
    if _zstd_codec() is None:
        results.append(
            AlgorithmResult("zstd -19", 0, 0, None, None, None, note="zstandard 未安装")
        )
    return results


# —— 表格渲染 ——


def fmt_size(size: float) -> str:
    """人类可读的字节数。"""
    if size < 1024:
        return f"{size:.0f} B"
    if size < 1024**2:
        return f"{size / 1024:.1f} KB"
    return f"{size / 1024**2:.2f} MB"


def fmt_speed(bytes_per_second: float) -> str:
    """人类可读的吞吐速度。"""
    if bytes_per_second < 1024:
        return f"{bytes_per_second:.0f} B/s"
    if bytes_per_second < 1024**2:
        return f"{bytes_per_second / 1024:.0f} KB/s"
    return f"{bytes_per_second / 1024**2:.2f} MB/s"


def _display_width(text: str) -> int:
    """终端显示宽度：CJK 全角字符按 2 列计，用于表格对齐。"""
    return sum(2 if unicodedata.east_asian_width(ch) in "WF" else 1 for ch in text)


def _pad(text: str, width: int) -> str:
    """按显示宽度右侧补空格。"""
    return text + " " * max(0, width - _display_width(text))


def format_benchmark_table(results: list[AlgorithmResult]) -> str:
    """把基准结果渲染为等宽对齐的文本表格。"""
    headers = ["算法", "压缩后", "压缩比", "bpb", "压缩速度", "解压速度", "往返"]
    rows = []
    for r in results:
        if r.available:
            rows.append([
                r.name,
                fmt_size(r.compressed_size),
                f"{r.ratio:.2f}x",
                f"{r.bits_per_byte:.3f}",
                fmt_speed(r.compress_speed),
                fmt_speed(r.decompress_speed),
                "✓" if r.roundtrip_ok else "✗",
            ])
        else:
            rows.append([r.name, "—", "—", "—", "—", "—", "—"])
    table = [headers] + rows
    widths = [
        max(_display_width(row[col]) for row in table) for col in range(len(headers))
    ]
    lines = ["  ".join(_pad(h, w) for h, w in zip(headers, widths))]
    lines.append("  ".join("-" * w for w in widths))
    for row in rows:
        lines.append("  ".join(_pad(cell, w) for cell, w in zip(row, widths)))
    for r in results:  # 不可用原因等备注
        if r.note:
            lines.append(f"# {r.name}: {r.note}")
    return "\n".join(lines)
