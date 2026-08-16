"""enwik8 基准评测（任务 6）。

流程：下载（缓存）enwik8 → 全文等分段采样固定大小文本块 → 对同一组块
逐块独立压缩-解压（taichi / gzip -9 / xz -9 / zstd -19，同口径公平对比）
→ 汇总表格 + JSON 报告。

用法::

    uv run python scripts/benchmark_enwik8.py                  # 8 块 × 8192 字符
    uv run python scripts/benchmark_enwik8.py --classic-only   # 不加载模型快速预览
    uv run python scripts/benchmark_enwik8.py --blocks 4 --block-chars 4096

耗时估计：Qwen2.5-0.5B 在 MPS 上约 20 token/s，8192 字符 ≈ 2400 token/块，
每块压缩+解压约 4 分钟；默认 8 块全程约 30 分钟（不含一次性模型加载与 36MB 语料下载）。
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from taichi_compress.arithmetic import DEFAULT_PRECISION
from taichi_compress.benchmark import (
    evaluate_corpus,
    format_benchmark_table,
    peak_rss_mib,
    sample_blocks,
)
from taichi_compress.model import DEFAULT_MODEL_ID, LLMPredictor, PredictorConfig

ENWIK8_URL = "https://mattmahoney.net/dc/enwik8.zip"
CACHE_DIR = Path(".benchmarks")


def ensure_enwik8(cache_dir: Path = CACHE_DIR) -> Path:
    """确保本地存在 enwik8 文本（自动下载解压，约 36MB zip / 100MB 文本）。"""
    target = cache_dir / "enwik8"
    if target.exists():
        return target
    cache_dir.mkdir(parents=True, exist_ok=True)
    zip_path = cache_dir / "enwik8.zip"
    if not zip_path.exists():
        print(f"下载 {ENWIK8_URL}（约 36MB）...")
        urllib.request.urlretrieve(ENWIK8_URL, zip_path, reporthook=_download_progress)
        print()
    print("解压 enwik8 ...")
    with zipfile.ZipFile(zip_path) as archive:
        member = next(n for n in archive.namelist() if Path(n).name == "enwik8")
        with archive.open(member) as src, open(target, "wb") as dst:
            dst.write(src.read())
    return target


def _download_progress(count: int, block_size: int, total: int) -> None:
    """urlretrieve 进度钩子：单行百分比。"""
    if total > 0:
        percent = min(100, count * block_size * 100 // total)
        sys.stdout.write(f"\r  {percent}%")
        sys.stdout.flush()


def load_text(path: Path) -> tuple[str, int]:
    """读取语料文本；返回 (文本, 因非法 UTF-8 被忽略的字节数)。"""
    raw = path.read_bytes()
    try:
        return raw.decode("utf-8"), 0
    except UnicodeDecodeError:
        text = raw.decode("utf-8", errors="ignore")
        dropped = len(raw) - len(text.encode("utf-8"))
        return text, dropped


def _block_progress(done: int, total: int, original: int, compressed: int) -> None:
    """逐块进度输出。"""
    bpb = compressed * 8 / original if original else 0.0
    print(f"  [taichi {done}/{total}] {original} B → {compressed} B（{bpb:.3f} bpb）", flush=True)


def main(argv: list[str] | None = None) -> int:
    """脚本入口；返回进程退出码。"""
    parser = argparse.ArgumentParser(description="enwik8 基准评测：taichi vs gzip/xz/zstd")
    parser.add_argument("--blocks", type=int, default=8, help="采样块数（默认 8）")
    parser.add_argument("--block-chars", type=int, default=8192, help="每块字符数（默认 8192）")
    parser.add_argument("--model", default=DEFAULT_MODEL_ID, help="HuggingFace 模型标识")
    parser.add_argument("--device", default=None, help="cpu / cuda / mps（默认自动）")
    parser.add_argument(
        "--dtype",
        choices=["auto", "float32", "float16"],
        default="auto",
        help="权重精度（默认 auto：mps/cuda 用 float16，cpu 用 float32）",
    )
    parser.add_argument("--precision", type=int, default=DEFAULT_PRECISION, help="CDF 精度")
    parser.add_argument("--classic-only", action="store_true", help="只跑传统算法（不加载模型）")
    parser.add_argument(
        "--out", default=None, help="JSON 报告输出路径（默认 .benchmarks/enwik8_results.json）"
    )
    args = parser.parse_args(argv)

    corpus = ensure_enwik8()
    text, dropped = load_text(corpus)
    if dropped:
        print(f"警告: 语料含非法 UTF-8 字节，已忽略 {dropped} 字节（各算法用同一份清洗文本）")
    blocks = sample_blocks(text, args.blocks, args.block_chars)
    sample_bytes = sum(len(block.encode("utf-8")) for block in blocks)
    print(
        f"enwik8: {len(text):,} 字符；采样 {len(blocks)} 块 × {args.block_chars} 字符"
        f" ≈ {sample_bytes / 1024:.0f} KB（{sample_bytes / len(text) * 100:.2f}% 语料）"
    )

    device = None
    if not args.classic_only:
        print(f"加载模型 {args.model} ...")
        predictor = LLMPredictor(
            PredictorConfig(
                model_id=args.model,
                device=args.device,
                dtype=None if args.dtype == "auto" else args.dtype,
            )
        )
        device = predictor.device
        dtype_used = predictor.dtype_name
        print(f"设备: {device}，权重精度: {dtype_used}\n")
        results = evaluate_corpus(
            blocks, predictor, precision=args.precision, progress=_block_progress
        )
    else:
        print("（--classic-only：跳过 taichi）\n")
        dtype_used = None
        results = evaluate_corpus(blocks, include_taichi=False)

    print(format_benchmark_table(results))
    peak = peak_rss_mib()
    print(f"\n进程峰值内存: {peak:.0f} MiB（含模型权重；传统算法本身仅需常数级内存）")
    print(f"采样方式: 全文等分 {len(blocks)} 段各取开头 {args.block_chars} 字符，逐块独立压缩")

    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "corpus": "enwik8",
        "config": {
            "model": None if args.classic_only else args.model,
            "device": device,
            "dtype": dtype_used,
            "precision": args.precision,
            "num_blocks": len(blocks),
            "block_chars": args.block_chars,
            "sample_bytes": sample_bytes,
            "invalid_bytes_dropped": dropped,
        },
        "results": [
            {
                "name": r.name,
                "original_size": r.original_size,
                "compressed_size": r.compressed_size,
                "ratio": round(r.ratio, 4),
                "bits_per_byte": round(r.bits_per_byte, 4),
                "compress_seconds": None if r.compress_seconds is None else round(r.compress_seconds, 3),
                "decompress_seconds": None if r.decompress_seconds is None else round(r.decompress_seconds, 3),
                "compress_speed_bps": round(r.compress_speed, 1),
                "decompress_speed_bps": round(r.decompress_speed, 1),
                "roundtrip_ok": r.roundtrip_ok,
            }
            for r in results
        ],
        "peak_rss_mib": round(peak, 1),
    }
    out = Path(args.out) if args.out else CACHE_DIR / "enwik8_results.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"结果已写入 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
