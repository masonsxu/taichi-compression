"""命令行入口（任务 5）。

用法::

    taichi-compress -c input.txt -o output.tc        # 压缩
    taichi-compress -d output.tc -o restored.txt     # 解压
    taichi-compress --benchmark input.txt            # 与 gzip/xz/zstd 对比
    taichi-compress --info output.tc                 # 查看容器头（不加载模型）

安装后（``uv sync``）即为 ``taichi-compress`` 可执行命令；也可用
``python -m taichi_compress`` 调用。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__
from .arithmetic import DEFAULT_PRECISION
from .benchmark import fmt_size, fmt_speed, format_benchmark_table, peak_rss_mib, run_benchmark
from .compressor import compress_file, parse_header, read_text_file
from .decompressor import decompress_file

__all__ = ["main", "build_parser"]


def build_parser() -> argparse.ArgumentParser:
    """构建命令行参数解析器。"""
    parser = argparse.ArgumentParser(
        prog="taichi-compress",
        description="太极压缩：LLM 概率预测 + 算术编码的无损文本压缩",
    )
    parser.add_argument(
        "-V", "--version", action="version", version=f"%(prog)s {__version__}"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "-c", "--compress", metavar="FILE", help="压缩文本文件（必须为 UTF-8）"
    )
    group.add_argument("-d", "--decompress", metavar="FILE", help="解压 .tc 文件")
    group.add_argument(
        "--benchmark", metavar="FILE", help="与 gzip/xz/zstd 对比压缩效果"
    )
    group.add_argument(
        "--info", metavar="FILE", help="查看 .tc 容器头信息（无需加载模型）"
    )
    parser.add_argument(
        "-o",
        "--output",
        metavar="FILE",
        help="输出路径（缺省：压缩加 .tc；解压去掉 .tc 后加 .out.txt）",
    )
    parser.add_argument(
        "--model",
        metavar="ID",
        default=None,
        help="HuggingFace 模型标识，默认 Qwen/Qwen2.5-0.5B（解压时以容器头为准）",
    )
    parser.add_argument(
        "--device", default=None, help="运行设备：cpu / cuda / mps（默认自动选择）"
    )
    parser.add_argument(
        "--quant",
        choices=["auto", "float32", "float16", "q8_0", "q4_k_m"],
        default="auto",
        help="权重精度/量化（默认 auto：mps/cuda 用 float16，cpu 用 float32；"
        "q8_0/q4_k_m 走 llama.cpp/GGUF 后端，需已转换的 .gguf 文件）；"
        "非 float32 文件须在与压缩侧相同的环境解压",
    )
    parser.add_argument(
        "--gguf",
        metavar="FILE",
        default=None,
        help="显式指定 GGUF 文件路径（配合 --quant q8_0/q4_k_m；"
        "缺省按 {模型名}-{量化}.gguf 约定搜索 $TAICHI_GGUF_DIR、./.gguf）",
    )
    parser.add_argument(
        "--precision",
        type=int,
        default=DEFAULT_PRECISION,
        help=f"CDF 量子精度，1~30 比特（默认 {DEFAULT_PRECISION}）",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """命令行入口；返回进程退出码（0 成功）。"""
    args = build_parser().parse_args(argv)
    try:
        if args.compress:
            return _cmd_compress(args)
        if args.decompress:
            return _cmd_decompress(args)
        if args.benchmark:
            return _cmd_benchmark(args)
        return _cmd_info(args)
    except (ValueError, OSError) as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 1


def _config_from_args(args: argparse.Namespace, model_id: str):
    """由命令行参数构建 PredictorConfig（--model/--device/--quant/--gguf）。"""
    from .model import PredictorConfig  # 延迟导入，--info 无需 torch

    return PredictorConfig(
        model_id=model_id,
        device=args.device,
        quant=None if args.quant == "auto" else args.quant,
        gguf_path=args.gguf,
    )


def _cmd_compress(args: argparse.Namespace) -> int:
    """压缩子命令。"""
    from .model import create_predictor  # 延迟导入，--info 无需 torch

    src = Path(args.compress)
    dst = Path(args.output) if args.output else src.with_name(src.name + ".tc")
    model_id = args.model or "Qwen/Qwen2.5-0.5B"
    print(f"加载模型 {model_id} ...")
    predictor = create_predictor(_config_from_args(args, model_id))
    print(
        f"设备: {predictor.device}，词表: {predictor.vocab_size}，"
        f"后端: {predictor.backend}，量化: {predictor.quant_name}"
    )
    stats = compress_file(src, dst, predictor, precision=args.precision)
    print(f"压缩完成: {src} ({fmt_size(stats.original_size)}) → {dst} ({fmt_size(stats.compressed_size)})")
    print(
        f"  压缩比 {stats.ratio:.2f}x | {stats.bits_per_byte:.3f} bpb"
        f" | {stats.token_count} tokens | {stats.duration_seconds:.2f}s"
        f" ({fmt_speed(stats.speed_bytes_per_second)})"
    )
    return 0


def _cmd_decompress(args: argparse.Namespace) -> int:
    """解压子命令（模型、量化与后端以容器头为准）。"""
    from .model import PredictorConfig, create_predictor  # 延迟导入

    src = Path(args.decompress)
    dst = Path(args.output) if args.output else _default_decompress_output(src)
    header, _ = parse_header(src.read_bytes())
    print(
        f"容器: 模型 {header.model_id}，{header.num_tokens} tokens，"
        f"精度 {header.precision} bit，{header.backend}/{header.quant}（v{header.version}）"
    )
    print(f"加载模型 {header.model_id} ...")
    predictor = create_predictor(
        PredictorConfig(
            model_id=header.model_id,
            device=args.device,
            logit_scale=header.logit_scale,
            quant=header.quant,
            backend=header.backend,
            gguf_path=args.gguf,
        )
    )
    print(f"设备: {predictor.device}")
    stats = decompress_file(src, dst, predictor)
    print(
        f"解压完成: {src} ({fmt_size(stats.compressed_size)}) → {dst} ({fmt_size(stats.original_size)})"
        f" | {stats.duration_seconds:.2f}s ({fmt_speed(stats.speed_bytes_per_second)})"
        " | CRC32 校验通过"
    )
    return 0


def _cmd_benchmark(args: argparse.Namespace) -> int:
    """基准对比子命令。"""
    from .model import create_predictor  # 延迟导入

    path = Path(args.benchmark)
    text = read_text_file(path)
    model_id = args.model or "Qwen/Qwen2.5-0.5B"
    print(f"加载模型 {model_id} ...")
    predictor = create_predictor(_config_from_args(args, model_id))
    print(
        f"输入: {path} ({fmt_size(len(text.encode('utf-8')))})，"
        f"设备: {predictor.device}，后端: {predictor.backend}，"
        f"量化: {predictor.quant_name}\n"
    )
    results = run_benchmark(text, predictor, precision=args.precision)
    print(format_benchmark_table(results))
    print(f"\n进程峰值内存: {peak_rss_mib():.0f} MiB（含模型权重；传统算法本身仅需常数级内存）")
    print("注：taichi 计时不含模型加载；往返列 = 压缩再解压后与原文逐字节一致")
    return 0


def _cmd_info(args: argparse.Namespace) -> int:
    """查看容器头子命令（不加载模型、不解压）。"""
    path = Path(args.info)
    header, _ = parse_header(path.read_bytes())
    print(f"文件        : {path}")
    print(f"格式        : TAICHI 容器 v{header.version}")
    print(f"模型        : {header.model_id}")
    print(f"后端/量化   : {header.backend} / {header.quant}")
    print(f"CDF 精度    : {header.precision} bit")
    print(f"logit 量化  : {header.logit_scale if header.logit_scale else '关闭'}")
    print(f"原始大小    : {fmt_size(header.original_size)}")
    print(f"token 数    : {header.num_tokens}")
    print(f"CRC32       : {header.crc32:08x}")
    return 0


def _default_decompress_output(src: Path) -> Path:
    """解压缺省输出：去掉 .tc 后缀再加 .out.txt（不覆盖原始文件）。"""
    base = src.name[:-3] if src.name.endswith(".tc") else src.name
    return src.with_name(base + ".out.txt")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
