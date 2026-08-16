"""任务 1 演示：算术编码的压缩效率 vs 信源熵理论下限。

运行：uv run python examples/demo_arithmetic.py
"""

from __future__ import annotations

import math
import random

from taichi_compress import decode, encode


def main() -> None:
    rng = random.Random(42)
    vocab = 64
    weights = [1.0 / (i + 1) for i in range(vocab)]  # Zipf 分布
    norm = sum(weights)
    probs = [w / norm for w in weights]
    symbols = rng.choices(range(vocab), weights=weights, k=4000)

    data = encode(probs, symbols)
    assert decode(data, probs) == symbols

    entropy = sum(p * math.log2(1.0 / p) for p in probs)
    bits = len(data) * 8
    print(f"符号数        : {len(symbols)}（词表 {vocab}，Zipf 分布）")
    print(f"信源熵        : {entropy:.3f} bit/符号（理论下界）")
    print(f"算术编码实际  : {bits / len(symbols):.3f} bit/符号（含文件头）")
    print("8 bit 定长    : 8.000 bit/符号")
    print("往返校验      : 通过")


if __name__ == "__main__":
    main()
