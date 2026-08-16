"""任务 2 演示：LLM 下一 token 概率预测（太极之"阴"）。

运行：uv run python examples/demo_model.py [模型id]
"""

from __future__ import annotations

import sys

import numpy as np

from taichi_compress.model import LLMPredictor, PredictorConfig
from taichi_compress.tokenizer import Tokenizer


def main() -> None:
    model_id = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen2.5-0.5B"
    tokenizer = Tokenizer(model_id)
    predictor = LLMPredictor(PredictorConfig(model_id=model_id))

    text = "The capital of France is Paris. The capital of Japan is"
    ids = tokenizer.encode(text)
    print(f"模型: {model_id}")
    print(f"设备: {predictor.device}，词表: {predictor.vocab_size}")
    print(f"文本: {text!r} → {len(ids)} tokens\n")

    for step in range(min(8, len(ids))):
        probs = predictor.predict_next_token_probabilities(ids[:step])
        top5 = np.argsort(probs)[::-1][:5]
        entropy = float(-(probs * np.log2(probs + 1e-300)).sum())
        actual = tokenizer.decode([ids[step]]) if step < len(ids) else "?"
        print(f"step {step}: 实际下一 token = {actual!r}，分布熵 = {entropy:.2f} bit")
        print("  top-5 预测: " + ", ".join(
            f"{tokenizer.decode([int(t)])!r}:{probs[t]:.3f}" for t in top5
        ))
    print("\n分布熵越低 → 预测越准 → 该 token 消耗的编码比特越少（-log2 P）。")


if __name__ == "__main__":
    main()
