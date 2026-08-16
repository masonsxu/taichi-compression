"""LLM 预测器自测（任务 2）：需要 torch + transformers，首次运行会下载模型权重。

默认使用项目默认模型 Qwen/Qwen2.5-0.5B；可用环境变量换用更小的模型加速：

    TAICHI_TEST_MODEL=HuggingFaceTB/SmolLM2-135M uv run pytest tests/test_model.py
    TAICHI_TEST_DEVICE=mps uv run pytest tests/test_model.py   # 默认 cpu（确定性最佳）
"""

from __future__ import annotations

import os
import unittest

import numpy as np

try:
    from taichi_compress.model import LLMPredictor, PredictorConfig, quantized_softmax
    HAVE_LLM = True
except ImportError:  # pragma: no cover
    HAVE_LLM = False

from taichi_compress.tokenizer import Tokenizer

MODEL_ID = os.environ.get("TAICHI_TEST_MODEL", "Qwen/Qwen2.5-0.5B")
DEVICE = os.environ.get("TAICHI_TEST_DEVICE", "cpu")

TEXT = "The quick brown fox jumps over the lazy dog. 太极压缩：预测为阴，编码为阳。"


class TestQuantizedSoftmax(unittest.TestCase):
    """纯数值测试，不需要模型权重。"""

    def test_basic_properties(self):
        logits = np.array([2.0, 1.0, -0.5, 0.0])
        probs = quantized_softmax(logits, scale=1000)
        self.assertEqual(probs.shape, logits.shape)
        self.assertTrue((probs > 0).all())
        self.assertAlmostEqual(float(probs.sum()), 1.0, places=12)
        self.assertEqual(int(np.argmax(probs)), 0)

    def test_quantization_absorbs_tiny_differences(self):
        # 量化应把 1e-7 量级的浮点差异吸收进同一格点，输出逐比特一致
        a = np.array([2.0000001, 1.0, -0.5])
        b = np.array([2.0000004, 1.0, -0.5])
        self.assertTrue(np.array_equal(quantized_softmax(a, 1000), quantized_softmax(b, 1000)))
        self.assertFalse(np.array_equal(quantized_softmax(a, None), quantized_softmax(b, None)))

    def test_numerical_stability_for_extreme_logits(self):
        probs = quantized_softmax(np.array([1000.0, 999.0, -1000.0]), scale=1000)
        self.assertTrue(np.all(np.isfinite(probs)))
        self.assertAlmostEqual(float(probs.sum()), 1.0, places=12)

    def test_deterministic(self):
        logits = np.array([0.1, -2.3, 4.5, 6.7])
        self.assertTrue(np.array_equal(quantized_softmax(logits), quantized_softmax(logits)))

    def test_invalid_scale_raises(self):
        with self.assertRaises(ValueError):
            quantized_softmax(np.array([1.0]), scale=0)
        with self.assertRaises(ValueError):
            quantized_softmax(np.array([1.0]), scale=-1)


@unittest.skipUnless(HAVE_LLM, "需要 torch/transformers")
class TestLLMPredictor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.predictor = LLMPredictor(PredictorConfig(model_id=MODEL_ID, device=DEVICE))
            cls.tokenizer = Tokenizer(MODEL_ID)
        except Exception as exc:  # 网络不可用/权重下载失败时跳过而非报错
            raise unittest.SkipTest(f"模型加载失败: {exc}") from exc
        cls.tokens = cls.tokenizer.encode(TEXT)[:12]
        if len(cls.tokens) < 4:  # pragma: no cover
            raise unittest.SkipTest("样本文本 token 数不足")

    def setUp(self):
        self.predictor.reset()

    def test_distribution_shape_and_normalized(self):
        probs = self.predictor.predict_next_token_probabilities([])
        self.assertEqual(probs.shape, (self.predictor.vocab_size,))
        self.assertTrue(np.all(np.isfinite(probs)))
        self.assertTrue((probs > 0).all())
        self.assertAlmostEqual(float(probs.sum()), 1.0, places=9)

    def test_vocab_size_covers_tokenizer(self):
        # 模型 logits 维度必须覆盖 tokenizer 全部 token id（可能有 embedding 填充位）
        self.assertGreaterEqual(self.predictor.vocab_size, self.tokenizer.vocab_size)

    def test_predict_deterministic(self):
        # 同一 context：命中缓存、清空重算，三条路径必须逐比特一致
        a = self.predictor.predict_next_token_probabilities(self.tokens)
        b = self.predictor.predict_next_token_probabilities(self.tokens)  # 命中缓存
        self.assertTrue(np.array_equal(a, b))
        self.predictor.reset()
        c = self.predictor.predict_next_token_probabilities(self.tokens)  # 重算
        self.assertTrue(np.array_equal(a, c))

    def test_incremental_matches_full_context(self):
        """KV Cache 增量预测必须与整段重算一致（任务 2 核心正确性测试）。"""
        incremental = []
        for i in range(1, len(self.tokens) + 1):
            incremental.append(self.predictor.predict_next_token_probabilities(self.tokens[:i]))
        for i, p_inc in enumerate(incremental, start=1):
            with self.subTest(prefix_len=i):
                self.predictor.reset()
                p_full = self.predictor.predict_next_token_probabilities(self.tokens[:i])
                self.assertEqual(int(np.argmax(p_inc)), int(np.argmax(p_full)))
                # 分块方式不同会有 ~1 量子级浮点差，分布应几乎一致
                self.assertTrue(np.allclose(p_inc, p_full, atol=1e-3))

    def test_empty_context_then_growth_matches_direct(self):
        # "先 predict([]) 再 predict([t])" 与直接 predict([t]) 分块一致 → 逐比特相同
        p_direct = self.predictor.predict_next_token_probabilities(self.tokens[:1])
        self.predictor.reset()
        self.predictor.predict_next_token_probabilities([])
        p_incremental = self.predictor.predict_next_token_probabilities(self.tokens[:1])
        self.assertTrue(np.array_equal(p_direct, p_incremental))

    def test_prefix_mismatch_triggers_reset(self):
        # 上下文回退/分叉时自动重置缓存，结果与全新预测器一致
        self.predictor.predict_next_token_probabilities(self.tokens)
        shorter = self.tokens[:-2]
        p_short = self.predictor.predict_next_token_probabilities(shorter)
        self.predictor.reset()
        p_ref = self.predictor.predict_next_token_probabilities(shorter)
        self.assertTrue(np.array_equal(p_short, p_ref))

    def test_invalid_token_id_raises(self):
        with self.assertRaises(ValueError):
            self.predictor.predict_next_token_probabilities([self.predictor.vocab_size])
        with self.assertRaises(ValueError):
            self.predictor.predict_next_token_probabilities([-1])


if __name__ == "__main__":
    unittest.main()
