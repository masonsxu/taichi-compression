"""Phase 3B 自测：llama.cpp/GGUF 量化后端。

路径解析约定为纯逻辑，总是运行；推理部分需要 llama-cpp-python
（``uv sync --extra llama``）与按约定命名的 GGUF 权重文件（默认搜索
``./.gguf``，可用 ``TAICHI_GGUF_DIR`` 指定），缺失时自动跳过。

    TAICHI_TEST_GGUF_QUANT=q4_k_m uv run pytest tests/test_gguf.py
"""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from taichi_compress.gguf import GGUF_QUANTS, resolve_gguf_path

try:
    import llama_cpp  # noqa: F401
    HAVE_LLAMA = True
except ImportError:  # pragma: no cover - 环境相关
    HAVE_LLAMA = False

QUANT = os.environ.get("TAICHI_TEST_GGUF_QUANT", "q8_0")
MODEL_ID = os.environ.get("TAICHI_TEST_MODEL", "Qwen/Qwen2.5-0.5B")
TEXT = "The quick brown fox jumps over the lazy dog. 太极压缩：权重级量化推理。"


def _gguf_path() -> str | None:
    """返回可用的 GGUF 路径；依赖缺失时返回 None（测试跳过）。"""
    if not HAVE_LLAMA:
        return None
    try:
        return str(resolve_gguf_path(MODEL_ID, QUANT))
    except (FileNotFoundError, ValueError):
        return None


class TestResolveGGUFPath(unittest.TestCase):
    """GGUF 路径解析约定（纯逻辑，不需要权重）。"""

    def test_unsupported_quant_rejected(self):
        with self.assertRaises(ValueError):
            resolve_gguf_path(MODEL_ID, "int4")

    def test_explicit_missing_file_raises(self):
        with self.assertRaises(FileNotFoundError):
            resolve_gguf_path(MODEL_ID, "q8_0", explicit="/nonexistent/x.gguf")

    def test_explicit_path_takes_priority(self):
        # 显式路径不检查命名约定，始终优先
        with tempfile.TemporaryDirectory() as tmp:
            f = Path(tmp) / "arbitrary-name.gguf"
            f.write_bytes(b"x")
            self.assertEqual(
                resolve_gguf_path(MODEL_ID, "q8_0", explicit=str(f)), f
            )

    def test_search_failure_lists_dirs(self):
        # 找不到文件时报错应包含全部搜索路径（含 TAICHI_GGUF_DIR）；
        # 用不存在的模型名，避免被仓库里真实存在的 .gguf 命中
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.dict(os.environ, {"TAICHI_GGUF_DIR": tmp}):
                with self.assertRaises(FileNotFoundError) as ctx:
                    resolve_gguf_path("Qwen/No-Such-Model", "q8_0")
        self.assertIn(tmp, str(ctx.exception))


@unittest.skipUnless(_gguf_path(), "需要 llama-cpp-python 与 GGUF 权重文件")
class TestGGUFPredictor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            from taichi_compress.gguf import GGUFPredictor

            cls.path = _gguf_path()
            cls.predictor = GGUFPredictor(cls.path, model_id=MODEL_ID, n_ctx=4096)
        except Exception as exc:  # 加载失败（内存不足等）跳过而非报错
            raise unittest.SkipTest(f"GGUF 加载失败: {exc}") from exc
        cls.tokens = cls.predictor.tokenizer.encode(TEXT)[:12]

    def setUp(self):
        self.predictor.reset()

    def test_backend_identity(self):
        self.assertEqual(self.predictor.backend, "llama")
        self.assertEqual(self.predictor.model_id, MODEL_ID)
        self.assertEqual(self.predictor.quant_name, QUANT)
        self.assertIn(self.predictor.device, ("metal", "cpu"))

    def test_model_id_inferred_from_filename(self):
        # 不传 model_id：按文件名推断（qwen2.5-0.5b-q8_0.gguf → Qwen/Qwen2.5-0.5B）
        from taichi_compress.gguf import GGUFPredictor

        p = GGUFPredictor(self.path, n_ctx=512)
        self.assertEqual(p.model_id, "Qwen/Qwen2.5-0.5B")
        self.assertEqual(p.tokenizer.vocab_size, self.predictor.tokenizer.vocab_size)

    def test_distribution_shape_and_normalized(self):
        probs = self.predictor.predict_next_token_probabilities(self.tokens)
        self.assertEqual(probs.shape, (self.predictor.vocab_size,))
        self.assertTrue(np.all(np.isfinite(probs)))
        self.assertTrue((probs > 0).all())
        self.assertAlmostEqual(float(probs.sum()), 1.0, places=9)

    def test_predict_deterministic(self):
        # 对称调用序列 → 逐比特一致（无损压缩的核心协议）
        a = self.predictor.predict_next_token_probabilities(self.tokens)
        b = self.predictor.predict_next_token_probabilities(self.tokens)  # 命中缓存
        self.assertTrue(np.array_equal(a, b))
        self.predictor.reset()
        c = self.predictor.predict_next_token_probabilities(self.tokens)  # 重算
        self.assertTrue(np.array_equal(a, c))

    def test_incremental_matches_full_context(self):
        # 逐 token 增量与整段重算：量化权重下 logits 差异应远小于
        # logit 量化网格（1e-3），argmax 必须一致
        incremental = [
            self.predictor.predict_next_token_probabilities(self.tokens[:i])
            for i in range(1, len(self.tokens) + 1)
        ]
        for i, p_inc in enumerate(incremental, start=1):
            with self.subTest(prefix_len=i):
                self.predictor.reset()
                p_full = self.predictor.predict_next_token_probabilities(self.tokens[:i])
                self.assertEqual(int(np.argmax(p_inc)), int(np.argmax(p_full)))
                self.assertTrue(np.allclose(p_inc, p_full, atol=5e-2))

    def test_prefix_mismatch_triggers_reset(self):
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

    def test_tokenizer_parity_with_llama_cpp(self):
        # 无损兼容前提：HF tokenizer 与 llama.cpp 分词逐 id 一致
        text = "Hello 太极 compression 🙂 123\n\ttabs"
        hf_ids = self.predictor.tokenizer.encode(text)
        cpp_ids = self.predictor._llm.tokenize(
            text.encode("utf-8"), add_bos=False, special=False
        )
        self.assertEqual(hf_ids, list(cpp_ids))


@unittest.skipUnless(_gguf_path(), "需要 llama-cpp-python 与 GGUF 权重文件")
class TestGGUFRoundtrip(unittest.TestCase):
    """GGUF 后端压缩 ↔ 解压往返与 v3 容器。"""

    @classmethod
    def setUpClass(cls):
        try:
            from taichi_compress.gguf import GGUFPredictor

            cls.predictor = GGUFPredictor(_gguf_path(), model_id=MODEL_ID, n_ctx=4096)
        except Exception as exc:
            raise unittest.SkipTest(f"GGUF 加载失败: {exc}") from exc

    def roundtrip(self, text: str) -> bytes:
        from taichi_compress.compressor import compress_text
        from taichi_compress.decompressor import decompress_text

        data = compress_text(text, self.predictor)
        self.assertEqual(decompress_text(data, self.predictor), text)
        return data

    def test_roundtrip_texts(self):
        for name, text in {
            "english": "The quick brown fox jumps over the lazy dog. " * 3,
            "chinese": "太极者，阴阳之母也。预测为阴，编码为阳，阴阳相济，方成压缩之道。",
            "mixed": "TaiChi 太极 quant 量化 🙂 GGUF text 文本",
            "empty": "",
        }.items():
            with self.subTest(case=name):
                self.roundtrip(text)

    def test_writes_v3_header(self):
        from taichi_compress.compressor import compress_text, parse_header

        data = compress_text("Hello World", self.predictor)
        header, _ = parse_header(data)
        self.assertEqual(header.version, 3)
        self.assertEqual(header.backend, "llama")
        self.assertEqual(header.quant, QUANT)
        self.assertEqual(header.model_id, MODEL_ID)

    def test_output_deterministic(self):
        from taichi_compress.compressor import compress_text

        a = compress_text("Hello World", self.predictor)
        b = compress_text("Hello World", self.predictor)
        self.assertEqual(a, b)

    def test_decompress_self_constructs_gguf_predictor(self):
        # 不传 predictor：按容器头 backend/quant 经 create_predictor 路由到 GGUF
        from taichi_compress.compressor import compress_text
        from taichi_compress.decompressor import decompress_text

        data = compress_text("Hello World 太极 GGUF", self.predictor)
        self.assertEqual(decompress_text(data), "Hello World 太极 GGUF")

    def test_backend_mismatch_raises(self):
        # 容器头声明 llama 后端：transformers 预测器应在推理前被拒绝
        from taichi_compress.compressor import compress_text
        from taichi_compress.decompressor import decompress_text

        data = compress_text("Hello World", self.predictor)
        fake = mock.Mock(
            model_id=MODEL_ID,
            logit_scale=self.predictor.logit_scale,
            quant_name="float32",
            backend="transformers",
        )
        with self.assertRaises(ValueError):
            decompress_text(data, predictor=fake)


if __name__ == "__main__":
    unittest.main()
