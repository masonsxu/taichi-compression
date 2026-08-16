"""benchmark 模块自测：传统算法对比、表格渲染、语料级采样评测（任务 6）。"""

from __future__ import annotations

import os
import unittest

from taichi_compress.benchmark import (
    AlgorithmResult,
    _zstd_codec,
    benchmark_classic,
    evaluate_corpus,
    evaluate_corpus_classic,
    fmt_size,
    fmt_speed,
    format_benchmark_table,
    sample_blocks,
)

try:
    from taichi_compress.model import LLMPredictor, PredictorConfig
    HAVE_LLM = True
except ImportError:  # pragma: no cover
    HAVE_LLM = False

MODEL_ID = os.environ.get("TAICHI_TEST_MODEL", "Qwen/Qwen2.5-0.5B")
DEVICE = os.environ.get("TAICHI_TEST_DEVICE", "cpu")


class TestClassicBenchmark(unittest.TestCase):
    def test_classic_results_and_roundtrip(self):
        data = ("The quick brown fox jumps over the lazy dog. " * 20).encode()
        results = benchmark_classic(data)
        names = [r.name for r in results]
        self.assertIn("gzip -9", names)
        self.assertIn("xz -9", names)
        self.assertIn("zstd -19", names)
        for r in results:
            with self.subTest(name=r.name):
                if not r.available:  # zstd 缺失时优雅降级
                    self.assertIn("zstd", r.name)
                    continue
                self.assertGreater(r.compressed_size, 0)
                self.assertLess(r.compressed_size, len(data))
                self.assertGreater(r.compress_seconds, 0)
                self.assertGreater(r.decompress_seconds, 0)
                self.assertTrue(r.roundtrip_ok)
                self.assertGreater(r.ratio, 1.0)
                self.assertLess(r.bits_per_byte, 8.0)

    def test_zstd_codec_available(self):
        # zstandard 是项目依赖，正常环境必须可用
        self.assertIsNotNone(_zstd_codec())


class TestTableRendering(unittest.TestCase):
    def test_table_contains_rows_and_headers(self):
        results = [
            AlgorithmResult("taichi", 100, 20, 1.0, 0.5, True),
            AlgorithmResult("gzip -9", 100, 50, 0.01, 0.001, True),
            AlgorithmResult("zstd -19", 100, 0, None, None, None, note="zstandard 未安装"),
        ]
        table = format_benchmark_table(results)
        for needle in ("算法", "bpb", "taichi", "gzip -9", "✓", "—", "# zstd -19"):
            self.assertIn(needle, table)

    def test_fmt_helpers(self):
        self.assertEqual(fmt_size(0), "0 B")
        self.assertEqual(fmt_size(1536), "1.5 KB")
        self.assertIn("MB", fmt_size(3 * 1024 * 1024))
        self.assertEqual(fmt_speed(2048), "2 KB/s")
        self.assertIn("B/s", fmt_speed(10))


class TestSampleBlocks(unittest.TestCase):
    def test_uniform_segments(self):
        text = "".join(f"{i % 10}" for i in range(1000))  # 位置可辨的确定性文本
        blocks = sample_blocks(text, 5, 100)
        self.assertEqual(len(blocks), 5)
        self.assertTrue(all(len(b) == 100 for b in blocks))
        self.assertEqual(blocks[0], text[:100])
        self.assertEqual(blocks[4], text[800:900])

    def test_deterministic(self):
        text = "太极压缩" * 500
        self.assertEqual(sample_blocks(text, 4, 50), sample_blocks(text, 4, 50))

    def test_invalid_args_raise(self):
        for kwargs in ({"num_blocks": 0, "block_chars": 10},
                       {"num_blocks": 2, "block_chars": 0},
                       {"num_blocks": 10, "block_chars": 5}):  # 文本过短
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    sample_blocks("abcde", **kwargs)


class TestCorpusEvaluation(unittest.TestCase):
    """语料级评测：传统算法路径（不需要模型）。"""

    BLOCKS = [
        f"block {i}: " + "The quick brown fox jumps over the lazy dog. " * (3 + i)
        + "垫一些中文内容以保证多字节字符参与评测。"
        for i in range(4)
    ]

    def test_classic_corpus_same_footing(self):
        results = evaluate_corpus(self.BLOCKS, include_taichi=False)
        self.assertEqual([r.name for r in results][:2], ["gzip -9", "xz -9"])
        total = sum(len(b.encode("utf-8")) for b in self.BLOCKS)
        for r in results:
            with self.subTest(name=r.name):
                if not r.available:
                    continue
                self.assertEqual(r.original_size, total)
                self.assertGreater(r.compressed_size, 0)
                self.assertTrue(r.roundtrip_ok)
                self.assertIn("4 个采样块", r.note)

    def test_classic_corpus_direct(self):
        from taichi_compress.benchmark import _classic_codecs

        name, compress_fn, decompress_fn = _classic_codecs()[0]
        result = evaluate_corpus_classic(self.BLOCKS, name, compress_fn, decompress_fn)
        self.assertEqual(result.name, "gzip -9")
        self.assertTrue(result.roundtrip_ok)


@unittest.skipUnless(HAVE_LLM, "需要 torch/transformers")
class TestCorpusTaichi(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.predictor = LLMPredictor(PredictorConfig(model_id=MODEL_ID, device=DEVICE))
        except Exception as exc:  # 网络不可用/权重下载失败时跳过而非报错
            raise unittest.SkipTest(f"模型加载失败: {exc}") from exc

    def test_small_corpus_roundtrip_and_report(self):
        blocks = [
            "Hello taichi corpus block one. ",
            "第二块：中文采样评测，含标点。",
            "third block with a bit more text to compress. " * 2,
        ]
        progress_log: list[tuple[int, int]] = []
        results = evaluate_corpus(
            blocks, self.predictor, progress=lambda d, t, o, c: progress_log.append((d, t))
        )
        self.assertEqual(results[0].name, "taichi")
        self.assertTrue(results[0].roundtrip_ok)
        self.assertGreater(results[0].bits_per_byte, 0)
        self.assertLess(results[0].bits_per_byte, 8.0)
        self.assertEqual(progress_log, [(1, 3), (2, 3), (3, 3)])
        names = [r.name for r in results]
        self.assertIn("gzip -9", names)  # 全部算法同口径参与


if __name__ == "__main__":
    unittest.main()
