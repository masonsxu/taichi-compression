"""任务 3/4 自测：压缩 ↔ 解压往返（需真实模型）。

默认 Qwen2.5-0.5B / CPU；环境变量 TAICHI_TEST_MODEL / TAICHI_TEST_DEVICE 可覆盖。
覆盖：往返无损（含边界文本）、文件级统计、压缩有效性、容器头字段、
确定性、模型不匹配/损坏/截断报错、预测器交错复用、上下文上限守卫。
"""

from __future__ import annotations

import dataclasses
import os
import tempfile
import unittest
import zlib
from pathlib import Path

try:
    from taichi_compress.model import LLMPredictor, PredictorConfig
    HAVE_LLM = True
except ImportError:  # pragma: no cover
    HAVE_LLM = False

from taichi_compress.compressor import compress_file, compress_text, parse_header
from taichi_compress.decompressor import decompress_file, decompress_text

MODEL_ID = os.environ.get("TAICHI_TEST_MODEL", "Qwen/Qwen2.5-0.5B")
DEVICE = os.environ.get("TAICHI_TEST_DEVICE", "cpu")

TEXTS = {
    "hello": "Hello World",
    "empty": "",
    "english": "The quick brown fox jumps over the lazy dog. " * 3,
    "chinese": "太极者，阴阳之母也。预测为阴，编码为阳，阴阳相济，方成压缩之道。",
    "mixed": "TaiChi 太极 compression 压缩 🙂 entropy 熵\n\tmixed 混合 text 文本",
    "repetitive": "the same token token token token token token " * 4,
}


@unittest.skipUnless(HAVE_LLM, "需要 torch/transformers")
class TestRoundtrip(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.predictor = LLMPredictor(PredictorConfig(model_id=MODEL_ID, device=DEVICE))
        except Exception as exc:  # 网络不可用/权重下载失败时跳过而非报错
            raise unittest.SkipTest(f"模型加载失败: {exc}") from exc

    def roundtrip(self, text: str) -> bytes:
        data = compress_text(text, self.predictor)
        self.assertEqual(decompress_text(data, self.predictor), text)
        return data

    def test_roundtrip_all_texts(self):
        for name, text in TEXTS.items():
            with self.subTest(case=name):
                self.roundtrip(text)

    def test_file_roundtrip_and_stats(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "in.txt"
            out = Path(tmp) / "out.tc"
            restored = Path(tmp) / "restored.txt"
            text = TEXTS["english"] + TEXTS["chinese"]
            src.write_bytes(text.encode("utf-8"))

            stats = compress_file(src, out, self.predictor)
            dstats = decompress_file(out, restored, self.predictor)

            self.assertEqual(restored.read_bytes(), text.encode("utf-8"))
            self.assertEqual(stats.original_size, len(text.encode("utf-8")))
            self.assertEqual(stats.compressed_size, out.stat().st_size)
            self.assertEqual(stats.token_count, len(self.predictor.tokenizer.encode(text)))
            self.assertGreater(stats.ratio, 0)
            self.assertEqual(dstats.original_size, stats.original_size)
            self.assertEqual(dstats.token_count, stats.token_count)

    def test_compression_effective(self):
        # 高冗余文本：LLM 预测应显著优于原始字节（>2 倍压缩比）
        data = self.roundtrip(TEXTS["repetitive"])
        self.assertLess(len(data), len(TEXTS["repetitive"].encode()) * 0.5)
        # 一般英文文本也应小于原始体积
        data_en = self.roundtrip(TEXTS["english"])
        self.assertLess(len(data_en), len(TEXTS["english"].encode()))

    def test_header_fields(self):
        text = "Hello World"
        data = compress_text(text, self.predictor)
        header, body_start = parse_header(data)
        self.assertEqual(header.model_id, self.predictor.model_id)
        self.assertEqual(header.precision, 24)
        self.assertEqual(header.logit_scale, self.predictor.logit_scale)
        self.assertEqual(header.original_size, len(text.encode()))
        self.assertEqual(header.num_tokens, len(self.predictor.tokenizer.encode(text)))
        self.assertEqual(header.crc32, zlib.crc32(text.encode()))
        self.assertGreater(body_start, 0)
        self.assertTrue(data.startswith(b"TAICHI"))

    def test_output_deterministic(self):
        # 相同设备上两次压缩必须逐字节一致（跨设备由 logit 量化 + CRC 兜底）
        a = compress_text(TEXTS["hello"], self.predictor)
        b = compress_text(TEXTS["hello"], self.predictor)
        self.assertEqual(a, b)

    def test_decompress_self_constructs_predictor(self):
        # 不传 predictor：按容器头自动加载同一模型
        data = compress_text(TEXTS["hello"], self.predictor)
        self.assertEqual(decompress_text(data), TEXTS["hello"])

    def test_model_mismatch_raises(self):
        data = compress_text(TEXTS["hello"], self.predictor)
        header, body_start = parse_header(data)
        forged = dataclasses.replace(header, model_id="someone/else-model")
        with self.assertRaises(ValueError):
            decompress_text(forged.to_bytes() + data[body_start:], self.predictor)

    def test_corrupt_bitstream_raises(self):
        data = bytearray(compress_text(TEXTS["english"], self.predictor))
        body_start = parse_header(bytes(data))[1]
        # 破坏比特流中部（末尾字节可能是解码器不读取的冲刷冗余，破坏无效果）
        data[body_start + (len(data) - body_start) // 2] ^= 0xFF
        with self.assertRaises(ValueError):
            decompress_text(bytes(data), self.predictor)

    def test_truncated_raises(self):
        data = compress_text(TEXTS["english"], self.predictor)
        with self.assertRaises(ValueError):
            decompress_text(data[: len(data) // 2], self.predictor)

    def test_predictor_reuse_interleaved(self):
        # 同一预测器交错处理多个文件：内部 reset 保证状态隔离
        a, b = TEXTS["english"], TEXTS["chinese"]
        data_a = compress_text(a, self.predictor)
        data_b = compress_text(b, self.predictor)
        self.assertEqual(decompress_text(data_b, self.predictor), b)
        self.assertEqual(decompress_text(data_a, self.predictor), a)

    def test_context_limit_guard(self):
        with self.assertRaises(ValueError):
            compress_text("word " * 40000, self.predictor)  # 远超 32768 token 上限

    def test_invalid_utf8_input_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            bad = Path(tmp) / "bad.bin"
            bad.write_bytes(b"\xff\xfe invalid utf8 \x00")
            with self.assertRaises(ValueError):
                compress_file(bad, Path(tmp) / "out.tc", self.predictor)


@unittest.skipUnless(HAVE_LLM, "需要 torch/transformers")
class TestDtypeRoundtrip(unittest.TestCase):
    """权重精度（dtype）× 容器版本：float16 走 v2 容器，float32 保持 v1 兼容。"""

    @classmethod
    def setUpClass(cls):
        try:
            cls.fp16 = LLMPredictor(
                PredictorConfig(model_id=MODEL_ID, device=DEVICE, dtype="float16")
            )
            cls.fp32 = LLMPredictor(
                PredictorConfig(model_id=MODEL_ID, device=DEVICE, dtype="float32")
            )
        except Exception as exc:  # pragma: no cover - 网络不可用/权重下载失败
            raise unittest.SkipTest(f"模型加载失败: {exc}") from exc

    def test_float16_writes_v2_and_roundtrips(self):
        data = compress_text("Hello World 太极 fp16", self.fp16)
        header, _ = parse_header(data)
        self.assertEqual(header.version, 2)
        self.assertEqual(header.dtype, "float16")
        self.assertEqual(decompress_text(data, self.fp16), "Hello World 太极 fp16")

    def test_float32_keeps_v1_layout(self):
        # float32 输出与历史 v1 文件逐字节同构（dtype 字段不出现）
        data = compress_text("Hello World", self.fp32)
        header, _ = parse_header(data)
        self.assertEqual((header.version, header.dtype), (1, "float32"))
        self.assertEqual(decompress_text(data, self.fp32), "Hello World")

    def test_decompress_self_constructs_fp16_predictor(self):
        # 不传 predictor：按容器头 dtype 自动加载（解压侧契约）。
        # 设备须显式固定为压缩侧同一设备：fp16 仅保证同设备逐比特一致，
        # auto 设备选择在多设备机器上可能选到另一台设备（如 cpu 压缩、
        # mps 解压），那是 CRC 拒绝的场景而非 bug。
        data = compress_text("Hello World", self.fp16)
        self.assertEqual(decompress_text(data, device=self.fp16.device), "Hello World")

    def test_dtype_mismatch_raises(self):
        data = compress_text("Hello World", self.fp16)
        with self.assertRaises(ValueError):
            decompress_text(data, self.fp32)


if __name__ == "__main__":
    unittest.main()
