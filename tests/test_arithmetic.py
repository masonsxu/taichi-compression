"""任务 1 自测：算术编码核心（build_cdf / 流式编解码器 / 便捷 API / 压缩效率）。

覆盖：往返无损、边界情形（空序列/单符号/确定性分布/零概率符号/E3 下溢风暴）、
多种精度、随机性质测试、大词表、坏输入报错、编码效率逼近信源熵。
"""

from __future__ import annotations

import math
import random
import unittest

from taichi_compress.arithmetic import (
    DEFAULT_PRECISION,
    ArithmeticDecoder,
    ArithmeticEncoder,
    build_cdf,
    decode,
    encode,
)
from taichi_compress.utils import BitReader, BitWriter

try:
    import numpy as np
except ImportError:
    np = None


def random_dist(rng: random.Random, n: int) -> list[float]:
    """生成全为正概率的随机分布（自动归一化）。"""
    weights = [rng.random() + 1e-9 for _ in range(n)]
    total = sum(weights)
    return [w / total for w in weights]


class TestBuildCdf(unittest.TestCase):
    def test_basic_properties(self):
        rng = random.Random(0)
        cases = [(4, 5), (12, 40), (16, 256), (24, 64), (30, 3)]
        for precision, vocab in cases:
            with self.subTest(precision=precision, vocab=vocab):
                cdf = build_cdf(random_dist(rng, vocab), precision)
                self.assertEqual(cdf[0], 0)
                self.assertEqual(cdf[-1], 1 << precision)
                for lower, upper in zip(cdf, cdf[1:]):
                    self.assertGreater(upper, lower)  # 严格递增 ⇒ 每符号 >= 1 量子

    def test_unnormalized_weights_match_normalized(self):
        self.assertEqual(build_cdf([2.0, 3.0, 5.0], 16), build_cdf([0.2, 0.3, 0.5], 16))

    def test_normalization_robust_to_float_drift(self):
        # softmax 输出的概率和常带 1e-7 级浮点误差，归一化后不应报错
        self.assertEqual(build_cdf([0.9, 0.9], 16), build_cdf([0.5, 0.5], 16))

    def test_rounding_keeps_strict_monotonicity(self):
        cdf = build_cdf([1 / 3, 1 / 3, 1 / 3], 24)  # 循环小数，边界必须舍入
        self.assertEqual(cdf[0], 0)
        self.assertEqual(cdf[-1], 1 << 24)
        self.assertTrue(all(b > a for a, b in zip(cdf, cdf[1:])))

    def test_zero_probability_symbol_gets_minimum_quantum(self):
        # LLM 场景关键约束：量化后概率为 0 的 token 也必须可编码
        cdf = build_cdf([0.5, 0.0, 0.5], 24)
        self.assertEqual(cdf, [0, 1 << 23, (1 << 23) + 1, 1 << 24])

    def test_llm_like_extreme_tail(self):
        # 回归（任务 3 发现）：真实 LLM 分布长尾概率 < float64 累加精度，
        # cumsum 提前饱和到 1.0，旧实现会把 CDF 推过总量而报错
        probs = [1.0] + [1e-30] * 1000
        for precision in (16, 24):
            with self.subTest(precision=precision):
                cdf = build_cdf(probs, precision)
                total = 1 << precision
                self.assertEqual(cdf[0], 0)
                self.assertEqual(cdf[-1], total)  # 总额必须精确
                self.assertTrue(all(b > a for a, b in zip(cdf, cdf[1:])))
                self.assertTrue(all(0 <= b <= total for b in cdf))

    def test_extreme_tail_roundtrip(self):
        # 极端长尾分布下，尾部 token（概率 1e-30 量级）也必须可无损编解码
        probs = [1.0] + [1e-30] * 50
        symbols = [0, 1, 50, 25, 0]
        data = encode(probs, symbols)
        self.assertEqual(decode(data, probs), symbols)

    def test_invalid_inputs_raise(self):
        with self.assertRaises(ValueError):
            build_cdf([], 16)  # 空词表
        with self.assertRaises(ValueError):
            build_cdf([0.5, -0.1], 16)  # 负概率
        with self.assertRaises(ValueError):
            build_cdf([0.5, float("nan")], 16)
        with self.assertRaises(ValueError):
            build_cdf([0.5, float("inf")], 16)
        with self.assertRaises(ValueError):
            build_cdf([0.2] * 5, 2)  # 词表 5 > 量子总数 4
        with self.assertRaises(ValueError):
            build_cdf([0.0, 0.0], 16)  # 权重全为 0

    @unittest.skipUnless(np is not None, "需要 numpy")
    def test_numpy_path_matches_python_path(self):
        rng = random.Random(1)
        for vocab in (2, 10, 257):
            with self.subTest(vocab=vocab):
                probs = random_dist(rng, vocab)
                np_cdf = build_cdf(np.asarray(probs, dtype=np.float64), 20)
                self.assertIsInstance(np_cdf, np.ndarray)
                self.assertEqual(np_cdf.tolist(), build_cdf(probs, 20))


class TestStreamingRoundtrip(unittest.TestCase):
    """底层流式接口（ArithmeticEncoder/Decoder）的往返测试。"""

    def _roundtrip(self, symbols, dists, precision=DEFAULT_PRECISION):
        """dists[i] 为编码/解码第 i 个符号时使用的分布。返回 (解码结果, 输出比特数)。"""
        cdfs = [build_cdf(p, precision) for p in dists]
        writer = BitWriter()
        encoder = ArithmeticEncoder(writer, precision)
        for symbol, cdf in zip(symbols, cdfs):
            encoder.encode_symbol(symbol, cdf)
        encoder.finish()
        decoder = ArithmeticDecoder(BitReader(writer.to_bytes()), precision)
        decoded = [decoder.decode_symbol(cdf) for cdf in cdfs[: len(symbols)]]
        return decoded, writer.bits_written

    def test_uniform_static(self):
        rng = random.Random(2)
        vocab = 256
        symbols = [rng.randrange(vocab) for _ in range(1000)]
        dists = [[1.0 / vocab] * vocab] * len(symbols)
        decoded, _ = self._roundtrip(symbols, dists)
        self.assertEqual(symbols, decoded)

    def test_adaptive_skewed(self):
        # 分布随上一符号变化（自适应模型的模拟）：上一符号占一半概率
        rng = random.Random(5)
        vocab = 8
        symbols = [rng.randrange(vocab) for _ in range(800)]
        dists = []
        for i in range(len(symbols)):
            prev = symbols[i - 1] if i else 0
            dists.append([0.5 if s == prev else 0.5 / (vocab - 1) for s in range(vocab)])
        decoded, _ = self._roundtrip(symbols, dists)
        self.assertEqual(symbols, decoded)

    def test_e3_underflow_storm(self):
        # 均匀二值 + 交替符号会持续触发 E3 下溢记账，验证 pending 补发正确
        symbols = [i % 2 for i in range(2000)]
        dists = [[0.5, 0.5]] * len(symbols)
        decoded, _ = self._roundtrip(symbols, dists)
        self.assertEqual(symbols, decoded)

    def test_empty_sequence(self):
        decoded, _ = self._roundtrip([], [[0.25] * 4])
        self.assertEqual(decoded, [])

    def test_deterministic_distribution(self):
        symbols = [0] * 100
        decoded, bits = self._roundtrip(symbols, [[1.0]] * 100)
        self.assertEqual(symbols, decoded)
        self.assertLess(bits, 64)  # p=1 时每个符号几乎不产生输出

    def test_precision_range(self):
        rng = random.Random(9)
        for precision, vocab in [(1, 2), (8, 5), (16, 5), (24, 5), (30, 5)]:
            with self.subTest(precision=precision):
                dist = random_dist(rng, vocab)
                symbols = [rng.randrange(vocab) for _ in range(500)]
                decoded, _ = self._roundtrip(symbols, [dist] * len(symbols), precision)
                self.assertEqual(symbols, decoded)

    def test_invalid_symbol_and_cdf_raise(self):
        writer = BitWriter()
        with self.assertRaises(ValueError):
            ArithmeticEncoder(writer, 31)  # 精度超上限
        with self.assertRaises(ValueError):
            ArithmeticEncoder(writer, 0)  # 精度低于下限
        encoder = ArithmeticEncoder(writer, 16)
        cdf = build_cdf([0.5, 0.5], 16)
        with self.assertRaises(ValueError):
            encoder.encode_symbol(2, cdf)  # 符号越界
        with self.assertRaises(ValueError):
            encoder.encode_symbol(-1, cdf)
        with self.assertRaises(ValueError):
            encoder.encode_symbol(0, [0, 2, 3])  # cdf 总量错误
        with self.assertRaises(ValueError):
            encoder.encode_symbol(0, [0, 0, 4])  # 零宽度符号
        with self.assertRaises(RuntimeError):
            finished = ArithmeticEncoder(BitWriter(), 16)
            finished.finish()
            finished.encode_symbol(0, cdf)  # finish 后继续编码


class TestConvenienceAPI(unittest.TestCase):
    """encode()/decode() 便捷接口测试。"""

    def test_static_model_roundtrip(self):
        rng = random.Random(11)
        dist = random_dist(rng, 32)
        symbols = [rng.randrange(32) for _ in range(600)]
        data = encode(dist, symbols)
        self.assertEqual(symbols, decode(data, dist))

    def test_per_symbol_model_roundtrip(self):
        rng = random.Random(12)
        length = 400
        dists = [random_dist(rng, 10) for _ in range(length)]
        symbols = [rng.randrange(10) for _ in range(length)]
        data = encode(dists, symbols)
        self.assertEqual(symbols, decode(data, dists))

    def test_callable_model_roundtrip(self):
        rng = random.Random(13)
        vocab = 16
        dists = [random_dist(rng, vocab) for _ in range(300)]
        symbols = [rng.randrange(vocab) for _ in range(300)]

        def model(context):
            return dists[len(context)]

        data = encode(model, symbols)
        self.assertEqual(symbols, decode(data, model))

    def test_header_fields(self):
        data = encode([0.5, 0.5], [0, 1, 1, 0], precision=16)
        self.assertEqual(data[:4], b"TCAC")
        self.assertEqual(data[4], 1)  # 格式版本
        self.assertEqual(data[5], 16)  # CDF 精度
        self.assertEqual(data[6], 4)  # varint 编码的符号数

    def test_output_deterministic(self):
        rng = random.Random(14)
        dist = random_dist(rng, 8)
        symbols = [rng.randrange(8) for _ in range(200)]
        self.assertEqual(encode(dist, symbols), encode(dist, symbols))

    def test_empty_symbols(self):
        data = encode([0.5, 0.5], [])
        self.assertEqual(decode(data, [0.5, 0.5]), [])

    def test_bad_header_raises(self):
        with self.assertRaises(ValueError):
            decode(b"", [0.5, 0.5])  # 空数据
        with self.assertRaises(ValueError):
            decode(b"XXXX" + bytes([1, 24, 0]), [0.5, 0.5])  # magic 错误
        data = bytearray(encode([0.5, 0.5], [0, 1]))
        data[4] = 99  # 破坏版本号
        with self.assertRaises(ValueError):
            decode(bytes(data), [0.5, 0.5])

    def test_corrupt_body_does_not_hang(self):
        rng = random.Random(16)
        dist = random_dist(rng, 16)
        symbols = [rng.randrange(16) for _ in range(100)]
        data = bytearray(encode(dist, symbols))
        data[len(data) // 2] ^= 0xFF
        try:
            decode(bytes(data), dist)  # 允许解出乱码或抛 ValueError，但必须终止
        except ValueError:
            pass

    @unittest.skipUnless(np is not None, "需要 numpy")
    def test_numpy_distribution_roundtrip(self):
        rng = random.Random(15)
        probs = np.asarray(random_dist(rng, 128))
        symbols = [rng.randrange(128) for _ in range(500)]
        data = encode(probs, symbols)
        self.assertEqual(symbols, decode(data, probs))


class TestEfficiency(unittest.TestCase):
    """编码效率应逼近信源熵（算术编码的核心价值）。"""

    def test_close_to_entropy_dyadic(self):
        probs = [0.5, 0.25, 0.125, 0.0625, 0.03125, 0.03125]
        rng = random.Random(21)
        symbols = rng.choices(range(6), weights=probs, k=5000)
        data = encode(probs, symbols)
        self.assertEqual(symbols, decode(data, probs))
        entropy_bits = sum(p * math.log2(1.0 / p) for p in probs) * len(symbols)
        # 允许 2% + 256 bit 冗余（含文件头、字节对齐、采样波动）
        self.assertLessEqual(len(data) * 8, entropy_bits * 1.02 + 256)

    def test_uniform_vocabulary_matches_log2(self):
        vocab = 100
        rng = random.Random(22)
        symbols = [rng.randrange(vocab) for _ in range(3000)]
        data = encode([1.0 / vocab] * vocab, symbols)
        self.assertEqual(symbols, decode(data, [1.0 / vocab] * vocab))
        # 均匀分布下每个符号信息量恒为 log2(vocab)，无采样波动，可收紧上界
        entropy_bits = math.log2(vocab) * len(symbols)
        self.assertLessEqual(len(data) * 8, math.ceil(entropy_bits) + 128)


class TestRandomProperty(unittest.TestCase):
    """随机性质测试：任意词表/分布/长度组合均须无损往返。"""

    def test_many_random_roundtrips(self):
        rng = random.Random(1234)
        for trial in range(25):
            vocab = rng.randint(2, 40)
            length = rng.randint(0, 300)
            dists = [random_dist(rng, vocab) for _ in range(length)]
            symbols = [rng.randrange(vocab) for _ in range(length)]
            with self.subTest(trial=trial, vocab=vocab, length=length):
                data = encode(dists, symbols)
                self.assertEqual(symbols, decode(data, dists))


class TestLargeVocabulary(unittest.TestCase):
    """模拟 LLM 级词表（约 15 万 token）下的功能与开销。"""

    VOCAB = 150_000

    def test_roundtrip(self):
        rng = random.Random(7)
        probs = random_dist(rng, self.VOCAB)
        symbols = [rng.randrange(self.VOCAB) for _ in range(300)]
        data = encode(probs, symbols)  # 静态分布：CDF 按身份缓存，只构建一次
        self.assertEqual(symbols, decode(data, probs))
        # 均匀分布熵 ≈ log2(150000) ≈ 17.2 bit/符号，校验体积没有异常膨胀
        self.assertLess(len(data), 1200)

    @unittest.skipUnless(np is not None, "需要 numpy")
    def test_numpy_fast_path(self):
        rng = random.Random(8)
        probs = np.asarray(random_dist(rng, self.VOCAB))
        symbols = [rng.randrange(self.VOCAB) for _ in range(300)]
        data = encode(probs, symbols)
        self.assertEqual(symbols, decode(data, probs))


if __name__ == "__main__":
    unittest.main()
