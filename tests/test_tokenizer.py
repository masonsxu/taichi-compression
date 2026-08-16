"""tokenizer 模块自测（任务 2）：只需 transformers，不需要 torch。

默认用项目默认模型 Qwen2.5-0.5B 的 tokenizer（仅下载词表，约 11MB）；
可用环境变量 TAICHI_TEST_MODEL 换用其他模型。
"""

from __future__ import annotations

import os
import unittest

try:
    from taichi_compress.tokenizer import Tokenizer
    HAVE_TRANSFORMERS = True
except ImportError:  # pragma: no cover
    HAVE_TRANSFORMERS = False

MODEL_ID = os.environ.get("TAICHI_TEST_MODEL", "Qwen/Qwen2.5-0.5B")

TEXTS = [
    "",
    "Hello World",
    "太极压缩：预测为阴，编码为阳。",
    "Mixed 中英文 & symbols! #42\ttab\nnewline trailing  ",
    "🙂🚀 emoji 也是文本",
    "The quick brown fox jumps over the lazy dog. " * 3,
]


@unittest.skipUnless(HAVE_TRANSFORMERS, "需要 transformers")
class TestTokenizer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.tokenizer = Tokenizer(MODEL_ID)
        except Exception as exc:  # 网络不可用等情形下跳过而非报错
            raise unittest.SkipTest(f"tokenizer 加载失败: {exc}") from exc

    def test_roundtrip_exact(self):
        # 无损压缩的前提：encode → decode 必须严格还原任意文本
        for text in TEXTS:
            with self.subTest(text=text[:20]):
                ids = self.tokenizer.encode(text)
                self.assertEqual(self.tokenizer.decode(ids), text)

    def test_empty_text_encodes_to_empty(self):
        self.assertEqual(self.tokenizer.encode(""), [])

    def test_encode_deterministic(self):
        text = TEXTS[2]
        self.assertEqual(self.tokenizer.encode(text), self.tokenizer.encode(text))

    def test_token_ids_within_vocab(self):
        ids = self.tokenizer.encode("".join(TEXTS))
        self.assertTrue(ids)
        self.assertTrue(all(self.tokenizer.is_valid_token_id(t) for t in ids))

    def test_properties(self):
        self.assertGreater(self.tokenizer.vocab_size, 0)
        self.assertEqual(self.tokenizer.model_id, MODEL_ID)


if __name__ == "__main__":
    unittest.main()
