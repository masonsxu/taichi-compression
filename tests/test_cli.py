"""任务 5 自测：命令行接口。

基础行为（参数校验 / --info / 错误处理）不需要模型权重；
端到端测试默认 Qwen2.5-0.5B / CPU，可用 TAICHI_TEST_MODEL /
TAICHI_TEST_DEVICE 覆盖（模型加载通过 mock 复用，避免每条命令重复加载）。
"""

from __future__ import annotations

import contextlib
import io
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from taichi_compress.cli import main
from taichi_compress.compressor import ContainerHeader

try:
    from taichi_compress.model import LLMPredictor, PredictorConfig
    HAVE_LLM = True
except ImportError:  # pragma: no cover
    HAVE_LLM = False

MODEL_ID = os.environ.get("TAICHI_TEST_MODEL", "Qwen/Qwen2.5-0.5B")
DEVICE = os.environ.get("TAICHI_TEST_DEVICE", "cpu")


def run_cli(*argv: str) -> tuple[int, str, str]:
    """调用 CLI main，返回 (退出码, stdout, stderr)。"""
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = main(list(argv))
    return code, out.getvalue(), err.getvalue()


class TestCliBasics(unittest.TestCase):
    """不需要模型权重的 CLI 行为。"""

    def test_version_flag(self):
        buf = io.StringIO()
        with self.assertRaises(SystemExit) as ctx, contextlib.redirect_stdout(buf):
            main(["--version"])
        self.assertEqual(ctx.exception.code, 0)
        self.assertIn("taichi-compress", buf.getvalue())

    def test_mode_is_required_and_exclusive(self):
        for argv in ([], ["-c", "a", "-d", "b"], ["--info", "x", "--benchmark", "y"]):
            with self.subTest(argv=argv):
                with self.assertRaises(SystemExit) as ctx, \
                        contextlib.redirect_stderr(io.StringIO()):
                    main(argv)
                self.assertEqual(ctx.exception.code, 2)

    def test_info_prints_header_without_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "empty.tc"
            path.write_bytes(
                ContainerHeader(
                    precision=24,
                    logit_scale=1000,
                    original_size=0,
                    num_tokens=0,
                    model_id="Qwen/Qwen2.5-0.5B",
                    crc32=0,
                ).to_bytes()
            )
            code, out, err = run_cli("--info", str(path))
            self.assertEqual((code, err), (0, ""))
            self.assertIn("Qwen/Qwen2.5-0.5B", out)
            self.assertIn("24", out)
            self.assertIn("00000000", out)

    def test_info_prints_v2_dtype(self):
        # float16 容器（v2）应显示权重精度与版本号；float32 头保持 v1 布局
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "fp16.tc"
            path.write_bytes(
                ContainerHeader(
                    precision=24,
                    logit_scale=1000,
                    original_size=0,
                    num_tokens=0,
                    model_id="Qwen/Qwen2.5-0.5B",
                    crc32=0,
                    dtype="float16",
                ).to_bytes()
            )
            code, out, err = run_cli("--info", str(path))
            self.assertEqual((code, err), (0, ""))
            self.assertIn("float16", out)
            self.assertIn("v2", out)
            v1 = ContainerHeader(
                precision=24,
                logit_scale=1000,
                original_size=0,
                num_tokens=0,
                model_id="Qwen/Qwen2.5-0.5B",
                crc32=0,
                dtype="float32",
            ).to_bytes()
            self.assertEqual(v1[6], 1)  # float32 输出版本字节 = 1（历史兼容）

    def test_missing_input_returns_error(self):
        code, _, err = run_cli("-c", "/nonexistent/x.txt", "-o", "/tmp/should_not_exist.tc")
        self.assertEqual(code, 1)
        self.assertIn("错误", err)

    def test_invalid_utf8_returns_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            bad = Path(tmp) / "bad.bin"
            bad.write_bytes(b"\xff\xfe binary")
            code, _, err = run_cli("-c", str(bad), "-o", str(Path(tmp) / "o.tc"))
            self.assertEqual(code, 1)
            self.assertIn("UTF-8", err)

    def test_info_bad_magic_returns_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.tc"
            path.write_bytes(b"NOTTAICHI1234567890")
            code, _, err = run_cli("--info", str(path))
            self.assertEqual(code, 1)


@unittest.skipUnless(HAVE_LLM, "需要 torch/transformers")
class TestCliEndToEnd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.predictor = LLMPredictor(PredictorConfig(model_id=MODEL_ID, device=DEVICE))
        except Exception as exc:  # 网络不可用/权重下载失败时跳过而非报错
            raise unittest.SkipTest(f"模型加载失败: {exc}") from exc

    def setUp(self):
        # mock 掉模型构造（cli 内部延迟导入，patch 源头模块即可），
        # 复用类级 predictor，避免每条命令重复加载权重
        patcher = mock.patch(
            "taichi_compress.model.LLMPredictor", return_value=self.predictor
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_compress_then_decompress(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "in.txt"
            src.write_text("Hello World! 太极压缩 CLI 端到端测试。", encoding="utf-8")
            tc = Path(tmp) / "in.tc"
            out = Path(tmp) / "restored.txt"

            code, stdout, _ = run_cli("-c", str(src), "-o", str(tc))
            self.assertEqual(code, 0)
            self.assertTrue(tc.read_bytes().startswith(b"TAICHI"))
            self.assertIn("压缩比", stdout)

            code, stdout, _ = run_cli("-d", str(tc), "-o", str(out))
            self.assertEqual(code, 0)
            self.assertIn("CRC32 校验通过", stdout)
            self.assertEqual(out.read_bytes(), src.read_bytes())

    def test_default_output_names(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "a.txt"
            src.write_text("cli default output names", encoding="utf-8")
            code, _, _ = run_cli("-c", str(src))
            self.assertEqual(code, 0)
            self.assertTrue((Path(tmp) / "a.txt.tc").exists())

            code, _, _ = run_cli("-d", str(Path(tmp) / "a.txt.tc"))
            self.assertEqual(code, 0)
            restored = Path(tmp) / "a.txt.out.txt"
            self.assertTrue(restored.exists())
            self.assertEqual(restored.read_bytes(), src.read_bytes())

    def test_benchmark_smoke(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "bench.txt"
            src.write_text("benchmark smoke test 基准冒烟测试 text. " * 2, encoding="utf-8")
            code, out, _ = run_cli("--benchmark", str(src))
            self.assertEqual(code, 0)
            for needle in ("taichi", "gzip -9", "xz -9", "bpb", "✓"):
                self.assertIn(needle, out)


if __name__ == "__main__":
    unittest.main()
