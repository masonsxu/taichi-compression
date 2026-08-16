"""llama.cpp / GGUF 后端 —— 权重级量化推理（Phase 3B）。

通过 llama-cpp-python（macOS 预编译 wheel 自带 Metal）做进程内推理：
单 token 增量 eval + 全词表 logits，duck-type :class:`LLMPredictor` 接口，
压缩/解压主流程零改动复用。评估实测（Apple M4 / llama.cpp 10450）：

- 速度：Q8_0 原生单流 ~119 tok/s，完整压缩循环 ~90 tok/s（ transformers
  fp16 同循环 ~33 tok/s）；Q4_K_M 循环 ~100 tok/s
- 内存：Q8_0 权重文件 506 MB，n_ctx 32768 进程峰值 ~1.9 GB（Q4_K_M ~1.8 GB）
- bpb：Q8_0 +0.003 / Q4_K_M +0.011（enwik8 实测，对比 transformers float32 基线）
- tokenizer：llama.cpp 与 HF 逐 id 一致（无损兼容前提，已验证）

与无损直接相关的两条协议（对称性论述见 ``model.py`` 模块 docstring）：

1. **对称预热**：llama.cpp 加载后首次 eval 走计算图构建路径，logits 与
   稳定态差 ~1e-3，恰超 logit 量化网格的吸收边界；本类加载后先执行一次
   dummy eval 并丢弃结果，压缩与解压两侧对称发生，此后逐比特确定；
2. **同环境解压**：量化模型的跨 llama.cpp 构建一致性未验证，容器头记录
   backend 与量化类型，解压侧必须使用同一 GGUF 文件与同一构建，
   不一致时 CRC32 拒绝而非解出错误数据。

GGUF 文件解析约定：``{TAICHI_GGUF_DIR:-?}/{模型名小写}-{quant}.gguf``，
搜索顺序为 ``$TAICHI_GGUF_DIR`` → ``./.gguf`` → ``~/.cache/taichi``；
显式路径始终优先。base 模型无官方 GGUF，需用 llama.cpp 的
``convert_hf_to_gguf.py`` 自行转换（转换配方见 README）。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Sequence

import numpy as np

from .tokenizer import Tokenizer

__all__ = ["GGUF_QUANTS", "resolve_gguf_path", "GGUFPredictor"]

GGUF_QUANTS = ("q8_0", "q4_k_m")

# GGUF 元数据 general.file_type（LLAMA_FTYPE 枚举）→ 量化名；
# 仅收录经转换管线验证过的组合，未知值显式报错而非冒险解压
_FILE_TYPE_TO_QUANT = {1: "float16", 7: "q8_0", 15: "q4_k_m"}


def resolve_gguf_path(model_id: str, quant: str, explicit: str | None = None) -> Path:
    """按约定定位 GGUF 文件；找不到时给出包含搜索路径的报错。"""
    if explicit:
        path = Path(explicit).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"GGUF 文件不存在: {path}")
        return path
    if quant not in GGUF_QUANTS:
        raise ValueError(f"量化类型 {quant!r} 不在 GGUF 后端支持范围 {GGUF_QUANTS}")
    name = f"{model_id.split('/')[-1].lower()}-{quant}.gguf"
    dirs = []
    env = os.environ.get("TAICHI_GGUF_DIR")
    if env:
        dirs.append(Path(env).expanduser())
    dirs += [Path.cwd() / ".gguf", Path.home() / ".cache" / "taichi"]
    for d in dirs:
        candidate = d / name
        if candidate.is_file():
            return candidate
    searched = "、".join(str(d) for d in dirs)
    raise FileNotFoundError(
        f"未找到 {name}；已搜索 {searched}。"
        "可用 TAICHI_GGUF_DIR 指定目录，或显式传入 GGUF 路径"
    )


class GGUFPredictor:
    """进程内 llama.cpp 因果 LLM 下一 token 概率预测器（接口同 LLMPredictor）。"""

    def __init__(
        self,
        gguf_path: str | Path,
        model_id: str | None = None,
        *,
        logit_scale: int | None = 1000,
        n_ctx: int = 32768,
        n_gpu_layers: int = -1,
    ) -> None:
        """加载 GGUF 模型并完成对称预热。

        Args:
            gguf_path: GGUF 文件路径
            model_id: HF 模型标识（tokenizer 来源与容器头身份记录）；
                None 时从文件名推断（如 qwen2.5-0.5b-q8_0 → Qwen/Qwen2.5-0.5B）
            logit_scale: logit 量化精度，语义同 LLMPredictor
            n_ctx: 上下文长度（KV Cache 容量）
            n_gpu_layers: GPU 卸载层数；-1 全卸载（无 GPU 时自动回退 CPU）
        """
        try:
            import llama_cpp
            from llama_cpp import Llama
        except ImportError as exc:  # pragma: no cover - 环境相关
            raise ImportError(
                "GGUF 后端需要 llama-cpp-python（uv sync --extra llama）"
            ) from exc

        self._gguf_path = Path(gguf_path)
        self._logit_scale = logit_scale
        self._backend_lib = llama_cpp

        model_name = self._gguf_path.name
        if model_id is None:
            # qwen2.5-0.5b-q8_0.gguf → Qwen/Qwen2.5-0.5B：剥掉 .gguf 与量化
            # 后缀，首字母大写，规格尾缀 b → B（Qwen 命名惯例）
            stem = model_name[: -len(".gguf")] if model_name.endswith(".gguf") else model_name
            for suffix in GGUF_QUANTS:
                if stem.endswith(f"-{suffix}"):
                    stem = stem[: -(len(suffix) + 1)]
                    break
            stem = stem[:1].upper() + stem[1:]
            if stem.endswith("b"):
                stem = stem[:-1] + "B"
            model_id = f"Qwen/{stem}" if stem.lower().startswith("qwen") else stem
        self._model_id = model_id
        self._tokenizer = Tokenizer(model_id)

        self._llm = Llama(
            model_path=str(self._gguf_path),
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            verbose=False,
        )
        self._n_ctx = n_ctx
        self._vocab = int(self._llm.n_vocab())

        file_type = self._llm.metadata.get("general.file_type")
        try:
            quant = _FILE_TYPE_TO_QUANT[int(file_type)]  # type: ignore[arg-type]
        except (TypeError, KeyError, ValueError) as exc:
            raise ValueError(
                f"{self._gguf_path.name} 的 general.file_type={file_type!r} "
                f"不在已验证的量化组合 {sorted(_FILE_TYPE_TO_QUANT)} 内，拒绝使用"
            ) from exc
        self._quant = quant

        # 引导 token：与 LLMPredictor 相同的 BOS → EOS → 0 回退链
        tok = self._tokenizer
        if tok.bos_token_id is not None:
            self._prime_token_id = tok.bos_token_id
        elif tok.eos_token_id is not None:
            self._prime_token_id = tok.eos_token_id
        else:
            self._prime_token_id = 0

        # 对称预热：首次 eval 触发计算图构建（结果与稳定态差 ~1e-3），
        # 丢弃其结果；压缩与解压两侧都经过本构造函数，协议天然对称
        self.reset()
        self._eval([self._prime_token_id])
        self.reset()

    # —— 对外属性（与 LLMPredictor 对齐） ——

    @property
    def model_id(self) -> str:
        """HF 模型标识（写入压缩文件头，tokenizer 与谱系身份）。"""
        return self._model_id

    @property
    def backend(self) -> str:
        """推理后端标识（"llama"）。"""
        return "llama"

    @property
    def device(self) -> str:
        """实际运行设备：GPU 卸载可用时 "metal"，否则 "cpu"。"""
        return "metal" if self._backend_lib.llama_supports_gpu_offload() else "cpu"

    @property
    def vocab_size(self) -> int:
        """词表大小（模型 logits 维度）。"""
        return self._vocab

    @property
    def prime_token_id(self) -> int:
        """内部引导 token id（不进入符号流）。"""
        return self._prime_token_id

    @property
    def logit_scale(self) -> int | None:
        """logit 量化精度（None 表示不量化）。"""
        return self._logit_scale

    @property
    def quant_name(self) -> str:
        """权重量化类型（"q8_0" / "q4_k_m" / "float16"），写入容器头。"""
        return self._quant

    @property
    def max_context_tokens(self) -> int:
        """上下文 token 上限（构造时的 n_ctx）。"""
        return self._n_ctx

    @property
    def tokenizer(self) -> Tokenizer:
        """配套 tokenizer（HF 同源；与 llama.cpp 分词逐 id 一致已验证）。"""
        return self._tokenizer

    # —— 核心接口 ——

    def predict_next_token_probabilities(self, context: Sequence[int]) -> np.ndarray:
        """给定 token 上文，返回下一 token 的全词表概率分布（增量式）。

        语义与 ``LLMPredictor.predict_next_token_probabilities`` 完全一致：
        前缀复用则只前向新增 token，回退/分叉自动重置重算。
        """
        from .model import quantized_softmax

        context = list(context)
        vocab = self._vocab
        for i, token in enumerate(context):
            if not 0 <= token < vocab:
                raise ValueError(f"context[{i}] = {token} 超出词表范围 [0, {vocab})")
        if len(context) < len(self._fed) or self._fed != context[: len(self._fed)]:
            self.reset()
        delta = context[len(self._fed) :]
        if delta or self._last_logits is None:
            chunks: list[list[int]] = []
            if self._last_logits is None:
                chunks.append([self._prime_token_id])  # 引导 token 独立最先喂入
            chunks.append(delta)
            for chunk in chunks:
                if chunk:
                    self._last_logits = self._eval(chunk)
        self._fed = context
        return quantized_softmax(self._last_logits, self._logit_scale)

    def reset(self) -> None:
        """清空 KV Cache 与簿记状态（计算图/预热状态不受影响）。"""
        self._llm.reset()
        self._fed: list[int] = []
        self._last_logits: np.ndarray | None = None

    # —— 内部实现 ——

    def _eval(self, tokens: Sequence[int]) -> np.ndarray:
        """前向一批 token（增量 KV），返回末位全词表 float32 logits 副本。

        llama-cpp-python 0.3.x 的 ``Llama.eval`` 为稳定可用的批量入口；
        logits 经 ``llama_get_logits`` 读取（仅末位有效）。
        """
        self._llm.eval(list(tokens))
        ptr = self._backend_lib.llama_get_logits(self._llm._ctx.ctx)
        logits: np.ndarray = np.ctypeslib.as_array(ptr, shape=(self._vocab,)).copy()
        if logits.size != self._vocab:  # pragma: no cover - 防御性断言
            raise RuntimeError(f"llama_get_logits 返回 {logits.size} 个值，预期 {self._vocab}")
        return logits
