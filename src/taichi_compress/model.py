"""LLM 预测器 —— 太极之"阴"。

给定 token 上文，返回下一个 token 的全词表概率分布，供算术编码器
（太极之"阳"）消费。压缩与解压必须以相同的模型、相同的配置、相同的调用
序列使用本模块，概率逐比特一致，无损还原才有保证。

三条工程底线：

1. **KV Cache 增量推理**：上下文每增长若干 token，只前向新增部分，
   单步 O(1) 而非整段重算 O(n)；
2. **数值确定性**：logit 量化（默认 ``round(x * 1000) / 1000``）+
   float64 softmax，把浮点差异吸收进量化网格。权重精度可配置：
   ``float32`` 时跨设备/跨内核差异仅 ~1e-6，量化网格完全吸收，
   **跨设备解压受支持**；``float16`` 时权重带宽减半（约 1.8x 提速、
   权重内存减半），但浮点差异超出网格吸收能力，仅保证**同设备**
   逐比特一致（对称调用序列 + 确定性内核），跨设备解压会被 CRC32
   拒绝而非解出错误数据。dtype 随容器头记录，解压侧据此加载；
3. **分布覆盖全词表且恒为正**：概率被 CDF 量化为 0 的 token 依然可编码
   （``arithmetic.build_cdf`` 为每个符号保底 1 个量子）。

上下文与调用序列契约：

- 空上下文时，预测器内部先喂入一个引导 token（优先
  ``tokenizer.bos_token_id``，回退 ``eos_token_id``，再回退 0）建立
  KV Cache；该 token 不进入符号流，对调用方不可见，但两侧对称发生；
- 引导 token 恒作为独立分块最先喂入，因此"整段一次预测"与"逐 token
  增长预测"在相同上下文边界处分块方式一致、logits 逐比特相同；
- 压缩与解压两侧必须以相同的上下文增长序列调用
  ``predict_next_token_probabilities``（通常都是逐 token 递增）。
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Sequence

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .tokenizer import Tokenizer

__all__ = [
    "DEFAULT_MODEL_ID",
    "PredictorConfig",
    "LLMPredictor",
    "quantized_softmax",
    "resolve_dtype",
]

DEFAULT_MODEL_ID = "Qwen/Qwen2.5-0.5B"

_DTYPE_NAMES = ("float32", "float16")


@dataclass(frozen=True)
class PredictorConfig:
    """LLM 预测器配置。

    Attributes:
        model_id: HuggingFace 模型标识，压缩与解压必须一致
        device: 运行设备；None 表示自动选择（cuda > mps > cpu）
        logit_scale: logit 量化精度（量化到 1/scale 的整数格点），
            None 表示不量化（不推荐用于压缩）
        prefill_chunk: 首次填充与长上下文重算时的分块 token 数，控制峰值内存
        dtype: 权重精度 "float32" / "float16"；None 表示按设备自动选择
            （cuda/mps 用 float16 换带宽与内存，cpu 用 float32 保兼容）
    """

    model_id: str = DEFAULT_MODEL_ID
    device: str | None = None
    logit_scale: int | None = 1000
    prefill_chunk: int = 1024
    dtype: str | None = None


def quantized_softmax(logits: np.ndarray, scale: int | None = 1000) -> np.ndarray:
    """logit 量化 + 数值稳定的 float64 softmax，返回全词表概率分布。

    量化把微小浮点差异（不同设备/内核的舍入误差）吸收进 1/scale 的格点，
    是跨设备解压一致性的第一道防线；softmax 用最大值平移保证数值稳定。
    纯 numpy 确定性运算：相同输入必得相同输出。

    Args:
        logits: 模型输出的 logits（任意形状，通常为 (vocab_size,)）
        scale: 量化精度；None 关闭量化

    Returns:
        与 logits 同形状的概率分布（float64，非负且和为 1）
    """
    values = np.asarray(logits, dtype=np.float64)
    if scale is not None:
        if scale <= 0:
            raise ValueError(f"logit_scale 必须为正整数或 None，当前为 {scale}")
        values = np.round(values * scale) / scale
    shifted = values - values.max()
    exp = np.exp(shifted)
    return exp / exp.sum()


def _auto_device() -> str:
    """自动选择设备：cuda > mps > cpu。"""
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_dtype(dtype: str | None, device: torch.device | str) -> str:
    """解析权重精度：None 时按设备自动选择（cuda/mps 用 float16，cpu 用 float32）。

    float16 只保证同设备逐比特一致（见模块 docstring），因此仅在有加速器时
    作为默认值；显式指定始终优先于自动选择。
    """
    if dtype is None:
        kind = torch.device(device).type
        return "float16" if kind in ("cuda", "mps") else "float32"
    if dtype not in _DTYPE_NAMES:
        raise ValueError(f"dtype 必须为 {'/'.join(_DTYPE_NAMES)} 或 None（自动），当前为 {dtype!r}")
    return dtype


class LLMPredictor:
    """带 KV Cache 的因果 LLM 下一 token 概率预测器。"""

    def __init__(self, config: PredictorConfig | None = None, **overrides: Any) -> None:
        """加载模型与 tokenizer。

        Args:
            config: 完整配置；None 使用默认值
            **overrides: 覆盖 config 中的个别字段（如 ``device="cpu"``）
        """
        if config is None:
            config = PredictorConfig()
        unknown = set(overrides) - set(config.__dataclass_fields__)
        if unknown:
            raise TypeError(f"未知的配置项: {sorted(unknown)}")
        cfg = replace(config, **overrides) if overrides else config

        # 跨设备一致性：禁用 TF32（Ampere+ GPU 上的低精度矩阵乘）
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

        self._config = cfg
        self._device = torch.device(cfg.device if cfg.device else _auto_device())
        self._dtype_name = resolve_dtype(cfg.dtype, self._device)
        torch_dtype = getattr(torch, self._dtype_name)
        self._tokenizer = AutoTokenizer.from_pretrained(cfg.model_id)
        try:
            model = AutoModelForCausalLM.from_pretrained(cfg.model_id, dtype=torch_dtype)
        except TypeError:  # 兼容旧版 transformers（参数名 torch_dtype）
            model = AutoModelForCausalLM.from_pretrained(cfg.model_id, torch_dtype=torch_dtype)
        self._model = model.to(self._device)
        if self._model.dtype != torch_dtype:  # 防止 dtype 参数被静默忽略
            self._model = self._model.to(torch_dtype)
        self._model.eval()

        # 引导 token：BOS → EOS → 0 逐级回退
        if self._tokenizer.bos_token_id is not None:
            self._prime_token_id = self._tokenizer.bos_token_id
        elif self._tokenizer.eos_token_id is not None:
            self._prime_token_id = self._tokenizer.eos_token_id
        else:
            self._prime_token_id = 0

        self._past: Any = None  # KV Cache（Cache 对象或 legacy tuple，原样回传）
        self._fed: list[int] = []  # 已进入 cache 的上下文 token（不含引导 token）
        self._last_logits: np.ndarray | None = None  # 最近一次前向的 float64 logits

    # —— 对外属性 ——

    @property
    def model_id(self) -> str:
        """HuggingFace 模型标识（写入压缩文件头，解压侧据此校验）。"""
        return self._config.model_id

    @property
    def device(self) -> str:
        """实际运行设备（"cpu" / "cuda" / "mps"）。"""
        return str(self._device)

    @property
    def vocab_size(self) -> int:
        """词表大小（模型 logits 维度；tokenizer 的 token id 均小于该值）。"""
        return int(self._model.config.vocab_size)

    @property
    def prime_token_id(self) -> int:
        """内部引导 token id（不进入符号流）。"""
        return self._prime_token_id

    @property
    def logit_scale(self) -> int | None:
        """logit 量化精度（None 表示不量化）；写入容器头供解压侧校验。"""
        return self._config.logit_scale

    @property
    def dtype_name(self) -> str:
        """权重精度（"float32" / "float16"）；写入容器头，解压侧据此加载。"""
        return self._dtype_name

    @property
    def max_context_tokens(self) -> int:
        """模型支持的最大上下文 token 数（含内部引导 token）；0 表示未知。"""
        limit = getattr(self._model.config, "max_position_embeddings", None)
        return int(limit) if limit else 0

    @property
    def tokenizer(self) -> Tokenizer:
        """配套的 tokenizer（与模型同源加载）。"""
        tok = getattr(self, "_tok_wrapper", None)
        if tok is None:
            tok = Tokenizer(self._config.model_id)
            self._tok_wrapper = tok
        return tok

    # —— 核心接口 ——

    def predict_next_token_probabilities(self, context: Sequence[int]) -> np.ndarray:
        """给定 token 上文，返回下一 token 的全词表概率分布。

        增量式：若 context 与已缓存前缀一致，只前向新增 token（KV Cache
        复用）；前缀不匹配或回退时自动重置重算；重复查询同一 context
        直接复用上次结果（配合 arithmetic.encode 的身份缓存零开销）。

        Args:
            context: 上文 token id 序列（可为空）

        Returns:
            形状 (vocab_size,) 的 float64 概率分布，非负且和为 1
        """
        context = list(context)
        vocab = self.vocab_size
        for i, token in enumerate(context):
            if not 0 <= token < vocab:
                raise ValueError(f"context[{i}] = {token} 超出词表范围 [0, {vocab})")
        fed = self._fed
        if len(context) < len(fed) or fed != context[: len(fed)]:
            self.reset()
        delta = context[len(self._fed) :]
        if delta or self._last_logits is None:
            self._feed(delta)
        return quantized_softmax(self._last_logits, self._config.logit_scale)

    def reset(self) -> None:
        """清空 KV Cache 与簿记状态，回到全新上下文。"""
        self._past = None
        self._fed = []
        self._last_logits = None

    # —— 内部实现 ——

    def _feed(self, new_tokens: Sequence[int]) -> None:
        """把新增 token 前向进模型并更新 KV Cache。

        引导 token 恒作为独立分块最先喂入（见模块 docstring 的一致性论证）。
        """
        chunks: list[list[int]] = []
        if self._past is None:
            chunks.append([self._prime_token_id])
        step = self._config.prefill_chunk
        for start in range(0, len(new_tokens), step):
            chunks.append(list(new_tokens[start : start + step]))
        with torch.inference_mode():
            out = None
            for chunk in chunks:
                ids = torch.tensor([chunk], dtype=torch.long, device=self._device)
                # 显式给出全序列 attention_mask（已缓存 + 本次新增），避免依赖
                # 各版本 transformers 的默认行为，提高跨版本一致性
                total = self._cache_len() + len(chunk)
                mask = torch.ones((1, total), dtype=torch.long, device=self._device)
                out = self._model(
                    input_ids=ids,
                    attention_mask=mask,
                    past_key_values=self._past,
                    use_cache=True,
                )
                self._past = out.past_key_values
            if out is None:  # pragma: no cover - 调用约定下不可能发生
                raise RuntimeError("_feed 未收到任何 token")
            self._last_logits = out.logits[0, -1].to(torch.float32).cpu().numpy().astype(np.float64)
        self._fed.extend(new_tokens)

    def _cache_len(self) -> int:
        """当前 KV Cache 中的序列长度（兼容 Cache 对象与 legacy tuple）。"""
        past = self._past
        if past is None:
            return 0
        get_seq_length = getattr(past, "get_seq_length", None)
        if callable(get_seq_length):
            return int(get_seq_length())
        return int(past[0][0].shape[-2])  # legacy tuple: (k, v) 中 k 的序列维
