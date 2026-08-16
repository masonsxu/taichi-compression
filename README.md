# 太极压缩 (TaiChi Compression)

利用大语言模型（LLM）作为概率预测器、结合算术编码（Arithmetic Coding）的**无损文本压缩工具**，目标是实现远超 gzip / xz / zstd 的压缩率。

> 太极生两仪：预测为"阴"，编码为"阳"——以概率消除不确定性，以编码兑现信息熵。

## 原理

压缩的本质是**利用概率消除不确定性**：信息量 = -log₂(P)，事件概率越高，编码所需比特越少。

| 步骤 | 名称 | 做什么 |
|------|------|--------|
| 第一步 | 预测（太极之"阴"） | LLM 根据上文预测下一个 token 的概率分布 |
| 第二步 | 编码（太极之"阳"） | 算术编码根据概率分布将 token 编码为比特流 |

- 压缩：`原文 → Tokenizer → LLM 预测概率 → 算术编码 → 压缩文件`
- 解压：`压缩文件 → 算术解码 → LLM 预测概率 → Tokenizer → 原文`
- 关键约束：压缩与解压必须使用**完全相同的 LLM 模型与推理配置**，保证概率逐比特一致，实现无损还原。

## 项目结构

```
taichi-compression/
├── src/taichi_compress/
│   ├── __init__.py
│   ├── arithmetic.py       # ✅ 算术编码器核心（WNC 整数算术编码，32 位寄存器 / 24 位 CDF）
│   ├── utils.py            # ✅ 比特级 IO（BitWriter/BitReader）与 varint
│   ├── model.py            # ✅ LLM 预测器（KV Cache 增量推理 + logit 量化，任务 2）
│   ├── gguf.py             # ✅ llama.cpp/GGUF 量化后端（q8_0 / q4_k_m，Phase 3B）
│   ├── tokenizer.py        # ✅ 分词与 token 管理（任务 2）
│   ├── compressor.py       # ✅ 压缩主流程 + TAICHI 容器格式（任务 3）
│   ├── decompressor.py     # ✅ 解压主流程 + CRC32 完整性校验（任务 4）
│   ├── benchmark.py        # ✅ 对比 gzip/xz/zstd 的基准原语（任务 5/6）
│   └── cli.py              # ✅ 命令行入口（任务 5）
├── examples/               # 演示脚本
├── scripts/
│   └── benchmark_enwik8.py # ✅ enwik8 基准评测（自动下载语料，任务 6）
├── tests/                  # 单元测试（pytest / unittest 双兼容）
├── pyproject.toml          # uv 项目配置
└── .python-version         # 锁定 Python 3.12（torch 兼容性最佳）
```

## 环境管理（uv）

本项目使用 [uv](https://docs.astral.sh/uv/) 管理依赖与虚拟环境：

```bash
uv sync                                # 创建 .venv 并安装全部依赖（含 dev 组）
uv run pytest                          # 运行测试
uv run python examples/demo_arithmetic.py  # 算术编码效率演示
uv run python examples/demo_model.py   # LLM 预测演示（首次运行下载模型权重）

# 模型测试可用环境变量定制（默认 Qwen2.5-0.5B / CPU）：
TAICHI_TEST_MODEL=HuggingFaceTB/SmolLM2-135M uv run pytest tests/test_model.py
TAICHI_TEST_DEVICE=mps uv run pytest tests/test_model.py
```

## 当前进度

- ✅ **任务 1：算术编码器**（`arithmetic.py` + `utils.py`）
  - Witten–Neal–Cleary 整数算术编码：32 位区间寄存器，E1/E2/E3 重归一化
  - CDF 默认 24 位量化（参考 Nacrith），可配置 1~30 位；每个符号保底 1 个量子，
    零概率 token 也能无损编解码（大词表场景的关键约束）
  - 双层 API：`ArithmeticEncoder` / `ArithmeticDecoder` 流式接口（供压缩/解压主流程
    复用）+ `encode()` / `decode()` 便捷接口（自带迷你文件头）
  - 确定性保证：`build_cdf` 纯整数运算，numpy 与纯 Python 路径逐位一致
- ✅ **任务 2：LLM 预测器**（`model.py` + `tokenizer.py`，默认 Qwen2.5-0.5B）
  - KV Cache 增量推理：上下文逐 token 增长时单步 O(1)，与整段重算结果一致（已验证）
  - 数值确定性：logit 量化（`round(x*1000)/1000`）+ float64 softmax；
    float32 权重实测 MPS 与 CPU 概率差仅 ~1e-6，被量化网格完全吸收
    （跨设备解压受支持）；fp16 权重仅保证同设备逐比特一致（见下）
  - tokenizer 严格往返（中英文/emoji/空白符），encode 不掺入特殊 token，
    BOS 引导由预测器内部对称处理
- ✅ **任务 3/4：压缩器与解压器**（`compressor.py` + `decompressor.py`）
  - `TAICHI` 容器格式：magic/版本/CDF 精度/logit 量化/原始长度/token 数/模型标识/CRC32
  - 逐 token「LLM 预测 → 算术编码」主循环；解压严格镜像压缩侧调用序列，
    CRC32 兜底校验（模型/配置不匹配或比特流损坏时拒绝输出而非吐出错误数据）
  - 预测器自动 reset，单实例可交错处理多个文件；超上下文长度给出明确报错
  - 实测（Qwen2.5-0.5B / MPS，examples/sample.txt 636B）：
    **1.472 bpb（float32）/ 1.484 bpb（fp16）**，gzip -9 的 28% 体积；
    比特流距理论熵下界 < 12 bit
- ✅ **任务 5：CLI**（`cli.py` + `benchmark.py`，安装后即 `taichi-compress` 命令）
  - `-c/-d/--benchmark/--info` 四种模式互斥；`--model/--device/--precision` 可调
  - `--info` 不加载模型即可查看容器头；解压按容器头自动恢复模型与量化配置
  - `--benchmark` 输出与 gzip -9 / xz -9 / zstd -19 的对比表（CJK 对齐）
  - 实测（sample.txt 636B / MPS）：taichi **1.472 bpb（float32）** vs
    gzip 5.107 / xz 5.786 / zstd 5.119 bpb
- ✅ **任务 6：enwik8 基准测试**（`scripts/benchmark_enwik8.py`，自动下载语料）
  - 全文等分段采样 8 × 8192 字符，各算法逐块独立压缩、同口径公平对比
  - **taichi 0.946 bpb（8.46x）**，达到 Phase 1 冲刺目标（<1.0 bpb），
    体积为 gzip -9 / xz -9 / zstd -19 的 **27~28%**（3.7 倍压缩率优势）
  - 全部往返校验通过；JSON 报告落盘 `.benchmarks/enwik8_results.json`

## enwik8 基准结果（Qwen2.5-0.5B / MPS / float16，2026-08）

| 算法 | 压缩比 | bpb | 压缩速度 | 解压速度 |
|------|-------:|--------:|---------:|---------:|
| **taichi (GGUF q8_0)** | **8.44x** | **0.948** | **395 B/s** | **396 B/s** |
| taichi (GGUF q4_k_m) | 8.37x | 0.956 | 446 B/s | 446 B/s |
| taichi (fp16) | 8.45x | 0.947 | 147 B/s | 148 B/s |
| taichi (fp32 基线) | 8.46x | 0.946 | 50 B/s | 48 B/s |
| gzip -9 | 2.25x | 3.560 | 55 MB/s | 413 MB/s |
| xz -9 | 2.27x | 3.520 | 4.0 MB/s | 55 MB/s |
| zstd -19 | 2.30x | 3.481 | 2.9 MB/s | 188 MB/s |

- 采样：enwik8（99.6M 字符）全文等分 8 段各取开头 8192 字符（64 KB，占语料 0.07%），
  各算法逐块独立压缩-解压并逐字节校验；峰值内存 2822 MiB（float32 基线 3772 MiB）
- 复现：`uv run python scripts/benchmark_enwik8.py`（默认 auto → MPS float16，
  约 15 分钟；`--quant q8_0` 约 5 分钟；`--quant float32` 复现基线约 40 分钟；
  支持 `--blocks/--block-chars/--model/--device`）
- 逐块 bpb 区间 0.721 ~ 1.080：纯英文段落低至 0.72，多语言/重标记块偏高
- ✅ **Phase 2 优化：FP16 权重推理**（容器 v2 记录量化类型）
  - 权重精度可配置（`--quant auto|float32|float16`，auto = 加速器用 float16、
    CPU 用 float32）；fp16 权重带宽减半，实测压缩/解压提速 **~2.8x**
    （50 → 147 B/s），峰值内存 3772 → **2822 MiB**，bpb 仅 +0.001
  - 容器 v2 在 v1 基础上新增量化类型字段；float32 输出与 v1 历史文件
    **逐字节兼容**，旧文件可正常解压
  - 一致性边界：fp16 仅保证**同设备**逐比特一致（对称调用序列 + 确定性内核），
    跨设备解压会被 CRC32 拒绝而非解出错误数据；跨设备需求请用 float32 压缩
- ✅ **Phase 3B 优化：llama.cpp / GGUF 量化后端**（`gguf.py`，容器 v3 记录后端+量化）
  - 权重级量化推理（`--quant q8_0 / q4_k_m`）：llama-cpp-python 进程内推理 +
    Metal 加速，duck-type `LLMPredictor` 接口，压缩/解压主流程零改动复用
  - 实测 enwik8（q8_0）：压缩/解压 **395/396 B/s**（较 fp16 再提速 **2.7x**，
    较 fp32 基线 **~8x**），峰值内存 **1923 MiB**（fp16 2822），bpb 仅 +0.002；
    q4_k_m 更快更省：**446 B/s**、峰值 **1799 MiB**，bpb +0.010（体积仍为
    gzip -9 的 27%）；权重文件 q8_0 506 MB / q4_k_m 379 MB（fp16 权重内存约 0.94 GB）
  - 无损协议：HF tokenizer 与 llama.cpp 分词**逐 id 一致**（已验证）；
    加载后对称预热丢弃首次计算图构建结果，此后逐比特确定；容器头记录
    后端+量化类型，解压侧同文件同构建还原，不一致时 CRC32 拒绝
  - `create_predictor()` 统一工厂路由：quant 决定 transformers / llama 后端

## 命令行用法（任务 5）

```bash
taichi-compress -c input.txt -o output.tc       # 压缩（加速器默认 fp16）
taichi-compress -d output.tc -o restored.txt    # 解压（CRC32 校验，配置以容器头为准）
taichi-compress --benchmark input.txt           # 与 gzip/xz/zstd 对比
taichi-compress --info output.tc                # 查看容器头（不加载模型）
taichi-compress -c big.txt --device mps --model Qwen/Qwen2.5-0.5B
taichi-compress -c doc.txt --quant float32      # 需要跨设备解压时用 float32
```

### GGUF 量化后端（Phase 3B）

```bash
uv sync --extra llama                          # 安装 llama-cpp-python（macOS wheel 自带 Metal）
taichi-compress -c input.txt --quant q8_0      # 权重级量化推理（也可 q4_k_m）
taichi-compress -d output.tc                   # 解压按容器头自动路由到 GGUF 后端
```

GGUF 文件按 `{模型名小写}-{quant}.gguf` 约定搜索：`$TAICHI_GGUF_DIR` →
`./.gguf` → `~/.cache/taichi`，也可用 `--gguf FILE` 显式指定。Qwen2.5 官方
repo 提供 `GGUF` 分支的现成量化；如需自行转换（base 模型无官方 GGUF 时）：

```bash
git clone https://github.com/ggml-org/llama.cpp
python llama.cpp/convert_hf_to_gguf.py ~/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B \
    --outfile .gguf/qwen2.5-0.5b-q8_0.gguf --outtype q8_0   # 或 q4_k_m
```

## 便捷 API 示例

```python
from taichi_compress import encode, decode

probs = [0.7, 0.2, 0.1]          # 概率模型（静态分布 / 逐符号序列 / 函数均可）
symbols = [0, 1, 0, 2, 0, 0]

data = encode(probs, symbols)     # → bytes（含文件头）
assert decode(data, probs) == symbols
```

## 文件压缩示例（任务 3/4）

```python
from taichi_compress.model import PredictorConfig, create_predictor
from taichi_compress.compressor import compress_file
from taichi_compress.decompressor import decompress_file

predictor = create_predictor()  # Qwen2.5-0.5B，自动选择设备（cuda > mps > cpu）
stats = compress_file("input.txt", "input.tc", predictor)
print(f"压缩比 {stats.ratio:.2f}x，{stats.bits_per_byte:.3f} bpb")
decompress_file("input.tc", "input.restored.txt", predictor)  # CRC32 校验通过

# GGUF 量化后端：quant 决定路由（q8_0 / q4_k_m → llama.cpp）
gguf_predictor = create_predictor(PredictorConfig(quant="q8_0"))
```

## 参考

- [Language Modeling is Compression](https://arxiv.org/abs/2306.04052) (DeepMind, ICLR 2024)
- [LLMA](https://github.com/WangXuan95/LLMA) · [llama-zip](https://github.com/AlexBuz/llama-zip) · [Nacrith](https://github.com/robtacconelli/Nacrith-GPU)

## License

MIT
