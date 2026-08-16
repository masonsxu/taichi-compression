"""算术编码核心 —— 太极之"阳"。

算术编码把整个符号序列映射到 [0, 1) 上的一个嵌套子区间，再输出区间内任一点
的二进制表示。符号概率越高，占用的区间越宽，消耗的比特越少（信息量 = -log2 P）。
与逐符号编码（如 Huffman）不同，算术编码可以无限逼近信源熵，且天然支持逐 token
变化的概率分布 —— 这正是 LLM 预测器（太极之"阴"）所需要的搭档。

实现：Witten–Neal–Cleary (CACM-87) 风格的整数算术编码。

- 32 位整数寄存器维护闭区间 ``[low, high]``；
- CDF 用 ``1 << precision`` 个量子表示（默认 24 位，参考 Nacrith 的 24 位精度），
  每个符号至少占 1 个量子 —— 保证 LLM 词表中概率被量化为 0 的 token 依然能无损
  编解码（这正是大词表场景的关键约束）；
- 重归一化三种情形：E1（输出 0）、E2（输出 1）、E3（pending 记账下溢）。

两层 API：

- 流式：``ArithmeticEncoder`` / ``ArithmeticDecoder``，逐符号喂 CDF，
  任务 3/4 的 compressor / decompressor 直接使用，可自定义容器格式；
- 便捷：``encode()`` / ``decode()``，自带迷你文件头。模型标识、原始文件长度等
  元数据由任务 3 的容器层在外层负责。

便捷格式文件头布局::

    偏移 0..3  magic = b"TCAC"（TaiChi Arithmetic Coding）
    偏移 4     格式版本 = 1
    偏移 5     CDF 精度（比特数）
    偏移 6..   符号总数（无符号 LEB128 varint）
    其后       算术编码比特流（MSB-first，末尾补 0 对齐字节）

跨设备确定性：压缩与解压两侧必须得到逐比特一致的概率分布（任务 2 通过 logit
量化保证）。``build_cdf`` 是纯整数确定性运算，相同概率必然得到相同 CDF。
"""

from __future__ import annotations

import math
from bisect import bisect_right
from typing import Callable, Sequence, Union

try:  # numpy 为可选依赖：仅用于大词表（如 15 万 token）的 CDF 构建与查找加速
    import numpy as _np
except ImportError:  # pragma: no cover
    _np = None

from .utils import BitReader, BitWriter, read_varint, write_varint

__all__ = [
    "DEFAULT_PRECISION",
    "MIN_PRECISION",
    "MAX_PRECISION",
    "MAGIC",
    "FORMAT_VERSION",
    "ArithmeticEncoder",
    "ArithmeticDecoder",
    "build_cdf",
    "encode",
    "decode",
]

DEFAULT_PRECISION = 24  # CDF 量化精度：2**24 = 16,777,216 个概率量子（参考 Nacrith）
MIN_PRECISION = 1
MAX_PRECISION = 30  # 重归一化保证区间宽度 > 2**30，故 2**precision 必须 <= 2**30

# —— 32 位区间寄存器常量 ——
_MASK32 = (1 << 32) - 1  # 0xFFFFFFFF
_HALF = 1 << 31  # 0x80000000，区间上半判定阈值
_QUARTER = 1 << 30  # 0x40000000
_THREE_QUARTERS = _HALF + _QUARTER  # 0xC0000000

MAGIC = b"TCAC"
FORMAT_VERSION = 1
_HEADER_FIXED = len(MAGIC) + 2  # magic + version + precision

# 概率模型的三种形态：静态分布 / 逐符号分布序列 / 依上下文的函数
ProbabilityModel = Union[
    Sequence[float],  # 静态分布，对每个符号相同
    "Sequence[Sequence[float]]",  # 逐符号分布，第 i 项对应第 i 个符号
    Callable[[Sequence[int]], Sequence[float]],  # f(已编码符号) -> 下一符号的分布
]


def _check_precision(precision: int) -> None:
    """校验 CDF 精度在合法范围（1~30 比特）内。"""
    if not MIN_PRECISION <= precision <= MAX_PRECISION:
        raise ValueError(
            f"CDF 精度必须在 {MIN_PRECISION}~{MAX_PRECISION} 比特之间，当前为 {precision}"
        )


def build_cdf(
    probabilities: Sequence[float], precision: int = DEFAULT_PRECISION
) -> "list[int] | _np.ndarray":
    """把概率分布转换为整数累积分布函数（CDF）。

    返回长度为 n+1 的整数序列 ``cdf``，满足 ``cdf[0] == 0``、
    ``cdf[n] == 2 ** precision`` 且严格递增（每个符号至少 1 个量子）。
    概率无需归一化：内部按权重总和归一化，天然容忍 softmax 输出 1e-7 量级
    的浮点误差。确定性运算：相同输入必得相同输出（跨设备一致性的前提），
    numpy 与纯 Python 两条路径结果逐位一致。

    LLM 长尾适配：真实 LLM 分布中大量 token 概率 < 1e-16，float64 累加会
    提前饱和到 1.0，量化格点随之提前到顶。本实现把每个累积边界夹紧在
    ``total - (n - i)`` 以内，为剩余 token 各保留 1 个量子——尾部 token
    均匀获得最小量子，分布头部的质量分配不受影响。

    Args:
        probabilities: 概率分布（权重），元素须为非负有限数且不全为 0
        precision: CDF 量化精度（比特数）

    Raises:
        ValueError: 词表为空 / 含负数、NaN、Inf / 词表超过量子总数 / 权重全为 0
    """
    _check_precision(precision)
    total = 1 << precision
    if _np is not None and isinstance(probabilities, _np.ndarray):
        return _build_cdf_numpy(probabilities, total)
    return _build_cdf_python(probabilities, total)


def _build_cdf_python(probabilities: Sequence[float], total: int) -> list[int]:
    """build_cdf 的纯 Python 实现（无 numpy 环境下的主路径）。"""
    n = len(probabilities)
    if n == 0:
        raise ValueError("词表为空，无法构建 CDF")
    if n > total:
        raise ValueError(f"词表大小 {n} 超过 CDF 量子总数 {total}，请提高精度")
    cdf = [0] * (n + 1)
    weights = []
    weight_sum = 0.0
    for i, p in enumerate(probabilities):
        p = float(p)
        if not math.isfinite(p) or p < 0.0:
            raise ValueError(f"概率必须为非负有限数，第 {i} 个为 {p!r}")
        weights.append(p)
        weight_sum += p  # 顺序累加，与 numpy.cumsum 逐位一致
    if weight_sum <= 0.0:
        raise ValueError("概率全为 0，无法构建 CDF")
    bound = 0
    run = 0.0
    for i, p in enumerate(weights, start=1):
        run += p
        # 目标量子格点（与 numpy 路径逐位一致：先除权重和、再乘总量、银行家舍入）
        target = round(run / weight_sum * total)
        # 上界夹紧：为剩余 n-i 个 token 各保留 1 个量子（LLM 长尾必需）
        if target > total - (n - i):
            target = total - (n - i)
        # 下界保底：每个符号至少 1 个量子；归纳可证不会超过上面的上界
        if target < bound + 1:
            target = bound + 1
        cdf[i] = target
        bound = target
    return cdf


def _build_cdf_numpy(probabilities: "_np.ndarray", total: int) -> "_np.ndarray":
    """build_cdf 的 numpy 加速路径，结果与纯 Python 路径完全一致。"""
    probs = _np.asarray(probabilities, dtype=_np.float64)
    if probs.ndim != 1:
        raise ValueError("概率分布必须是一维数组")
    n = probs.shape[0]
    if n == 0:
        raise ValueError("词表为空，无法构建 CDF")
    if n > total:
        raise ValueError(f"词表大小 {n} 超过 CDF 量子总数 {total}，请提高精度")
    if not bool(_np.isfinite(probs).all()) or bool((probs < 0).any()):
        raise ValueError("概率必须为非负有限数")
    cum = _np.cumsum(probs)  # 顺序累加，与 Python 路径逐位一致
    if not cum[-1] > 0.0:
        raise ValueError("概率全为 0，无法构建 CDF")
    # 目标格点：归一化后映射到量子（先除总和、再乘总量、rint 银行家舍入）。
    # 与 Python 路径相同的递推 cdf[i] = max(cdf[i-1]+1, min(scaled[i], total-(n-i)))，
    # 向量化为 cdf[i] = i + running_max( min(scaled[j]-j, total-n) )（0 号位
    # 虚拟边界 0）——上界夹紧保证长尾 token 各得 1 个量子且总额恰为 total
    scaled = _np.rint(cum / cum[-1] * total).astype(_np.int64)
    idx = _np.arange(n + 1, dtype=_np.int64)
    shifted = _np.empty(n + 1, dtype=_np.int64)
    shifted[0] = 0
    shifted[1:] = _np.minimum(scaled - idx[1:], total - n)
    return idx + _np.maximum.accumulate(shifted)


class ArithmeticEncoder:
    """流式算术编码器：32 位区间寄存器 + E1/E2/E3 重归一化。

    用法::

        writer = BitWriter()
        encoder = ArithmeticEncoder(writer, precision=24)
        for symbol, cdf in ...:
            encoder.encode_symbol(symbol, cdf)
        encoder.finish()
        payload = writer.to_bytes()
    """

    def __init__(self, bitout: BitWriter, precision: int = DEFAULT_PRECISION) -> None:
        """初始化编码器。

        Args:
            bitout: 比特输出流（通常为 ``BitWriter``）
            precision: CDF 量子精度，1~30 比特
        """
        _check_precision(precision)
        self._bitout = bitout
        self._total = 1 << precision
        self._low = 0  # 当前区间下界（含）
        self._high = _MASK32  # 当前区间上界（含）
        self._pending = 0  # E3 下溢计数：待补发的反相比特数
        self._finished = False

    @property
    def precision(self) -> int:
        """CDF 量子精度（比特数）。"""
        return self._total.bit_length() - 1

    def encode_symbol(self, symbol: int, cdf: Sequence[int]) -> None:
        """按 CDF 编码一个符号。

        Args:
            symbol: 符号索引，0 <= symbol < len(cdf) - 1
            cdf: 整数累积分布，严格递增且两端为 0 与 ``2 ** precision``
        """
        if self._finished:
            raise RuntimeError("finish() 之后不能再编码符号")
        n = len(cdf) - 1
        if n < 1:
            raise ValueError("cdf 至少要有两个边界值")
        if not 0 <= symbol < n:
            raise ValueError(f"符号 {symbol} 超出词表范围 [0, {n})")
        if cdf[0] != 0 or cdf[-1] != self._total:
            raise ValueError(f"cdf 边界必须为 0 和 {self._total}")
        if not cdf[symbol] < cdf[symbol + 1]:
            raise ValueError("cdf 必须严格递增（符号概率量化后为 0）")
        rng = self._high - self._low + 1
        lo = cdf[symbol]
        hi = cdf[symbol + 1]
        # 子区间缩放：把 [lo, hi)/total 按比例映射进当前区间。
        # 先基于旧 low 算好两个新边界再赋值，避免相互覆盖。
        new_low = self._low + (rng * lo) // self._total
        new_high = self._low + (rng * hi) // self._total - 1
        self._low, self._high = new_low, new_high
        self._renormalize()

    def finish(self) -> None:
        """冲刷残余状态，输出足以锁定最终区间的最后几个比特。"""
        if self._finished:
            return
        # 循环不变量：low < 0.5 <= high（否则早已重归一化）。补 1 个 pending 后，
        # 输出 "01…1"（区间重心偏下）或 "10…0"（偏上）即可唯一定位区间中点 0.5。
        self._pending += 1
        if self._low < _QUARTER:
            self._output_bit_plus_pending(0)
        else:
            self._output_bit_plus_pending(1)
        self._finished = True

    def _renormalize(self) -> None:
        """重归一化：区间变窄时输出已确定的比特并扩张区间。

        循环不变量：退出时 low < HALF <= high 且区间宽度 > 2**30，
        因此任何 1 量子宽的子区间映射后仍 >= 1，编码永不退化。
        """
        while True:
            if self._high < _HALF:
                # E1：区间完全位于 [0, 0.5)，下一位必为 0
                self._output_bit_plus_pending(0)
                self._low = (self._low << 1) & _MASK32
                self._high = ((self._high << 1) | 1) & _MASK32
            elif self._low >= _HALF:
                # E2：区间完全位于 [0.5, 1)，下一位必为 1
                self._output_bit_plus_pending(1)
                self._low = (self._low - _HALF) << 1
                self._high = ((self._high - _HALF) << 1) | 1
            elif self._low >= _QUARTER and self._high < _THREE_QUARTERS:
                # E3：区间跨 0.5 且落在 [0.25, 0.75) 内。下一位暂时未知，
                # 但再下一位必为其反相 —— 先记账，待 E1/E2 时一并补发
                self._pending += 1
                self._low = (self._low - _QUARTER) << 1
                self._high = ((self._high - _QUARTER) << 1) | 1
            else:
                return

    def _output_bit_plus_pending(self, bit: int) -> None:
        """输出 1 个比特，并补发此前 E3 挂起的全部反相比特。"""
        self._bitout.write_bit(bit)
        while self._pending > 0:
            self._bitout.write_bit(bit ^ 1)
            self._pending -= 1


class ArithmeticDecoder:
    """流式算术解码器：与 :class:`ArithmeticEncoder` 严格互逆。

    用法::

        decoder = ArithmeticDecoder(BitReader(payload), precision=24)
        for cdf in ...:
            symbol = decoder.decode_symbol(cdf)
    """

    def __init__(self, bitin: BitReader, precision: int = DEFAULT_PRECISION) -> None:
        """初始化解码器。

        Args:
            bitin: 比特输入流（通常为 ``BitReader``）
            precision: CDF 量子精度，必须与编码时一致
        """
        _check_precision(precision)
        self._bitin = bitin
        self._total = 1 << precision
        self._low = 0
        self._high = _MASK32
        self._code = 0
        # 预读 32 位对齐寄存器；数据不足时 BitReader 以 0 填充
        for _ in range(32):
            self._code = (self._code << 1) | bitin.read_bit()

    def decode_symbol(self, cdf: Sequence[int]) -> int:
        """用与编码侧完全一致的 CDF 解出下一个符号。

        Args:
            cdf: 整数累积分布，约束同 ``ArithmeticEncoder.encode_symbol``

        Returns:
            解码出的符号索引

        Raises:
            ValueError: cdf 非法或比特流损坏
        """
        n = len(cdf) - 1
        if n < 1:
            raise ValueError("cdf 至少要有两个边界值")
        if cdf[0] != 0 or cdf[-1] != self._total:
            raise ValueError(f"cdf 边界必须为 0 和 {self._total}")
        rng = self._high - self._low + 1
        offset = self._code - self._low
        # 反向映射：把 code 的相对偏移还原为 CDF 量子值（与编码侧缩放互逆），
        # 可证 value 必落在 [cdf[s], cdf[s+1]) 内
        value = ((offset + 1) * self._total - 1) // rng
        symbol = _find_symbol(cdf, value)
        if not 0 <= symbol < n:
            raise ValueError("比特流损坏：解码位置落在 CDF 覆盖范围之外")
        lo = cdf[symbol]
        hi = cdf[symbol + 1]
        new_low = self._low + (rng * lo) // self._total
        new_high = self._low + (rng * hi) // self._total - 1
        self._low, self._high = new_low, new_high
        self._renormalize()
        return symbol

    def _renormalize(self) -> None:
        """与编码器三情形一一对应；code 施加与 low 相同的仿射变换，
        每次动作移入 1 个新比特（E3 时消费的正是编码侧挂起补发的比特）。"""
        while True:
            if self._high < _HALF:
                self._code = ((self._code << 1) & _MASK32) | self._bitin.read_bit()
                self._low = (self._low << 1) & _MASK32
                self._high = (self._high << 1) | 1
            elif self._low >= _HALF:
                self._code = ((self._code - _HALF) << 1) | self._bitin.read_bit()
                self._low = (self._low - _HALF) << 1
                self._high = ((self._high - _HALF) << 1) | 1
            elif self._low >= _QUARTER and self._high < _THREE_QUARTERS:
                self._code = ((self._code - _QUARTER) << 1) | self._bitin.read_bit()
                self._low = (self._low - _QUARTER) << 1
                self._high = ((self._high - _QUARTER) << 1) | 1
            else:
                return


def _find_symbol(cdf: Sequence[int], value: int) -> int:
    """二分定位满足 ``cdf[s] <= value < cdf[s+1]`` 的 s，O(log n)。"""
    if _np is not None and isinstance(cdf, _np.ndarray):
        return int(_np.searchsorted(cdf, value, side="right")) - 1
    return bisect_right(cdf, value) - 1


def _model_fn(model: ProbabilityModel) -> Callable[[Sequence[int]], Sequence[float]]:
    """把三种概率模型统一为 ``next_probs(context) -> probs`` 形式的函数。

    注意：静态分布请让模型返回同一个对象，encode/decode 按对象身份缓存 CDF，
    避免大词表下的重复构建；模型不得原地修改已返回的分布。
    """
    if callable(model):
        return model
    if len(model) == 0:
        return lambda context: model  # 仅当没有符号需要编码时才会被调用
    first = model[0]
    if isinstance(first, (bool, int, float)) or (
        _np is not None and isinstance(first, _np.number)
    ):
        return lambda context: model  # 静态单一分布
    return lambda context: model[len(context)]  # 逐符号分布序列


def encode(
    probabilities: ProbabilityModel,
    symbols: Sequence[int],
    precision: int = DEFAULT_PRECISION,
) -> bytes:
    """便捷压缩：把符号序列编码为含迷你文件头的字节流。

    Args:
        probabilities: 概率模型，三种形态皆可 —— 静态分布 / 逐符号分布序列
            （第 i 项对应第 i 个符号）/ ``f(已编码符号) -> 分布`` 的函数
        symbols: 符号索引序列
        precision: CDF 量子精度，1~30 比特

    Returns:
        文件头 + 算术编码比特流（末尾补 0 对齐字节）
    """
    _check_precision(precision)
    symbols = list(symbols)
    model = _model_fn(probabilities)

    header = bytearray()
    header += MAGIC
    header.append(FORMAT_VERSION)
    header.append(precision)
    write_varint(header, len(symbols))

    writer = BitWriter()
    encoder = ArithmeticEncoder(writer, precision)
    context: list[int] = []
    cached_probs = None
    cached_cdf = None
    for symbol in symbols:
        probs = model(context)
        if probs is not cached_probs:  # 静态模型按对象身份缓存 CDF
            cached_cdf = build_cdf(probs, precision)
            cached_probs = probs
        encoder.encode_symbol(symbol, cached_cdf)
        context.append(symbol)
    encoder.finish()
    return bytes(header) + writer.to_bytes()


def decode(data: bytes, probabilities_func: ProbabilityModel) -> list[int]:
    """便捷解压：还原 :func:`encode` 的输出。

    Args:
        data: ``encode`` 产生的字节流
        probabilities_func: 概率模型，必须与编码时逐比特一致

    Returns:
        原始符号索引序列

    Raises:
        ValueError: 文件头非法（magic / 版本 / 精度）或数据截断
    """
    if len(data) < _HEADER_FIXED or data[: len(MAGIC)] != MAGIC:
        raise ValueError("不是有效的太极压缩算术编码流（magic 不符）")
    if data[len(MAGIC)] != FORMAT_VERSION:
        raise ValueError(f"不支持的格式版本: {data[len(MAGIC)]}")
    precision = data[_HEADER_FIXED - 1]
    _check_precision(precision)
    num_symbols, body_start = read_varint(data, _HEADER_FIXED)
    if num_symbols == 0:
        return []

    model = _model_fn(probabilities_func)
    decoder = ArithmeticDecoder(BitReader(data[body_start:]), precision)
    symbols: list[int] = []
    cached_probs = None
    cached_cdf = None
    for _ in range(num_symbols):
        probs = model(symbols)
        if probs is not cached_probs:
            cached_cdf = build_cdf(probs, precision)
            cached_probs = probs
        symbols.append(decoder.decode_symbol(cached_cdf))
    return symbols
