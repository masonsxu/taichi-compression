"""二进制与比特级 IO 工具。

算术编码器以"比特"为单位输出，而文件以"字节"为单位存储，本模块负责两者之间的转换：

- ``BitWriter``  把比特流（MSB-first，高位在前）打包为字节序列；
- ``BitReader``  反向逐比特读取，读越界时恒返回 0（算术解码器冲刷尾部时依赖此行为）；
- ``write_varint`` / ``read_varint``  无符号 LEB128 变长整数，用于文件头元数据。
"""

from __future__ import annotations

__all__ = ["BitWriter", "BitReader", "write_varint", "read_varint"]


class BitWriter:
    """MSB-first 比特输出流，边写边打包成字节。"""

    def __init__(self) -> None:
        self._buffer = bytearray()
        self._current = 0  # 正在累积的未满字节（高位在前）
        self._nbits = 0  # _current 中已累积的比特数（0~7）
        self._total_bits = 0

    def write_bit(self, bit: int | bool) -> None:
        """写入单个比特（0 或 1）。"""
        self._current = (self._current << 1) | (1 if bit else 0)
        self._nbits += 1
        self._total_bits += 1
        if self._nbits == 8:
            self._buffer.append(self._current)
            self._current = 0
            self._nbits = 0

    @property
    def bits_written(self) -> int:
        """累计写入的比特数（不含末尾补齐的 0）。"""
        return self._total_bits

    def to_bytes(self) -> bytes:
        """返回字节序列；未满一字节的尾部用 0 补齐。

        不改变内部状态，可重复调用，也可在调用后继续写入。
        """
        if self._nbits == 0:
            return bytes(self._buffer)
        return bytes(self._buffer) + bytes([(self._current << (8 - self._nbits)) & 0xFF])


class BitReader:
    """MSB-first 比特输入流；数据耗尽后继续读取恒返回 0。"""

    def __init__(self, data: bytes) -> None:
        self._data = data
        self._pos = 0  # 当前比特位置

    def read_bit(self) -> int:
        """读取下一个比特；越界时返回 0 并继续推进位置。"""
        byte_index = self._pos >> 3
        if byte_index >= len(self._data):
            self._pos += 1
            return 0
        bit = (self._data[byte_index] >> (7 - (self._pos & 7))) & 1
        self._pos += 1
        return bit

    @property
    def bits_read(self) -> int:
        """累计请求读取的比特数（含越界补零的部分）。"""
        return self._pos


def write_varint(buf: bytearray, value: int) -> None:
    """把非负整数以无符号 LEB128 编码追加到 buf。"""
    if value < 0:
        raise ValueError(f"varint 不能编码负数: {value}")
    while True:
        byte = value & 0x7F
        value >>= 7
        if value:
            buf.append(byte | 0x80)  # 高位置 1 表示后面还有字节
        else:
            buf.append(byte)
            return


def read_varint(data: bytes, pos: int) -> tuple[int, int]:
    """从 data[pos:] 读取一个 LEB128 变长整数。

    Returns:
        (value, 读取结束后的新位置)

    Raises:
        ValueError: 数据被截断或编码长度非法
    """
    result = 0
    shift = 0
    while True:
        if pos >= len(data):
            raise ValueError("varint 数据被截断")
        byte = data[pos]
        pos += 1
        result |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return result, pos
        shift += 7
        if shift > 63:
            raise ValueError("varint 编码过长")
