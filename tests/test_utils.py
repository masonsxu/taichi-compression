"""utils 模块自测：BitWriter / BitReader / varint。"""

from __future__ import annotations

import unittest

from taichi_compress.utils import BitReader, BitWriter, read_varint, write_varint


class TestBitWriter(unittest.TestCase):
    def test_full_byte_msb_first(self):
        writer = BitWriter()
        for bit in [1, 0, 1, 1, 0, 0, 0, 1]:
            writer.write_bit(bit)
        self.assertEqual(writer.to_bytes(), bytes([0b10110001]))
        self.assertEqual(writer.bits_written, 8)

    def test_partial_byte_padded_with_zeros(self):
        writer = BitWriter()
        writer.write_bit(1)
        writer.write_bit(1)
        self.assertEqual(writer.to_bytes(), bytes([0b11000000]))
        self.assertEqual(writer.bits_written, 2)

    def test_to_bytes_idempotent_and_non_destructive(self):
        writer = BitWriter()
        writer.write_bit(1)
        first = writer.to_bytes()
        self.assertEqual(first, writer.to_bytes())
        writer.write_bit(0)  # 仍可继续写入
        self.assertEqual(writer.to_bytes(), bytes([0b10000000]))

    def test_multi_byte(self):
        writer = BitWriter()
        for bit in [1] + [0] * 9:  # 1 后跟 9 个 0 → 2 字节
            writer.write_bit(bit)
        self.assertEqual(writer.to_bytes(), bytes([0b10000000, 0b00000000]))

    def test_empty_writer(self):
        self.assertEqual(BitWriter().to_bytes(), b"")


class TestBitReader(unittest.TestCase):
    def test_roundtrip(self):
        bits = [1, 0, 1, 1, 0, 0, 0, 1, 0, 1]
        writer = BitWriter()
        for bit in bits:
            writer.write_bit(bit)
        reader = BitReader(writer.to_bytes())
        self.assertEqual([reader.read_bit() for _ in range(len(bits))], bits)

    def test_read_past_end_returns_zero(self):
        reader = BitReader(b"")
        self.assertEqual([reader.read_bit() for _ in range(40)], [0] * 40)

    def test_bits_read_counter(self):
        reader = BitReader(b"\xff")
        for _ in range(12):
            reader.read_bit()
        self.assertEqual(reader.bits_read, 12)


class TestVarint(unittest.TestCase):
    def test_roundtrip(self):
        for value in [0, 1, 127, 128, 129, 300, 2**7, 2**14, 2**28, 2**31, 2**40, 2**63]:
            with self.subTest(value=value):
                buf = bytearray()
                write_varint(buf, value)
                decoded, pos = read_varint(bytes(buf), 0)
                self.assertEqual(decoded, value)
                self.assertEqual(pos, len(buf))

    def test_known_encoding(self):
        buf = bytearray()
        write_varint(buf, 300)  # 300 = 0b100101100 → LEB128: 0xAC 0x02
        self.assertEqual(bytes(buf), bytes([0xAC, 0x02]))

    def test_negative_raises(self):
        with self.assertRaises(ValueError):
            write_varint(bytearray(), -1)

    def test_truncated_raises(self):
        buf = bytearray()
        write_varint(buf, 2**14)
        with self.assertRaises(ValueError):
            read_varint(bytes(buf)[:-1], 0)


if __name__ == "__main__":
    unittest.main()
