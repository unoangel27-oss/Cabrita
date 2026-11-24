"""GGUF mínimo y determinista para los tests del conversor."""

from __future__ import annotations

import os
import struct
from typing import Any, List, Tuple


def _s(text: str) -> bytes:
    b = text.encode("utf-8")
    return struct.pack("<Q", len(b)) + b


def _kv_u32(key: str, value: int) -> bytes:
    return _s(key) + struct.pack("<I", 4) + struct.pack("<I", value)


def _kv_str(key: str, value: str) -> bytes:
    return _s(key) + struct.pack("<I", 8) + _s(value)


def _kv_str_array(key: str, values: List[str]) -> bytes:
    out = _s(key) + struct.pack("<I", 9) + struct.pack("<I", 8) + struct.pack("<Q", len(values))
    for v in values:
        out += _s(v)
    return out


def _kv_i32_array(key: str, values: List[int]) -> bytes:
    out = _s(key) + struct.pack("<I", 9) + struct.pack("<I", 5) + struct.pack("<Q", len(values))
    for v in values:
        out += struct.pack("<i", v)
    return out


def tiny_tensors() -> List[Tuple[str, Tuple[int, ...], int, bytes]]:
    f32 = struct.pack("<4f", 1.0, 2.0, 3.0, 4.0)
    f16 = b"".join(struct.pack("<e", 0.5 * i) for i in range(12))
    q8 = bytes(range(34)) + bytes(range(34, 68))
    return [
        ("blk.0.attn_norm.weight", (4,), 0, f32),
        ("blk.0.attn_q.weight", (4, 3), 1, f16),
        ("token_embd.weight", (32, 2), 8, q8),
    ]


def write_tiny_gguf(path: str, alignment: int = 32) -> str:
    tensors = tiny_tensors()
    offsets = []
    cursor = 0
    for _, _, _, data in tensors:
        cursor = (cursor + alignment - 1) // alignment * alignment
        offsets.append(cursor)
        cursor += len(data)
    payload_len = (cursor + alignment - 1) // alignment * alignment

    header = bytearray()
    header += b"GGUF"
    header += struct.pack("<I", 3)
    header += struct.pack("<Q", len(tensors))
    kvs = [
        _kv_u32("general.alignment", alignment),
        _kv_str("general.name", "Tiny UKJ Test"),
        _kv_str("general.architecture", "tiny"),
        _kv_u32("general.file_type", 0),
        _kv_str_array("tokenizer.ggml.tokens", ["a", "b", "c"]),
        _kv_i32_array("tokenizer.ggml.token_type", [1, 1, 1]),
    ]
    header += struct.pack("<Q", len(kvs))
    for kv in kvs:
        header += kv
    for (name, dims, ggml_type, _), offset in zip(tensors, offsets):
        header += _s(name)
        header += struct.pack("<I", len(dims))
        for d in dims:
            header += struct.pack("<Q", d)
        header += struct.pack("<I", ggml_type)
        header += struct.pack("<Q", offset)

    pad = (alignment - len(header) % alignment) % alignment
    header += b"\0" * pad

    payload = bytearray(b"\0" * payload_len)
    for (_, _, _, data), offset in zip(tensors, offsets):
        payload[offset : offset + len(data)] = data

    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "wb") as f:
        f.write(header)
        f.write(payload)
    return path
