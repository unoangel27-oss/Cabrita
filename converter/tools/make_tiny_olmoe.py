"""Genera un OLMoE mínimo en GGUF para probar `continue parameter`.

No es un modelo útil: hidden=4, 1 capa, 2 expertos, F32. Sirve para que las
normas (16 bytes) no alcancen 8 subparámetros y el conversor las reparta en
parámetros independientes.
"""

from __future__ import annotations

import os
import struct
import sys


def _s(text: str) -> bytes:
    b = text.encode("utf-8")
    return struct.pack("<Q", len(b)) + b


def _kv_u32(key: str, value: int) -> bytes:
    return _s(key) + struct.pack("<I", 4) + struct.pack("<I", value)


def _kv_f32(key: str, value: float) -> bytes:
    return _s(key) + struct.pack("<I", 6) + struct.pack("<f", value)


def _kv_str(key: str, value: str) -> bytes:
    return _s(key) + struct.pack("<I", 8) + _s(value)


def _kv_str_array(key: str, values: list[str]) -> bytes:
    out = _s(key) + struct.pack("<I", 9) + struct.pack("<I", 8)
    out += struct.pack("<Q", len(values))
    for v in values:
        out += _s(v)
    return out


def _kv_i32_array(key: str, values: list[int]) -> bytes:
    out = _s(key) + struct.pack("<I", 9) + struct.pack("<I", 5)
    out += struct.pack("<Q", len(values))
    for v in values:
        out += struct.pack("<i", v)
    return out


def _byte_chars() -> list[str]:
    bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    cs = list(bs)
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return [chr(c) for c in cs]


def make(path: str) -> str:
    hidden, inter, experts, layers = 4, 4, 2, 1
    vocab = _byte_chars() + ["<|user|>", "<|assistant|>", "<|system|>", "<|endoftext|>"]
    tensors = [
        ("token_embd.weight", [hidden, len(vocab)]),
        ("output_norm.weight", [hidden]),
        ("output.weight", [hidden, len(vocab)]),
        ("blk.0.attn_norm.weight", [hidden]),
        ("blk.0.attn_q.weight", [hidden, hidden]),
        ("blk.0.attn_k.weight", [hidden, hidden]),
        ("blk.0.attn_v.weight", [hidden, hidden]),
        ("blk.0.attn_output.weight", [hidden, hidden]),
        ("blk.0.attn_q_norm.weight", [hidden]),
        ("blk.0.attn_k_norm.weight", [hidden]),
        ("blk.0.ffn_norm.weight", [hidden]),
        ("blk.0.ffn_gate_inp.weight", [hidden, experts]),
        ("blk.0.ffn_gate_exps.weight", [hidden, inter * experts]),
        ("blk.0.ffn_up_exps.weight", [hidden, inter * experts]),
        ("blk.0.ffn_down_exps.weight", [inter, hidden * experts]),
    ]
    alignment = 32
    blobs = []
    for name, dims in tensors:
        n = 1
        for d in dims:
            n *= d
        vals = [((i * 37 + len(name)) % 41 - 20) / 200.0 for i in range(n)]
        blobs.append((name, dims, struct.pack(f"<{n}f", *vals)))

    cursor = 0
    offsets = []
    for _, _, data in blobs:
        cursor = (cursor + alignment - 1) // alignment * alignment
        offsets.append(cursor)
        cursor += len(data)
    payload_len = (cursor + alignment - 1) // alignment * alignment

    kvs = [
        _kv_u32("general.alignment", alignment),
        _kv_str("general.name", "Tiny OLMoE Continue"),
        _kv_str("general.architecture", "olmoe"),
        _kv_u32("general.file_type", 0),
        _kv_u32("olmoe.embedding_length", hidden),
        _kv_u32("olmoe.block_count", layers),
        _kv_u32("olmoe.attention.head_count", 1),
        _kv_u32("olmoe.attention.head_count_kv", 1),
        _kv_u32("olmoe.feed_forward_length", inter),
        _kv_u32("olmoe.expert_count", experts),
        _kv_u32("olmoe.expert_used_count", 1),
        _kv_u32("olmoe.context_length", 64),
        _kv_f32("olmoe.attention.layer_norm_rms_epsilon", 1e-5),
        _kv_f32("olmoe.rope.freq_base", 10000.0),
        _kv_str_array("tokenizer.ggml.tokens", vocab),
        _kv_i32_array("tokenizer.ggml.token_type", [1] * len(vocab)),
        _kv_str_array("tokenizer.ggml.merges", []),
        _kv_u32("tokenizer.ggml.bos_token_id", 256),
        _kv_u32("tokenizer.ggml.eos_token_id", 257),
    ]

    header = bytearray()
    header += b"GGUF" + struct.pack("<I", 3)
    header += struct.pack("<Q", len(blobs)) + struct.pack("<Q", len(kvs))
    for kv in kvs:
        header += kv
    for (name, dims, _), offset in zip(blobs, offsets):
        header += _s(name)
        header += struct.pack("<I", len(dims))
        for d in dims:
            header += struct.pack("<Q", d)
        header += struct.pack("<I", 0)  # GGML_TYPE_F32
        header += struct.pack("<Q", offset)
    pad = (alignment - len(header) % alignment) % alignment
    header += b"\0" * pad

    payload = bytearray(b"\0" * payload_len)
    for (_, _, data), offset in zip(blobs, offsets):
        payload[offset : offset + len(data)] = data

    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "wb") as f:
        f.write(header)
        f.write(payload)
    return path


if __name__ == "__main__":
    print(make(sys.argv[1] if len(sys.argv) > 1 else "tiny-olmoe.gguf"))
