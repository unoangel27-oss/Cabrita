"""Lector GGUF mínimo, sin dependencias, para el conversor UKJ.

Soporta GGUF v2 y v3 (los que publica llama.cpp / Hugging Face).
No interpreta los pesos: solo metadatos, índice de tensores y bytes crudos.
"""

from __future__ import annotations

import os
import struct
from dataclasses import dataclass
from typing import Any, BinaryIO, Dict, Iterator, List, Tuple


class GGUFError(Exception):
    """Error de formato o de lectura de un archivo GGUF."""


@dataclass(frozen=True)
class GGMLType:
    name: str
    block_elems: int
    block_bytes: int


# Tipos GGML que aparecen en los GGUF habituales. El conversor rechaza con
# mensaje claro cualquier tipo que no esté aquí, en vez de adivinar tamaños.
GGML_TYPES: Dict[int, GGMLType] = {
    0: GGMLType("F32", 1, 4),
    1: GGMLType("F16", 1, 2),
    2: GGMLType("Q4_0", 32, 18),
    3: GGMLType("Q4_1", 32, 20),
    6: GGMLType("Q5_0", 32, 22),
    7: GGMLType("Q5_1", 32, 24),
    8: GGMLType("Q8_0", 32, 34),
    9: GGMLType("Q8_1", 32, 36),
    10: GGMLType("Q2_K", 256, 84),
    11: GGMLType("Q3_K", 256, 110),
    12: GGMLType("Q4_K", 256, 144),
    13: GGMLType("Q5_K", 256, 176),
    14: GGMLType("Q6_K", 256, 210),
    15: GGMLType("Q8_K", 256, 292),
    16: GGMLType("IQ2_XXS", 256, 66),
    17: GGMLType("IQ2_XS", 256, 74),
    18: GGMLType("IQ3_XXS", 256, 98),
    19: GGMLType("IQ1_S", 256, 50),
    20: GGMLType("IQ4_NL", 32, 18),
    21: GGMLType("IQ3_S", 256, 110),
    22: GGMLType("IQ2_S", 256, 82),
    23: GGMLType("IQ4_XS", 256, 136),
    24: GGMLType("I8", 1, 1),
    25: GGMLType("I16", 1, 2),
    26: GGMLType("I32", 1, 4),
    27: GGMLType("I64", 1, 8),
    28: GGMLType("F64", 1, 8),
    29: GGMLType("IQ1_M", 256, 56),
    30: GGMLType("BF16", 1, 2),
    34: GGMLType("TQ1_0", 256, 54),
    35: GGMLType("TQ2_0", 256, 66),
    39: GGMLType("MXFP4", 32, 17),
}

# Tipos de valor de los metadatos GGUF.
GGUF_UINT8, GGUF_INT8, GGUF_UINT16, GGUF_INT16 = 0, 1, 2, 3
GGUF_UINT32, GGUF_INT32, GGUF_FLOAT32, GGUF_BOOL = 4, 5, 6, 7
GGUF_STRING, GGUF_ARRAY, GGUF_UINT64, GGUF_INT64, GGUF_FLOAT64 = 8, 9, 10, 11, 12

_SCALAR = {
    GGUF_UINT8: ("<B", 1),
    GGUF_INT8: ("<b", 1),
    GGUF_UINT16: ("<H", 2),
    GGUF_INT16: ("<h", 2),
    GGUF_UINT32: ("<I", 4),
    GGUF_INT32: ("<i", 4),
    GGUF_FLOAT32: ("<f", 4),
    GGUF_UINT64: ("<Q", 8),
    GGUF_INT64: ("<q", 8),
    GGUF_FLOAT64: ("<d", 8),
}


def align_up(value: int, alignment: int) -> int:
    if alignment <= 1:
        return value
    return (value + alignment - 1) // alignment * alignment


@dataclass
class TensorInfo:
    name: str
    dims: Tuple[int, ...]
    ggml_type: int
    offset: int  # relativo al inicio del bloque de datos
    nbytes: int

    @property
    def type_name(self) -> str:
        return GGML_TYPES[self.ggml_type].name

    @property
    def n_elements(self) -> int:
        n = 1
        for d in self.dims:
            n *= d
        return n

    @property
    def rows(self) -> int | None:
        """Filas contiguas completas del tensor, si es 2D o más.

        En GGUF dims[0] es la dimensión rápida (columnas); el resto de
        dimensiones son filas exteriores. Un tensor de expertos [n_embd,
        n_ff, n_expert] tiene dims[1]*dims[2] filas completas.
        """
        if len(self.dims) >= 2:
            rows = 1
            for d in self.dims[1:]:
                rows *= d
            return rows
        return None

    @property
    def block_bytes(self) -> int:
        return GGML_TYPES[self.ggml_type].block_bytes


class GGUFReader:
    def __init__(self, path: str):
        self.path = os.path.abspath(path)
        self.f: BinaryIO = open(self.path, "rb")
        self.file_size = os.path.getsize(self.path)
        self.version = 0
        self.metadata: Dict[str, Any] = {}
        self.tensors: List[TensorInfo] = []
        self.data_start = 0
        self.alignment = 32
        try:
            self._read_header()
        except Exception:
            self.f.close()
            raise

    # ---------- primitivas ----------
    def _read(self, n: int) -> bytes:
        b = self.f.read(n)
        if len(b) != n:
            raise GGUFError(
                f"archivo truncado: se esperaban {n} bytes y quedaban {len(b)}"
            )
        return b

    def _u32(self) -> int:
        return struct.unpack("<I", self._read(4))[0]

    def _u64(self) -> int:
        return struct.unpack("<Q", self._read(8))[0]

    def _string(self) -> str:
        n = self._u64()
        if n > 64 * 1024 * 1024:
            raise GGUFError(f"cadena GGUF absurda ({n} bytes); archivo corrupto")
        raw = self._read(n)
        return raw.decode("utf-8", errors="replace")

    def _value(self, vtype: int) -> Any:
        if vtype in _SCALAR:
            fmt, size = _SCALAR[vtype]
            return struct.unpack(fmt, self._read(size))[0]
        if vtype == GGUF_BOOL:
            return self._read(1)[0] != 0
        if vtype == GGUF_STRING:
            return self._string()
        if vtype == GGUF_ARRAY:
            etype = self._u32()
            count = self._u64()
            if count > 200_000_000:
                raise GGUFError(f"array GGUF absurdo ({count} elementos)")
            # Los arrays de strings del tokenizer son los más grandes del archivo.
            return [self._value(etype) for _ in range(count)]
        raise GGUFError(f"tipo de metadato GGUF desconocido: {vtype}")

    # ---------- cabecera ----------
    def _read_header(self) -> None:
        magic = self._read(4)
        if magic != b"GGUF":
            raise GGUFError("no es un GGUF: magic distinto de 'GGUF'")
        self.version = self._u32()
        if self.version not in (2, 3):
            raise GGUFError(
                f"versión GGUF {self.version} no soportada (se admiten 2 y 3)"
            )
        tensor_count = self._u64()
        kv_count = self._u64()
        if tensor_count > 1_000_000 or kv_count > 1_000_000:
            raise GGUFError("conteos GGUF absurdos; archivo corrupto")

        for _ in range(kv_count):
            key = self._string()
            vtype = self._u32()
            self.metadata[key] = self._value(vtype)

        raw_tensors: List[Tuple[str, Tuple[int, ...], int, int]] = []
        for _ in range(tensor_count):
            name = self._string()
            n_dims = self._u32()
            if n_dims > 8:
                raise GGUFError(f"tensor '{name}' con {n_dims} dimensiones")
            dims = tuple(self._u64() for _ in range(n_dims))
            ggml_type = self._u32()
            offset = self._u64()
            raw_tensors.append((name, dims, ggml_type, offset))

        align = self.metadata.get("general.alignment", 32)
        try:
            self.alignment = int(align)
        except (TypeError, ValueError):
            self.alignment = 32
        if self.alignment <= 0 or self.alignment > 4096:
            self.alignment = 32

        pos = self.f.tell()
        self.data_start = align_up(pos, self.alignment)
        if self.data_start > self.file_size:
            raise GGUFError("cabecera GGUF más grande que el archivo")

        seen = set()
        for name, dims, ggml_type, offset in raw_tensors:
            if name in seen:
                raise GGUFError(f"tensor duplicado en el GGUF: '{name}'")
            seen.add(name)
            if ggml_type not in GGML_TYPES:
                raise GGUFError(
                    f"tensor '{name}': tipo GGML {ggml_type} no soportado por este conversor"
                )
            t = GGML_TYPES[ggml_type]
            n_elements = 1
            for d in dims:
                n_elements *= d
            nblocks = (n_elements + t.block_elems - 1) // t.block_elems
            nbytes = nblocks * t.block_bytes
            if self.data_start + offset + nbytes > self.file_size:
                raise GGUFError(
                    f"tensor '{name}' fuera del archivo (offset {offset}, {nbytes} bytes)"
                )
            self.tensors.append(
                TensorInfo(
                    name=name,
                    dims=dims,
                    ggml_type=ggml_type,
                    offset=offset,
                    nbytes=nbytes,
                )
            )

        # Orden físico del bloque de datos: así se copia de forma contigua.
        self.tensors.sort(key=lambda t: t.offset)
        prev_end = 0
        for t in self.tensors:
            if t.offset < prev_end:
                raise GGUFError(f"tensores solapados en '{t.name}'")
            prev_end = t.offset + t.nbytes

    # ---------- datos ----------
    def iter_tensor_bytes(self, tensor: TensorInfo, chunk: int = 8 * 1024 * 1024) -> Iterator[bytes]:
        self.f.seek(self.data_start + tensor.offset)
        remaining = tensor.nbytes
        while remaining > 0:
            b = self.f.read(min(chunk, remaining))
            if not b:
                raise GGUFError(f"archivo truncado leyendo '{tensor.name}'")
            remaining -= len(b)
            yield b

    def read_tensor(self, tensor: TensorInfo) -> bytes:
        return b"".join(self.iter_tensor_bytes(tensor))

    def close(self) -> None:
        self.f.close()

    def __enter__(self) -> "GGUFReader":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()
