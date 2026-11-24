"""Lectura y verificación de contenedores UKJ y SUBS."""

from __future__ import annotations

import hashlib
import json
import os
import struct
from dataclasses import dataclass
from typing import Any, BinaryIO, Dict, List, Optional

from .format import (
    ALIGN,
    MAGIC,
    PREAMBLE,
    PREAMBLE_LEN,
    SUBS_MAGIC,
    SUBS_VERSION,
    TRAILER_LEN,
    VERSION,
    UkjError,
    read_exact,
    validate_header,
)

MAX_HEADER = 256 * 1024 * 1024


@dataclass
class ContainerFile:
    path: str
    magic: bytes
    version: int
    header: Dict[str, Any]
    header_bytes: bytes
    header_sha256: str
    payload_offset: int
    payload_len: int
    payload_sha256: str
    file_sha256: Optional[str] = None

    @property
    def file_size(self) -> int:
        return self.payload_offset + self.payload_len + TRAILER_LEN


def _parse_preamble(f, path: str, magic: bytes, version: int) -> tuple[int, str, int, str]:
    raw = read_exact(f, PREAMBLE_LEN)
    try:
        got_magic, got_version, header_len, header_sha, payload_len, payload_sha = (
            PREAMBLE.unpack(raw)
        )
    except struct.error as exc:
        raise UkjError(f"{path}: preámbulo ilegible") from exc
    if got_magic != magic:
        raise UkjError(
            f"{path}: no es un contenedor {magic.decode('ascii')} "
            f"(magic {got_magic!r})"
        )
    if got_version != version:
        raise UkjError(
            f"{path}: versión {got_version} no soportada (se espera {version})"
        )
    if header_len <= 0 or header_len > MAX_HEADER:
        raise UkjError(f"{path}: tamaño de cabecera inválido ({header_len})")
    return (
        int(header_len),
        header_sha.hex(),
        int(payload_len),
        payload_sha.hex(),
    )


def _load(path: str, magic: bytes, version: int) -> tuple[ContainerFile, BinaryIO]:
    path = os.path.abspath(path)
    if not os.path.isfile(path):
        raise UkjError(f"no existe el archivo: {path}")
    f = open(path, "rb")
    try:
        header_len, header_sha, payload_len, payload_sha = _parse_preamble(
            f, path, magic, version
        )
        header_bytes = read_exact(f, header_len)
        got_header_sha = hashlib.sha256(header_bytes).hexdigest()
        if got_header_sha != header_sha:
            raise UkjError(
                f"{path}: la cabecera no coincide con su checksum "
                "(archivo corrupto o modificado)"
            )
        try:
            header = json.loads(header_bytes.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise UkjError(f"{path}: cabecera JSON inválida: {exc}") from exc
        payload_offset = PREAMBLE_LEN + header_len
        expected_size = payload_offset + payload_len + TRAILER_LEN
        actual_size = os.path.getsize(path)
        if actual_size != expected_size:
            raise UkjError(
                f"{path}: tamaño {actual_size} distinto del declarado {expected_size} "
                "(archivo truncado o con bytes extra)"
            )
        obj = ContainerFile(
            path=path,
            magic=magic,
            version=version,
            header=header,
            header_bytes=header_bytes,
            header_sha256=header_sha,
            payload_offset=payload_offset,
            payload_len=payload_len,
            payload_sha256=payload_sha,
        )
        return obj, f
    except Exception:
        f.close()
        raise


def open_ukj(path: str, verify_payload: bool = False, verify_trailer: bool = True) -> ContainerFile:
    """Abre y valida estructuralmente un .ukj.

    verify_payload recalcula el SHA-256 del payload (lento pero completo).
    verify_trailer comprueba el checksum final del archivo.
    """
    obj, f = _load(path, MAGIC, VERSION)
    try:
        validate_header(obj.header, obj.payload_len)
        if verify_payload or verify_trailer:
            h_file = hashlib.sha256() if verify_trailer else None
            h_payload = hashlib.sha256() if verify_payload else None
            f.seek(0)
            total = obj.payload_offset + obj.payload_len
            pos = 0
            while pos < total:
                b = f.read(min(16 * 1024 * 1024, total - pos))
                if not b:
                    raise UkjError(f"{obj.path}: archivo truncado")
                if h_file is not None:
                    h_file.update(b)
                if h_payload is not None and pos + len(b) > obj.payload_offset:
                    lo = max(0, obj.payload_offset - pos)
                    h_payload.update(b[lo:])
                pos += len(b)
            if h_payload is not None:
                computed_payload = h_payload.hexdigest()
                if computed_payload != obj.payload_sha256:
                    raise UkjError(
                        f"{obj.path}: el payload no coincide con su checksum "
                        "(archivo corrupto o modificado)"
                    )
            if h_file is not None:
                computed_file = h_file.hexdigest()
                trailer = read_exact(f, TRAILER_LEN).hex()
                if trailer != computed_file:
                    raise UkjError(
                        f"{obj.path}: checksum final incorrecto "
                        "(archivo corrupto o modificado)"
                    )
                obj.file_sha256 = computed_file
        return obj
    finally:
        f.close()


def open_subs(path: str, verify_payload: bool = True) -> ContainerFile:
    obj, f = _load(path, SUBS_MAGIC, SUBS_VERSION)
    try:
        if obj.header.get("format") != "SUBS":
            raise UkjError(f"{obj.path}: la cabecera no declara format=SUBS")
        if verify_payload:
            h_payload = hashlib.sha256()
            h_file = hashlib.sha256()
            f.seek(0)
            total = obj.payload_offset + obj.payload_len
            pos = 0
            while pos < total:
                b = f.read(min(16 * 1024 * 1024, total - pos))
                if not b:
                    raise UkjError(f"{obj.path}: archivo truncado")
                h_file.update(b)
                if pos + len(b) > obj.payload_offset:
                    lo = max(0, obj.payload_offset - pos)
                    h_payload.update(b[lo:])
                pos += len(b)
            if h_payload.hexdigest() != obj.payload_sha256:
                raise UkjError(f"{obj.path}: el payload .subs no coincide con su checksum")
            trailer = read_exact(f, TRAILER_LEN).hex()
            if trailer != h_file.hexdigest():
                raise UkjError(f"{obj.path}: checksum final incorrecto")
        return obj
    finally:
        f.close()


def iter_range(f: BinaryIO, offset: int, length: int, chunk: int = 8 * 1024 * 1024):
    f.seek(offset)
    remaining = length
    while remaining > 0:
        b = f.read(min(chunk, remaining))
        if not b:
            raise UkjError("archivo truncado")
        remaining -= len(b)
        yield b


def hash_range(f: BinaryIO, offset: int, length: int, chunk: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    for b in iter_range(f, offset, length, chunk):
        h.update(b)
    return h.hexdigest()


def subs_entries(header: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Lista plana de subparámetros de un .ukj, con offsets ya resueltos."""
    out: List[Dict[str, Any]] = []
    for t in header["tensors"]:
        base = int(t["offset"])
        for s in t["subs"]:
            out.append(
                {
                    "tensor": t["name"],
                    "index": int(s["index"]),
                    "byte_offset": base + int(s["byte_offset"]),
                    "nbytes": int(s["nbytes"]),
                }
            )
    return out


def plan_subs_payload(header: Dict[str, Any]) -> tuple[List[Dict[str, Any]], int]:
    """Asigna offsets alineados dentro del payload .subs."""
    entries: List[Dict[str, Any]] = []
    cursor = 0
    for e in subs_entries(header):
        cursor = (cursor + ALIGN - 1) // ALIGN * ALIGN
        item = dict(e)
        item["offset"] = cursor
        entries.append(item)
        cursor += item["nbytes"]
    return entries, cursor
