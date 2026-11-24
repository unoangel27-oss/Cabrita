"""Formato UKJ v1: contenedor, cabecera y plan de subparámetros.

Layout del archivo .ukj (little endian):

    magic        4 bytes   b"UKJ1"
    version      u32       1
    header_len   u64
    header_sha   32 bytes  SHA-256 de la cabecera JSON tal cual se guarda
    payload_len  u64
    payload_sha  32 bytes  SHA-256 del bloque de datos (con padding)
    header       header_len bytes de JSON UTF-8
    payload      payload_len bytes
    file_sha     32 bytes  SHA-256 de todo lo anterior

Los offsets de tensor son relativos al inicio del payload y están alineados
a 64 bytes. Los offsets de subparámetro son relativos al inicio del tensor.
El payload conserva el bloque de datos GGUF byte a byte (incluido su padding),
de modo que un roundtrip no toca ni un bit de los pesos.
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
import re
import struct
from datetime import datetime, timezone
from typing import Any, Dict, List, Sequence

MAGIC = b"UKJ1"
VERSION = 1
PREAMBLE = struct.Struct("<4sIQ32sQ32s")
PREAMBLE_LEN = PREAMBLE.size  # 88
TRAILER_LEN = 32
ALIGN = 64

SUBS_MAGIC = b"SUBS"
SUBS_VERSION = 1

ALLOWED_DENSITIES = [f"p{i}_p" for i in range(1, 9)]
DENSITY_ALIASES = {"p_p": "p1_p", "p": "p1_p"}


class UkjError(Exception):
    """Contenedor .ukj/.subs inválido o incoherente."""


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def parse_density(code: str) -> int:
    """Devuelve p (subparámetros por parámetro) a partir del código p_p."""
    if code is None:
        return 1
    c = code.strip().lower()
    c = DENSITY_ALIASES.get(c, c)
    m = re.fullmatch(r"p([1-8])_p", c)
    if not m:
        raise ValueError(
            f"densidad '{code}' inválida; usa p1_p…p8_p (p_p equivale a p1_p)"
        )
    return int(m.group(1))


def density_code(p: int) -> str:
    return f"p{p}_p"


def json_bytes(obj: Any) -> bytes:
    return json.dumps(
        obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def json_safe(value: Any) -> Any:
    """Convierte metadatos GGUF a algo serializable sin perder información."""
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, bytes):
        return {"__ukj_bytes__": base64.b64encode(value).decode("ascii")}
    if isinstance(value, float) and not math.isfinite(value):
        return {"__ukj_float__": str(value)}
    return value


def sha256_file(path: str, chunk: int = 16 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def align_up(value: int, alignment: int = ALIGN) -> int:
    return (value + alignment - 1) // alignment * alignment


def plan_subs(
    nbytes: int, rows: int | None, p: int, block_bytes: int = 1
) -> List[Dict[str, int]]:
    """Divide un tensor en subparámetros contiguos.

    Si es una matriz con filas completas suficientes, el corte es por filas
    (lo útil para cargar solo una parte del tensor). Si no, se reparte por
    bytes, respetando bloques GGUF completos para no romper la cuantización.
    En ambos casos la concatenación reproduce el tensor exacto.

    Devuelve entre 1 y p subparámetros: si el tensor es demasiado pequeño para
    p cortes válidos, se reduce el número efectivo (la cabecera declara la
    densidad real de ese tensor).
    """
    if p < 1:
        raise ValueError("p debe ser >= 1")
    if nbytes <= 0:
        raise ValueError("un tensor no puede tener 0 bytes")

    if rows and rows >= p and nbytes % rows == 0:
        row_bytes = nbytes // rows
        if row_bytes % max(1, block_bytes) == 0:
            base = rows // p
            rem = rows % p
            out: List[Dict[str, int]] = []
            start = 0
            for k in range(p):
                nrows = base + (1 if k < rem else 0)
                out.append(
                    {
                        "index": k,
                        "row_start": start,
                        "row_count": nrows,
                        "byte_offset": start * row_bytes,
                        "nbytes": nrows * row_bytes,
                    }
                )
                start += nrows
            return out

    block = max(1, block_bytes)
    max_blocks = nbytes // block
    p_eff = min(p, max(1, max_blocks))
    base_b = max_blocks // p_eff
    rem_b = max_blocks % p_eff
    out = []
    start = 0
    for k in range(p_eff):
        nb = (base_b + (1 if k < rem_b else 0)) * block
        out.append(
            {
                "index": k,
                "row_start": 0,
                "row_count": 0,
                "byte_offset": start,
                "nbytes": nb,
            }
        )
        start += nb
    # Si nbytes no es múltiplo del bloque, el resto pertenece al último sub.
    if start < nbytes:
        out[-1]["nbytes"] += nbytes - start
    return out


def slugify(text: str) -> str:
    s = text.strip().lower()
    s = re.sub(r"[^a-z0-9._-]+", "-", s)
    s = re.sub(r"-{2,}", "-", s).strip("-.")
    return s or "modelo"


def build_routes(tensor_names: Sequence[str]) -> List[Dict[str, Any]]:
    """Taxonomía automática provisional por capas.

    La spec pide etiquetado manual; hasta que exista el editor de mundos,
    cada bloque `blk.N.*` es una ruta y el resto va a `global`. Queda marcada
    como `auto: true` para que el etiquetado manual la sustituya.
    """
    layers: Dict[int, List[str]] = {}
    global_tensors: List[str] = []
    for name in tensor_names:
        m = re.match(r"blk\.(\d+)\.", name)
        if m:
            layers.setdefault(int(m.group(1)), []).append(name)
        else:
            global_tensors.append(name)

    routes: List[Dict[str, Any]] = []
    if global_tensors:
        routes.append(
            {
                "id": "global",
                "label": "Global",
                "kind": "global",
                "auto": True,
                "tensors": global_tensors,
            }
        )
    for i in sorted(layers):
        routes.append(
            {
                "id": f"capa.{i}",
                "label": f"Capa {i}",
                "kind": "layer",
                "auto": True,
                "tensors": layers[i],
            }
        )
    return routes


def default_operators() -> List[Dict[str, Any]]:
    return [
        {
            "name": "stop-subparameter",
            "enabled": True,
            "scope": "hot",
            "tics": {"normal": 2, "anomalous": 20},
            "condition": {"type": "memory_threshold", "value_bytes": None},
            "note": "No borra .subs: deja de desplegar y permite offload en caliente.",
        },
        {
            "name": "continue parameter",
            "enabled": True,
            "definition": 46,
            "condition": {"type": "route_match", "value": None},
            "note": "Reparte parámetros irregulares en parámetros sin subparámetros.",
        },
    ]


def density_table(code: str) -> Dict[str, Any]:
    return {
        "default": code,
        "unit": "subparámetros por parámetro",
        "allowed": ALLOWED_DENSITIES,
        "aliases": {"p_p": "p1_p"},
        "declarations": [{"scope": "model", "code": code}],
    }


def validate_header(header: Dict[str, Any], payload_len: int | None = None) -> None:
    """Validación estructural de una cabecera ya parseada."""
    if not isinstance(header, dict):
        raise UkjError("la cabecera no es un objeto JSON")
    if header.get("format") != "UKJ":
        raise UkjError("la cabecera no declara format=UKJ")
    if int(header.get("version", 0)) != VERSION:
        raise UkjError(
            f"versión de esquema {header.get('version')} no soportada (se espera {VERSION})"
        )

    tensors = header.get("tensors")
    if not isinstance(tensors, list) or not tensors:
        raise UkjError("la cabecera no contiene tensores")

    names = set()
    intervals: List[tuple[int, int, str]] = []
    for t in tensors:
        if not isinstance(t, dict):
            raise UkjError("tensor con entrada no válida")
        name = t.get("name")
        if not name or not isinstance(name, str) or name in names:
            raise UkjError(f"tensor inválido o duplicado: {name!r}")
        names.add(name)
        nbytes = t.get("nbytes")
        if not isinstance(nbytes, int) or nbytes <= 0:
            raise UkjError(f"tensor '{name}': nbytes inválido")
        offset = t.get("offset")
        if not isinstance(offset, int) or offset < 0:
            raise UkjError(f"tensor '{name}': offset inválido")
        intervals.append((offset, offset + nbytes, name))

        subs = t.get("subs")
        if not isinstance(subs, list) or not subs:
            raise UkjError(f"tensor '{name}': sin subparámetros")
        cursor = 0
        for i, s in enumerate(subs):
            if not isinstance(s, dict):
                raise UkjError(f"tensor '{name}': subparámetro inválido")
            if s.get("index") != i:
                raise UkjError(f"tensor '{name}': índices de subparámetro no consecutivos")
            if s.get("byte_offset") != cursor:
                raise UkjError(f"tensor '{name}': subparámetros no contiguos")
            nb = s.get("nbytes")
            if not isinstance(nb, int) or nb <= 0:
                raise UkjError(f"tensor '{name}': subparámetro con tamaño inválido")
            cursor += nb
        if cursor != nbytes:
            raise UkjError(
                f"tensor '{name}': los subparámetros suman {cursor} y el tensor {nbytes}"
            )
        dens = t.get("density")
        if dens is not None:
            try:
                if parse_density(str(dens)) != len(subs):
                    raise UkjError(
                        f"tensor '{name}': densidad {dens} no coincide con {len(subs)} subparámetros"
                    )
            except ValueError as exc:
                raise UkjError(f"tensor '{name}': {exc}") from exc

    intervals.sort()
    prev_end = 0
    prev_name = ""
    for start, end, name in intervals:
        if start < prev_end:
            raise UkjError(f"tensores solapados: '{prev_name}' y '{name}'")
        prev_end = end
        prev_name = name

    if payload_len is not None:
        if not isinstance(payload_len, int) or payload_len <= 0:
            raise UkjError("payload_len inválido")
        for start, end, name in intervals:
            if end > payload_len:
                raise UkjError(
                    f"tensor '{name}' fuera del payload ({end} > {payload_len})"
                )

    routes = header.get("routes")
    if not isinstance(routes, list) or not routes:
        raise UkjError("la cabecera no contiene rutas")
    route_ids = set()
    referenced: set[str] = set()
    for r in routes:
        rid = r.get("id")
        if not rid or rid in route_ids:
            raise UkjError(f"ruta inválida o duplicada: {rid!r}")
        route_ids.add(rid)
        rt = r.get("tensors")
        if not isinstance(rt, list):
            raise UkjError(f"ruta '{rid}': lista de tensores inválida")
        for name in rt:
            if name not in names:
                raise UkjError(f"ruta '{rid}' referencia un tensor inexistente: {name!r}")
            if name in referenced:
                raise UkjError(f"tensor '{name}' asignado a más de una ruta")
            referenced.add(name)
    missing = names - referenced
    if missing:
        sample = ", ".join(sorted(missing)[:5])
        raise UkjError(f"{len(missing)} tensores sin ruta (p. ej. {sample})")

    ops = header.get("operators")
    if not isinstance(ops, list) or not ops:
        raise UkjError("la cabecera no contiene operadores")
    for op in ops:
        if not isinstance(op, dict) or not op.get("name"):
            raise UkjError("operador sin nombre")

    densities = header.get("densities")
    if not isinstance(densities, dict) or not densities.get("default"):
        raise UkjError("la cabecera no declara densidades")
    try:
        parse_density(str(densities["default"]))
    except ValueError as exc:
        raise UkjError(str(exc)) from exc


def read_exact(f, n: int) -> bytes:
    b = f.read(n)
    if len(b) != n:
        raise UkjError("archivo truncado")
    return b
