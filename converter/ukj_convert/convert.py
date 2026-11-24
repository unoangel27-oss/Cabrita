"""Conversor GGUF → UKJ v1."""

from __future__ import annotations

import hashlib
import os
import re
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .format import (
    ALIGN,
    MAGIC,
    PREAMBLE,
    VERSION,
    UkjError,
    align_up,
    build_routes,
    default_operators,
    density_code,
    density_table,
    json_bytes,
    json_safe,
    parse_density,
    plan_subs,
    slugify,
    utc_now,
)
from .gguf import GGUFReader

CHUNK = 16 * 1024 * 1024

GGUF_FILE_TYPE = {
    0: "f32",
    1: "f16",
    2: "q4_0",
    3: "q4_1",
    7: "q8_0",
    8: "q5_0",
    9: "q5_1",
    10: "q2_k",
    11: "q3_k_s",
    12: "q3_k_m",
    13: "q3_k_l",
    14: "q4_k_s",
    15: "q4_k_m",
    16: "q5_k_s",
    17: "q5_k_m",
    18: "q6_k",
}


def detect_quantization(reader: GGUFReader, stem: str) -> str:
    m = re.search(r"(iq\d+_[a-z0-9]+|q\d+_k(?:_[sml])?|q\d+_\d|bf16|f16|f32)", stem.lower())
    if m:
        return m.group(1)
    ft = reader.metadata.get("general.file_type")
    if isinstance(ft, int) and ft in GGUF_FILE_TYPE:
        return GGUF_FILE_TYPE[ft]
    return "desconocida"


def _iter_entry_bytes(reader: GGUFReader, entry: Dict[str, Any], chunk: int = CHUNK):
    """Bytes de una entrada del layout.

    Una entrada normal copia el tensor GGUF completo. Una entrada creada por
    `continue parameter` copia solo su rango de origen, de modo que varias
    entradas independientes comparten el mismo tensor fuente.
    """
    by_name = {t.name: t for t in reader.tensors}
    source = entry.get("continued_from")
    if source is None:
        yield from reader.iter_tensor_bytes(by_name[entry["name"]], chunk=chunk)
        return
    t = by_name[source]
    start = int(entry.get("source_offset", 0))
    remaining = int(entry["nbytes"])
    if start + remaining > t.nbytes:
        raise UkjError(
            f"corte de '{source}' fuera del tensor: {start}+{remaining} > {t.nbytes}"
        )
    reader.f.seek(reader.data_start + t.offset + start)
    while remaining:
        b = reader.f.read(min(chunk, remaining))
        if not b:
            raise UkjError(f"GGUF truncado al leer el corte de '{source}'")
        remaining -= len(b)
        yield b


def _hash_layout(
    reader: GGUFReader, layout: List[Dict[str, Any]]
) -> tuple[str, Dict[str, str]]:
    """SHA-256 del payload UKJ (con padding a 64) y de cada tensor.

    El payload se recompone en el orden físico del GGUF, alineando cada
    tensor a 64 bytes; los bytes de cada tensor no cambian.
    """
    h_payload = hashlib.sha256()
    hashes: Dict[str, str] = {}
    pos = 0
    for entry in layout:
        off = int(entry["offset"])
        if off < pos:
            raise UkjError("layout de tensores solapado")
        if off > pos:
            h_payload.update(b"\0" * (off - pos))
        h = hashlib.sha256()
        for b in _iter_entry_bytes(reader, entry):
            h.update(b)
            h_payload.update(b)
        hashes[entry["name"]] = h.hexdigest()
        pos = off + int(entry["nbytes"])
    payload_len = int(reader_layout_len(layout))
    if payload_len < pos:
        raise UkjError("layout de tensores excede el payload")
    if payload_len > pos:
        h_payload.update(b"\0" * (payload_len - pos))
    return h_payload.hexdigest(), hashes


def reader_layout_len(layout: List[Dict[str, Any]]) -> int:
    if not layout:
        return 0
    last = layout[-1]
    return align_up(int(last["offset"]) + int(last["nbytes"]), ALIGN)


def _write_payload_layout(
    reader: GGUFReader,
    layout: List[Dict[str, Any]],
    out,
    payload_len: int,
    file_hasher=None,
) -> str:
    h = hashlib.sha256()
    pos = 0
    for entry in layout:
        off = int(entry["offset"])
        if off < pos:
            raise UkjError("layout de tensores solapado")
        if off > pos:
            zeros = b"\0" * (off - pos)
            out.write(zeros)
            h.update(zeros)
            if file_hasher is not None:
                file_hasher.update(zeros)
        for b in _iter_entry_bytes(reader, entry):
            out.write(b)
            h.update(b)
            if file_hasher is not None:
                file_hasher.update(b)
        pos = off + int(entry["nbytes"])
    if pos > payload_len:
        raise UkjError("layout de tensores excede el payload")
    if pos < payload_len:
        zeros = b"\0" * (payload_len - pos)
        out.write(zeros)
        h.update(zeros)
        if file_hasher is not None:
            file_hasher.update(zeros)
    return h.hexdigest()


def _continued_entries(
    t, p: int, route: str
) -> List[Dict[str, Any]]:
    """Divide un tensor irregular en parámetros independientes (definición 46).

    Cada rebanada conserva un solo subparámetro (ella misma), no tiene
    subparámetros propios y declara `continued_from` para que el runtime pueda
    reensamblar el tensor lógico si lo necesita.
    """
    slices = plan_subs(t.nbytes, t.rows, p, t.block_bytes)
    row_bytes = t.nbytes // t.rows if t.rows else 0
    entries: List[Dict[str, Any]] = []
    for s in slices:
        nbytes = int(s["nbytes"])
        row_count = int(s.get("row_count") or 0)
        if row_count == 0 and row_bytes and nbytes % row_bytes == 0:
            row_count = nbytes // row_bytes
        dims = list(t.dims)
        rows = None
        if row_count and len(dims) >= 2:
            dims[1] = row_count
            rows = row_count
        entries.append(
            {
                "name": f"{t.name}.c{s['index']}",
                "dims": dims,
                "ggml_type": t.ggml_type,
                "type": t.type_name,
                "nbytes": nbytes,
                "rows": rows,
                "route": route,
                "density": "p1_p",
                "subs": [
                    {
                        "index": 0,
                        "row_start": 0,
                        "row_count": row_count,
                        "byte_offset": 0,
                        "nbytes": nbytes,
                    }
                ],
                "continued_from": t.name,
                "independent": True,
                "source_offset": int(s["byte_offset"]),
            }
        )
    return entries


def build_header(
    reader: GGUFReader,
    density: str,
    model_id: str,
    source_path: str,
    continue_irregular: bool = False,
) -> Dict[str, Any]:
    p = parse_density(density)
    density = density_code(p)
    names = [t.name for t in reader.tensors]
    route_of: Dict[str, str] = {}
    for r in build_routes(names):
        for name in r["tensors"]:
            route_of[name] = r["id"]

    groups: List[List[Dict[str, Any]]] = []
    continued = 0
    for t in reader.tensors:
        subs = plan_subs(t.nbytes, t.rows, p, t.block_bytes)
        if continue_irregular and len(subs) < p:
            entries = _continued_entries(t, p, route_of[t.name])
            continued += 1
            groups.append(entries)
            continue
        groups.append(
            [
                {
                    "name": t.name,
                    "dims": list(t.dims),
                    "ggml_type": t.ggml_type,
                    "type": t.type_name,
                    "nbytes": t.nbytes,
                    "rows": t.rows,
                    "route": route_of[t.name],
                    "density": density_code(len(subs)),
                    "subs": subs,
                }
            ]
        )

    # Los cortes de un mismo tensor van contiguos; la alineación se aplica
    # entre grupos para que el runtime pueda tratarlos como un solo bloque.
    tensors: List[Dict[str, Any]] = []
    cursor = 0
    for group in groups:
        cursor = align_up(cursor, ALIGN)
        for entry in group:
            entry["offset"] = cursor
            cursor += int(entry["nbytes"])
            tensors.append(entry)
    payload_len = align_up(cursor, ALIGN)

    names = [e["name"] for e in tensors]
    routes = build_routes(names)
    # Reasignar la ruta de cada entrada (los nombres cambiaron al continuar).
    route_of = {}
    for r in routes:
        for name in r["tensors"]:
            route_of[name] = r["id"]
    for entry in tensors:
        entry["route"] = route_of[entry["name"]]

    metadata = json_safe(reader.metadata)
    metadata_sha = hashlib.sha256(json_bytes(metadata)).hexdigest()
    total_params = sum(t.n_elements for t in reader.tensors)
    quant = detect_quantization(reader, Path(source_path).stem)

    return {
        "format": "UKJ",
        "version": VERSION,
        "created_at": utc_now(),
        "generator": "ukj-convert 0.1.0",
        "layout": {
            "header": "json-utf8",
            "tensor_offset_base": "payload",
            "sub_offset_base": "tensor",
            "endianness": "little",
            "alignment": ALIGN,
        },
        "model": {
            "id": model_id,
            "name": reader.metadata.get("general.name") or model_id,
            "architecture": reader.metadata.get("general.architecture"),
            "quantization": quant,
            "tensor_count": len(tensors),
            "continued_tensors": continued,
            "parameter_count": total_params,
            "metadata_sha256": metadata_sha,
            "source": {
                "format": "GGUF",
                "version": reader.version,
                "file": os.path.basename(source_path),
                "size": reader.file_size,
                "data_start": reader.data_start,
            },
            "metadata": metadata,
        },
        "payload": {
            "length": payload_len,
            "alignment": ALIGN,
            "sha256": None,  # se rellena tras la pasada de checksum
        },
        "routes": routes,
        "operators": default_operators(),
        "densities": density_table(density),
        "tensors": tensors,
    }


def convert_gguf(
    source: str,
    output: str,
    density: str = "p1_p",
    model_id: Optional[str] = None,
    write_subs: bool = True,
    subs_dir: Optional[str] = None,
    force: bool = False,
    continue_irregular: bool = False,
    log: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    """Convierte un GGUF a .ukj y, opcionalmente, materializa su .subs."""
    log = log or (lambda msg: None)
    source = os.path.abspath(source)
    output = os.path.abspath(output)
    if not os.path.isfile(source):
        raise UkjError(f"no existe el GGUF: {source}")
    if source == output:
        raise UkjError("la salida no puede ser el mismo archivo que la entrada")
    if os.path.exists(output) and not force:
        raise UkjError(f"la salida ya existe: {output} (usa --force para reemplazarla)")
    if not output.endswith(".ukj"):
        output += ".ukj"

    t0 = time.time()
    with GGUFReader(source) as reader:
        stem = Path(source).stem
        quant = detect_quantization(reader, stem)
        mid = model_id or f"{stem}-{density}"
        mid = slugify(mid)
        log(
            f"GGUF v{reader.version}: {len(reader.tensors)} tensores, "
            f"{reader.file_size / 1e9:.2f} GB, cuantización {quant}"
        )

        header = build_header(
            reader, density, mid, source, continue_irregular=continue_irregular
        )
        payload_len = header["payload"]["length"]
        log(
            f"plan {density}: {sum(len(t['subs']) for t in header['tensors'])} "
            f"subparámetros; calculando checksums de {payload_len / 1e9:.2f} GB…"
        )
        payload_sha, tensor_sha = _hash_layout(reader, header["tensors"])
        header["payload"]["sha256"] = payload_sha
        for t in header["tensors"]:
            t["sha256"] = tensor_sha[t["name"]]

        header_bytes = json_bytes(header)
        header_sha = hashlib.sha256(header_bytes).hexdigest()
        log(
            f"cabecera {len(header_bytes) / 1e6:.1f} MB; escribiendo {output}…"
        )

        os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
        tmp = f"{output}.tmp-{os.getpid()}"
        try:
            with open(tmp, "wb") as out:
                file_hasher = hashlib.sha256()
                preamble = PREAMBLE.pack(
                    MAGIC,
                    VERSION,
                    len(header_bytes),
                    bytes.fromhex(header_sha),
                    payload_len,
                    bytes.fromhex(payload_sha),
                )
                out.write(preamble)
                file_hasher.update(preamble)
                out.write(header_bytes)
                file_hasher.update(header_bytes)
                copied_sha = _write_payload_layout(
                    reader, header["tensors"], out, payload_len, file_hasher
                )
                if copied_sha != payload_sha:
                    raise UkjError(
                        "el GGUF cambió durante la conversión: el payload copiado "
                        "no coincide con el checksum calculado"
                    )
                out.write(file_hasher.digest())
                out.flush()
                os.fsync(out.fileno())
            os.replace(tmp, output)
        except Exception:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    result: Dict[str, Any] = {
        "ok": True,
        "output": output,
        "model_id": mid,
        "density": density,
        "payload_sha256": payload_sha,
        "header_sha256": header_sha,
        "size": os.path.getsize(output),
        "seconds": round(time.time() - t0, 2),
        "subs": None,
        "continued_tensors": header["model"].get("continued_tensors", 0),
    }
    log(f".ukj escrito: {result['size'] / 1e9:.2f} GB")

    if write_subs:
        from .subs import extract_subs

        subs_path = extract_subs(output, subs_dir=subs_dir, force=True, log=log)
        result["subs"] = subs_path
    return result
