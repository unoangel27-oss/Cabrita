"""Materialización, verificación y reparación de .subs."""

from __future__ import annotations

import hashlib
import os
from typing import Any, Callable, Dict, Optional

from .container import (
    iter_range,
    open_subs,
    open_ukj,
    plan_subs_payload,
)
from .format import (
    ALIGN,
    PREAMBLE,
    SUBS_MAGIC,
    SUBS_VERSION,
    UkjError,
    json_bytes,
    utc_now,
)

CHUNK = 16 * 1024 * 1024


def default_subs_root() -> str:
    home = os.environ.get("CABRITA_HOME") or os.path.join(
        os.path.expanduser("~"), ".cabrita"
    )
    return os.path.join(home, "models")


def subs_path_for(model_id: str, subs_root: Optional[str] = None) -> str:
    if not model_id or "/" in model_id or "\\" in model_id or model_id in (".", ".."):
        raise UkjError(f"model_id inválido para la ruta .subs: {model_id!r}")
    root = subs_root or default_subs_root()
    return os.path.join(root, model_id, "subs.subs")


def _entry_map(header: Dict[str, Any]) -> Dict[tuple[str, int], Dict[str, int]]:
    out = {}
    for t in header["tensors"]:
        for s in t["subs"]:
            out[(t["name"], int(s["index"]))] = {
                "byte_offset": int(t["offset"]) + int(s["byte_offset"]),
                "nbytes": int(s["nbytes"]),
            }
    return out


def extract_subs(
    ukj_path: str,
    subs_path: Optional[str] = None,
    subs_dir: Optional[str] = None,
    force: bool = True,
    log: Optional[Callable[[str], None]] = None,
) -> str:
    """Genera el .subs desde el .ukj (sin necesitar el GGUF original)."""
    log = log or (lambda msg: None)
    ukj = open_ukj(ukj_path, verify_payload=False, verify_trailer=True)
    model_id = str(ukj.header["model"]["id"])
    target = subs_path or subs_path_for(model_id, subs_dir)
    if os.path.exists(target) and not force:
        raise UkjError(f"el .subs ya existe: {target} (usa --force)")
    os.makedirs(os.path.dirname(target) or ".", exist_ok=True)

    entries, payload_len = plan_subs_payload(ukj.header)
    log(f".subs: {len(entries)} subparámetros, {payload_len / 1e9:.2f} GB…")

    # Pasada 1: checksum de cada subparámetro leído del .ukj.
    with open(ukj.path, "rb") as src:
        for e in entries:
            h = hashlib.sha256()
            abs_off = ukj.payload_offset + e["byte_offset"]
            for b in iter_range(src, abs_off, e["nbytes"]):
                h.update(b)
            e["sha256"] = h.hexdigest()

    header = {
        "format": "SUBS",
        "version": SUBS_VERSION,
        "created_at": utc_now(),
        "model_id": model_id,
        "source": {
            "file": os.path.basename(ukj.path),
            "size": ukj.file_size,
            "header_sha256": ukj.header_sha256,
            "payload_sha256": ukj.payload_sha256,
            "payload_len": ukj.payload_len,
        },
        "entries": entries,
    }
    header_bytes = json_bytes(header)
    header_sha = hashlib.sha256(header_bytes).hexdigest()

    tmp = f"{target}.tmp-{os.getpid()}"
    try:
        # Pasada 2: escribir payload y calcular su checksum real.
        h_payload = hashlib.sha256()
        h_file = hashlib.sha256()
        with open(ukj.path, "rb") as src, open(tmp, "wb") as out:
            preamble = PREAMBLE.pack(
                SUBS_MAGIC,
                SUBS_VERSION,
                len(header_bytes),
                bytes.fromhex(header_sha),
                payload_len,
                b"\0" * 32,  # se reescribe tras conocer el hash
            )
            out.write(preamble)
            h_file.update(preamble)
            out.write(header_bytes)
            h_file.update(header_bytes)
            cursor = 0
            for e in entries:
                pad = e["offset"] - cursor
                if pad:
                    zeros = b"\0" * pad
                    out.write(zeros)
                    h_payload.update(zeros)
                    h_file.update(zeros)
                abs_off = ukj.payload_offset + e["byte_offset"]
                for b in iter_range(src, abs_off, e["nbytes"]):
                    out.write(b)
                    h_payload.update(b)
                    h_file.update(b)
                cursor = e["offset"] + e["nbytes"]
            if cursor != payload_len:
                raise UkjError(
                    f"el payload .subs mide {cursor} y se esperaban {payload_len}"
                )
            out.write(h_file.digest())
            out.flush()
            os.fsync(out.fileno())
        # Reescribir el preámbulo con el checksum del payload definitivo.
        with open(tmp, "r+b") as out:
            out.seek(0)
            out.write(
                PREAMBLE.pack(
                    SUBS_MAGIC,
                    SUBS_VERSION,
                    len(header_bytes),
                    bytes.fromhex(header_sha),
                    payload_len,
                    h_payload.digest(),
                )
            )
            out.flush()
            os.fsync(out.fileno())
        # El trailer cubre el preámbulo definitivo: recalcularlo.
        with open(tmp, "rb") as f:
            h = hashlib.sha256()
            remaining = PREAMBLE.size + len(header_bytes) + payload_len
            while remaining > 0:
                b = f.read(min(CHUNK, remaining))
                if not b:
                    raise UkjError("archivo .subs temporal truncado")
                h.update(b)
                remaining -= len(b)
        with open(tmp, "r+b") as f:
            f.seek(PREAMBLE.size + len(header_bytes) + payload_len)
            f.write(h.digest())
            f.truncate()
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, target)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise

    check = verify_subs(ukj.path, target)
    if not check["ok"]:
        raise UkjError(f"el .subs recién creado no valida: {check['message']}")
    log(f".subs escrito: {target} ({os.path.getsize(target) / 1e9:.2f} GB)")
    return target


def verify_subs(
    ukj_path: str,
    subs_path: Optional[str] = None,
    subs_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Comprueba que el .subs corresponde al .ukj y que sus bytes son íntegros."""
    ukj = open_ukj(ukj_path, verify_payload=False, verify_trailer=True)
    target = subs_path or subs_path_for(str(ukj.header["model"]["id"]), subs_dir)
    if not os.path.isfile(target):
        return {
            "ok": False,
            "status": "missing",
            "path": target,
            "message": f"no existe el .subs: {target}",
        }
    try:
        subs = open_subs(target, verify_payload=True)
    except UkjError as exc:
        return {"ok": False, "status": "invalid", "path": target, "message": str(exc)}

    h = subs.header
    if h.get("model_id") != ukj.header["model"]["id"]:
        return {
            "ok": False,
            "status": "mismatch",
            "path": target,
            "message": "el .subs pertenece a otro modelo",
        }
    src = h.get("source") or {}
    if (
        src.get("header_sha256") != ukj.header_sha256
        or src.get("payload_sha256") != ukj.payload_sha256
        or int(src.get("payload_len", -1)) != ukj.payload_len
    ):
        return {
            "ok": False,
            "status": "desync",
            "path": target,
            "message": "el .subs está desincronizado con el .ukj actual",
        }

    expected = _entry_map(ukj.header)
    seen = set()
    cursor = 0
    with open(target, "rb") as f:
        for e in h.get("entries") or []:
            key = (e.get("tensor"), int(e.get("index", -1)))
            if key in seen:
                return {
                    "ok": False,
                    "status": "invalid",
                    "path": target,
                    "message": f"entrada duplicada en .subs: {key}",
                }
            seen.add(key)
            exp = expected.get(key)
            if exp is None:
                return {
                    "ok": False,
                    "status": "invalid",
                    "path": target,
                    "message": f"el .subs tiene un subparámetro que el .ukj no declara: {key}",
                }
            if int(e.get("nbytes", -1)) != exp["nbytes"]:
                return {
                    "ok": False,
                    "status": "invalid",
                    "path": target,
                    "message": f"tamaño distinto para {key}",
                }
            off = int(e.get("offset", -1))
            if off < cursor or off % ALIGN != 0:
                return {
                    "ok": False,
                    "status": "invalid",
                    "path": target,
                    "message": f"offset inválido en .subs para {key}",
                }
            got = hashlib.sha256()
            for b in iter_range(f, subs.payload_offset + off, exp["nbytes"]):
                got.update(b)
            if got.hexdigest() != e.get("sha256"):
                return {
                    "ok": False,
                    "status": "corrupt",
                    "path": target,
                    "message": f"bytes corruptos en {key[0]} sub {key[1]}",
                }
            cursor = off + exp["nbytes"]
    missing = set(expected) - seen
    if missing:
        return {
            "ok": False,
            "status": "incomplete",
            "path": target,
            "message": f"faltan {len(missing)} subparámetros en el .subs",
        }
    return {
        "ok": True,
        "status": "ok",
        "path": target,
        "model_id": ukj.header["model"]["id"],
        "entries": len(seen),
        "bytes": cursor,
        "message": "el .subs coincide con el .ukj",
    }


def ensure_subs(
    ukj_path: str,
    subs_path: Optional[str] = None,
    subs_dir: Optional[str] = None,
    log: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    """Repara el .subs si falta o no coincide; no toca nada si ya está bien."""
    log = log or (lambda msg: None)
    check = verify_subs(ukj_path, subs_path, subs_dir)
    if check["ok"]:
        log(f".subs OK: {check['path']}")
        return check
    log(f".subs {check['status']}: {check['message']}; regenerando…")
    existed = os.path.exists(check["path"])
    target = extract_subs(
        ukj_path,
        subs_path=check["path"],
        subs_dir=subs_dir,
        force=True,
        log=log,
    )
    result = verify_subs(ukj_path, target)
    result["repaired"] = True
    result["action"] = "reparado" if existed else "creado"
    return result
