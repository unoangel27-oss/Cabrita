#!/usr/bin/env python3
"""Genera una variante pN_p de un .ukj sin tocar los pesos.

`p_p` solo reparte cada tensor en subparámetros: el payload es idéntico. Este
tool reescribe la cabecera y copia el payload, útil para el experimento de
degradación p1_p…p12_p sin repetir la conversión desde GGUF.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import struct
import sys
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ukj_convert.container import open_ukj  # noqa: E402
from ukj_convert.format import (  # noqa: E402
    ALLOWED_DENSITIES,
    EXPERIMENTAL_DENSITIES,
    MAGIC,
    PREAMBLE,
    TRAILER_LEN,
    VERSION,
    density_code,
    parse_density,
    plan_subs,
    validate_header,
)

CHUNK = 16 * 1024 * 1024


def _new_model_id(model_id: str, density: str) -> str:
    base = re.sub(r"-p\d+_p$", "", model_id)
    return f"{base}-{density}"


def resplit(src: str, density: str, out: str | None = None) -> Dict[str, Any]:
    code = density_code(parse_density(density))
    ukj = open_ukj(src, verify_payload=False, verify_trailer=True)
    header: Dict[str, Any] = copy.deepcopy(ukj.header)
    header["model"]["id"] = _new_model_id(header["model"]["id"], code)
    densities = header.setdefault("densities", {})
    densities["default"] = code
    densities["declarations"] = [{"scope": "model", "code": code}]
    densities["allowed"] = ALLOWED_DENSITIES + EXPERIMENTAL_DENSITIES

    subs_total = 0
    for tensor in header["tensors"]:
        if tensor.get("continued_from"):
            continue
        subs = plan_subs(
            int(tensor["nbytes"]),
            tensor.get("rows"),
            parse_density(code),
            block_bytes=1,
        )
        tensor["subs"] = subs
        tensor["density"] = density_code(len(subs))
        subs_total += len(subs)

    validate_header(header, ukj.payload_len)
    header_bytes = json.dumps(
        header, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")
    header_sha = hashlib.sha256(header_bytes).digest()
    preamble = PREAMBLE.pack(
        MAGIC,
        VERSION,
        len(header_bytes),
        header_sha,
        ukj.payload_len,
        bytes.fromhex(ukj.payload_sha256),
    )

    target = Path(out or Path(src).with_name(f"{_new_model_id(Path(src).stem, code)}.ukj"))
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f".{target.name}.tmp-{os.getpid()}")
    h_file = hashlib.sha256()
    try:
        with open(src, "rb") as source, open(tmp, "wb") as dst:
            dst.write(preamble)
            h_file.update(preamble)
            dst.write(header_bytes)
            h_file.update(header_bytes)
            source.seek(ukj.payload_offset)
            remaining = ukj.payload_len
            while remaining:
                block = source.read(min(CHUNK, remaining))
                if not block:
                    raise RuntimeError(f"{src}: payload truncado")
                dst.write(block)
                h_file.update(block)
                remaining -= len(block)
            dst.write(h_file.digest())
            dst.flush()
            os.fsync(dst.fileno())
        os.replace(tmp, target)
    except Exception:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        raise

    check = open_ukj(str(target), verify_payload=True, verify_trailer=True)
    if check.header["model"]["id"] != header["model"]["id"]:
        raise RuntimeError("el modelo reescrito no conserva el id esperado")
    return {
        "source": str(Path(src).resolve()),
        "output": str(target.resolve()),
        "density": code,
        "subparameters": subs_total,
        "tensors": len(header["tensors"]),
        "payload_bytes": ukj.payload_len,
        "payload_sha256": ukj.payload_sha256,
        "bytes": target.stat().st_size,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", help=".ukj de entrada")
    parser.add_argument("--density", default="p12_p", help="p1_p…p12_p")
    parser.add_argument("--out", help=".ukj de salida")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    report = resplit(args.source, args.density, args.out)
    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        print(
            f"{report['output']} · {report['density']} · "
            f"{report['subparameters']} subparámetros · {report['bytes']} bytes"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
