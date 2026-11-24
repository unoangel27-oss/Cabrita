#!/usr/bin/env python3
"""Comprueba que el payload .ukj reproduce el GGUF byte a byte, tensor a tensor."""

from __future__ import annotations

import argparse
import hashlib
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "converter"))

from ukj_convert.container import open_ukj  # noqa: E402
from ukj_convert.gguf import GGUFReader  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("gguf")
    ap.add_argument("ukj")
    args = ap.parse_args()

    ukj = open_ukj(args.ukj, verify_payload=True)
    with GGUFReader(args.gguf) as g:
        payload_len = g.file_size - g.data_start
        by_name = {t.name: t for t in g.tensors}
        with open(ukj.path, "rb") as f:
            for t in ukj.header["tensors"]:
                src = by_name.get(t["name"])
                if src is None:
                    print(f"FALLO: tensor ausente en GGUF: {t['name']}", file=sys.stderr)
                    return 1
                if (
                    src.offset != t["offset"]
                    or src.nbytes != t["nbytes"]
                    or list(src.dims) != t["dims"]
                    or src.type_name != t["type"]
                ):
                    print(f"FALLO: metadatos distintos en {t['name']}", file=sys.stderr)
                    return 1
                f.seek(ukj.payload_offset + t["offset"])
                remaining = t["nbytes"]
                ht = hashlib.sha256()
                while remaining:
                    b = f.read(min(16 * 1024 * 1024, remaining))
                    if not b:
                        print(f"FALLO: UKJ truncado en {t['name']}", file=sys.stderr)
                        return 1
                    ht.update(b)
                    remaining -= len(b)
                hg = hashlib.sha256()
                for b in g.iter_tensor_bytes(src):
                    hg.update(b)
                if ht.hexdigest() != hg.hexdigest() or ht.hexdigest() != t["sha256"]:
                    print(f"FALLO: sha distinto en {t['name']}", file=sys.stderr)
                    return 1
        print(
            f"OK · {len(ukj.header['tensors'])} tensores · "
            f"{ukj.payload_len / 1e9:.2f} GB de payload UKJ · "
            f"todos los tensores idénticos al GGUF · "
            f"sha256 {ukj.payload_sha256}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
