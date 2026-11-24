#!/usr/bin/env python3
"""Genera los fixtures pequeños compartidos por Python y Rust."""

from __future__ import annotations

import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "converter"))

from tests.fixture import write_tiny_gguf  # noqa: E402
from ukj_convert.convert import convert_gguf  # noqa: E402
from ukj_convert.subs import extract_subs  # noqa: E402

FIXTURES = os.path.join(ROOT, "rust", "tests", "fixtures")


def main() -> int:
    os.makedirs(FIXTURES, exist_ok=True)
    gguf = os.path.join(FIXTURES, "tiny.gguf")
    ukj = os.path.join(FIXTURES, "tiny.ukj")
    subs = os.path.join(FIXTURES, "tiny.subs")
    write_tiny_gguf(gguf)
    convert_gguf(gguf, ukj, density="p3_p", write_subs=False, force=True)
    extract_subs(ukj, subs_path=subs, force=True)
    for path in (gguf, ukj, subs):
        print(f"{path}: {os.path.getsize(path)} bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
