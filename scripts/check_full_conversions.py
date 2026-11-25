#!/usr/bin/env python3
"""Comprueba las conversiones completas GGUF → UKJ de `p1_p`…`p8_p`.

Convierte el GGUF real con cada densidad, verifica el contenedor y compara el
SHA-256 del payload con el de referencia: las ocho densidades describen el
mismo payload, así que cualquier diferencia es un fallo del conversor.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "converter"))

from ukj_convert.container import open_ukj  # noqa: E402

TOOLS_DIR = ROOT / "converter/tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from notes import with_notes  # noqa: E402


def run_json(cmd: List[str]) -> Dict[str, Any]:
    proc = subprocess.run(cmd, text=True, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd)} falló:\n{proc.stderr}")
    return json.loads(proc.stdout)


def render_markdown(rows: List[Dict[str, Any]]) -> str:
    """Tabla publicada con las notas metodológicas fijas."""
    lines = [
        "| Density | File bytes | Payload bytes | Tensors | Payload SHA-256 | Verified | Seconds |",
        "|---|---:|---:|---:|---|---:|---:|",
    ]
    for r in rows:
        sha = "same" if r["matches_reference"] else "**differs**"
        lines.append(
            f"| {r['density']} | {r['file_bytes']} | {r['payload_bytes']} | {r['tensors']} | "
            f"{sha} | {'yes' if r['verified'] else 'no'} | {r['seconds']:.0f} |"
        )
    return with_notes("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gguf", required=True)
    parser.add_argument("--reference", required=True, help=".ukj de referencia")
    parser.add_argument("--cabrita", default=str(ROOT / "rust/target/release/cabrita"))
    parser.add_argument("--out", default=str(ROOT / "benchmarks/results"))
    parser.add_argument("--work", default=str(ROOT / "out/convert-check"))
    parser.add_argument("--from", dest="start", type=int, default=1)
    parser.add_argument("--to", dest="end", type=int, default=8)
    args = parser.parse_args()

    reference = open_ukj(args.reference)
    expected = reference.payload_sha256
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    for p in range(args.start, args.end + 1):
        density = f"p{p}_p"
        model = work / f"check-{density}.ukj"
        print(f"convirtiendo {density}", file=sys.stderr)
        started = time.monotonic()
        subprocess.run(
            [
                sys.executable,
                "-m",
                "ukj_convert.cli",
                "convert",
                args.gguf,
                "-p",
                density,
                "-o",
                str(model),
                "-q",
                "--no-subs",
                "--force",
            ],
            cwd=ROOT / "converter",
            check=True,
        )
        elapsed = time.monotonic() - started
        ukj = open_ukj(str(model))
        checks = run_json([args.cabrita, "verify", str(model), "--json"])
        row = {
            "density": density,
            "file_bytes": model.stat().st_size,
            "payload_bytes": ukj.payload_len,
            "tensors": len(ukj.header.get("tensors", [])),
            "declared_density": ukj.header.get("densities", {}).get("default"),
            "payload_sha256": ukj.payload_sha256,
            "matches_reference": ukj.payload_sha256 == expected,
            "verified": bool(checks.get("ok")),
            "seconds": round(elapsed, 1),
        }
        rows.append(row)
        model.unlink(missing_ok=True)
        print(
            f"  {density}: {row['file_bytes']} B, sha {'OK' if row['matches_reference'] else 'DISTINTO'}, "
            f"{row['seconds']:.0f} s",
            file=sys.stderr,
        )

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "gguf": str(Path(args.gguf).resolve()),
        "reference": str(Path(args.reference).resolve()),
        "reference_payload_sha256": expected,
        "rows": rows,
        "all_match": all(r["matches_reference"] and r["verified"] for r in rows),
    }
    (out_dir / "conversions.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    table = render_markdown(rows)
    (out_dir / "conversions.md").write_text(table, encoding="utf-8")
    print(table)
    return 0 if report["all_match"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
