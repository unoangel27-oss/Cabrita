#!/usr/bin/env python3
"""Experimento de degradación p1_p…p12_p sobre un UKJ base.

Reescribe la densidad del `.ukj` base (los pesos no cambian), evalúa
perplejidad y QA corto, y calcula el umbral de degradación tomando p1_p como
primera línea base.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "converter"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from tools.resplit_density import resplit  # noqa: E402
from tools.notes import with_notes  # noqa: E402


def run_json(cmd: List[str]) -> Dict[str, Any]:
    proc = subprocess.run(cmd, text=True, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd)} falló:\n{proc.stderr}")
    return json.loads(proc.stdout)


def evaluate(
    cabrita: str,
    model: Path,
    corpus: Path,
    qa: Path,
    ctx: int,
) -> Dict[str, Any]:
    ppl = run_json(
        [
            cabrita,
            "eval",
            "perplexity",
            str(model),
            "--text",
            str(corpus),
            "--ctx",
            str(ctx),
            "--json",
        ]
    )
    quality = run_json(
        [
            cabrita,
            "eval",
            "qa",
            str(model),
            "--data",
            str(qa),
            "--ctx",
            str(ctx),
            "--json",
        ]
    )
    return {
        "perplexity": ppl["perplexity"],
        "tokens": ppl["tokens"],
        "qa_accuracy": quality["accuracy"],
        "qa_correct": quality["correct"],
        "qa_total": quality["total"],
    }


def render_markdown(rows: List[Dict[str, Any]]) -> str:
    """Tabla publicada con las notas metodológicas fijas."""
    lines = [
        "| Density | Subparameters | Perplexity | Δ perplexity | QA | Δ QA |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['density']} | {row['subparameters']} | {row['perplexity']:.3f} | "
            f"{row['perplexity_delta']*100:+.1f}% | {row['qa_accuracy']*100:.1f}% | "
            f"{row['qa_delta']*100:+.1f} pp |"
        )
    return with_notes("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, help=".ukj p1_p de partida")
    parser.add_argument("--cabrita", default=str(ROOT / "rust/target/release/cabrita"))
    parser.add_argument("--corpus", default=str(ROOT / "benchmarks/corpus.txt"))
    parser.add_argument("--qa", default=str(ROOT / "benchmarks/qa.jsonl"))
    parser.add_argument("--out", default=str(ROOT / "benchmarks/results"))
    parser.add_argument("--ctx", type=int, default=512)
    parser.add_argument("--from", dest="start", type=int, default=1)
    parser.add_argument("--to", dest="end", type=int, default=12)
    parser.add_argument("--keep", action="store_true", help="conserva los .ukj generados")
    args = parser.parse_args()

    base = Path(args.base)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    models_dir = out_dir / "models"
    models_dir.mkdir(exist_ok=True)

    rows: List[Dict[str, Any]] = []
    for p in range(args.start, args.end + 1):
        density = f"p{p}_p"
        model = models_dir / f"degradation-{density}.ukj"
        print(f"generando {density}", file=sys.stderr)
        report = resplit(str(base), density, str(model))
        print(f"evaluando {density}", file=sys.stderr)
        quality = evaluate(args.cabrita, model, Path(args.corpus), Path(args.qa), args.ctx)
        row = {
            "density": density,
            "p": p,
            "subparameters": report["subparameters"],
            **quality,
        }
        rows.append(row)
        if not args.keep:
            model.unlink(missing_ok=True)

    baseline = rows[0]
    for row in rows:
        row["perplexity_delta"] = row["perplexity"] / baseline["perplexity"] - 1.0
        row["qa_delta"] = row["qa_accuracy"] - baseline["qa_accuracy"]

    # Primera línea base: p1_p. El umbral se fija después de ver la curva.
    threshold = next(
        (
            row["density"]
            for row in rows
            if row["perplexity_delta"] >= 0.10 or row["qa_delta"] <= -0.10
        ),
        None,
    )
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "base": str(base.resolve()),
        "corpus": str(Path(args.corpus).resolve()),
        "qa": str(Path(args.qa).resolve()),
        "ctx": args.ctx,
        "baseline": baseline["density"],
        "threshold": {
            "first_degraded_density": threshold,
            "rule": "perplexity_delta >= 10% or qa_delta <= -10 points vs p1_p",
        },
        "rows": rows,
    }
    (out_dir / "degradation.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    table = render_markdown(rows)
    (out_dir / "degradation.md").write_text(table, encoding="utf-8")
    print(table)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
