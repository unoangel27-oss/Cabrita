#!/usr/bin/env python3
"""Benchmark público reproducible para modelos UKJ.

Ejecuta `cabrita bench` sobre un conjunto fijo de prompts, agrega tokens/s,
TTFT, latencia y RSS, y opcionalmente añade perplejidad y QA corto. Genera JSON
y una tabla Markdown lista para la documentación.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent))
from notes import with_notes  # noqa: E402

DEFAULT_PROMPTS = [
    "Explain what a transformer is in one paragraph.",
    "Write a Python function that adds two numbers.",
    "Summarize the main idea of a mixture of experts model.",
]

Q_RE = re.compile(r"(iq\d+_[a-z0-9]+|q\d+_k(?:_[sml])?|q\d+_\d|bf16|f16|f32)", re.I)
P_RE = re.compile(r"(p\d+_p)", re.I)


def infer_q(path: Path) -> str:
    match = Q_RE.search(path.name)
    return match.group(1).lower() if match else "desconocida"


def infer_p(path: Path, header: Dict[str, Any] | None = None) -> str:
    match = P_RE.search(path.name)
    if match:
        return match.group(1).lower()
    if header:
        return str(header.get("densities", {}).get("default", "desconocida"))
    return "desconocida"


def run_json(cmd: List[str]) -> Dict[str, Any]:
    proc = subprocess.run(cmd, text=True, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd)} falló:\n{proc.stderr}")
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"salida JSON inválida de {' '.join(cmd)}: {exc}") from exc


def bench_model(
    cabrita: str,
    model: Path,
    prompts: List[str],
    tokens: int,
    ctx: int,
    backend: str,
) -> Dict[str, Any]:
    runs = []
    for prompt in prompts:
        data = run_json(
            [
                cabrita,
                "bench",
                str(model),
                "--prompt",
                prompt,
                "--tokens",
                str(tokens),
                "--ctx",
                str(ctx),
                "--backend",
                backend,
                "--json",
            ]
        )
        runs.append(data)
    metrics = [r["metrics"] for r in runs]
    return {
        "model": str(model),
        "q": infer_q(model),
        "density": infer_p(model),
        "bytes": model.stat().st_size,
        "tokens_per_s": statistics.mean(m["tokens_per_s"] for m in metrics),
        "ttft_ms": statistics.mean(m["ttft_ms"] for m in metrics),
        "latency_ms": statistics.mean(m["latency_ms"] for m in metrics),
        "rss_mb": statistics.mean(m["rss_mb"] for m in metrics),
        "peak_rss_mb": statistics.mean(m["peak_rss_mb"] for m in metrics),
        "prompt_tokens": statistics.mean(m["prompt_tokens"] for m in metrics),
        "generated_tokens": statistics.mean(m["generated_tokens"] for m in metrics),
    }


def add_quality(
    cabrita: str,
    row: Dict[str, Any],
    text: Path | None,
    qa: Path | None,
    ctx: int,
) -> None:
    if text:
        data = run_json(
            [
                cabrita,
                "eval",
                "perplexity",
                row["model"],
                "--text",
                str(text),
                "--ctx",
                str(ctx),
                "--json",
            ]
        )
        row["perplexity"] = data["perplexity"]
        row["perplexity_tokens"] = data["tokens"]
    if qa:
        data = run_json(
            [
                cabrita,
                "eval",
                "qa",
                row["model"],
                "--data",
                str(qa),
                "--ctx",
                str(ctx),
                "--json",
            ]
        )
        row["qa_accuracy"] = data["accuracy"]
        row["qa_total"] = data["total"]


def markdown(rows: List[Dict[str, Any]]) -> str:
    lines = [
        "| Model | q* | p | tok/s | TTFT ms | Latency ms | RSS MB | Perplexity | QA |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        ppl = f"{row['perplexity']:.3f}" if "perplexity" in row else "—"
        qa = f"{row['qa_accuracy'] * 100:.1f}%" if "qa_accuracy" in row else "—"
        lines.append(
            f"| `{Path(row['model']).name}` | {row['q']} | {row['density']} | "
            f"{row['tokens_per_s']:.2f} | {row['ttft_ms']:.0f} | "
            f"{row['latency_ms']:.0f} | {row['rss_mb']:.1f} | {ppl} | {qa} |"
        )
    return "\n".join(lines)


def render_markdown(rows: List[Dict[str, Any]]) -> str:
    """Tabla publicada con las notas metodológicas fijas."""
    return with_notes(markdown(rows))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cabrita", default="rust/target/release/cabrita", help="binario Cabrita"
    )
    parser.add_argument(
        "models", nargs="+", help="uno o más archivos .ukj"
    )
    parser.add_argument("--tokens", type=int, default=16)
    parser.add_argument("--ctx", type=int, default=512)
    parser.add_argument("--backend", default="cpu", choices=["cpu", "vulkan"])
    parser.add_argument("--text", help="corpus UTF-8 para perplejidad")
    parser.add_argument("--qa", help="JSONL de QA corto")
    parser.add_argument("--out", help="JSON de resultados")
    parser.add_argument("--markdown", help="archivo Markdown de la tabla")
    args = parser.parse_args()

    rows = []
    for model in args.models:
        path = Path(model)
        print(f"benchmark: {path.name}", file=sys.stderr)
        row = bench_model(
            args.cabrita,
            path,
            DEFAULT_PROMPTS,
            args.tokens,
            args.ctx,
            args.backend,
        )
        add_quality(
            args.cabrita,
            row,
            Path(args.text) if args.text else None,
            Path(args.qa) if args.qa else None,
            args.ctx,
        )
        rows.append(row)

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "prompts": DEFAULT_PROMPTS,
        "tokens": args.tokens,
        "ctx": args.ctx,
        "backend": args.backend,
        "rows": rows,
    }
    table = render_markdown(rows)
    if args.out:
        Path(args.out).write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    if args.markdown:
        Path(args.markdown).write_text(table + "\n", encoding="utf-8")
    print(table)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
