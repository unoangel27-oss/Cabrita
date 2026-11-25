#!/usr/bin/env python3
"""Notas fijas de los resultados publicados.

Los generadores reescriben sus `.md` al regenerarlos, así que las aclaraciones
metodológicas viven aquí y se imprimen siempre, en vez de quedar como texto
pegado a mano que la siguiente ejecución borra.
"""

from __future__ import annotations

QUALITY_RESOLUTION = (
    "Resolution of the quality suite: perplexity over **160 tokens** "
    "(`benchmarks/corpus.txt`, 798 bytes) and QA over **10 questions** "
    "(`benchmarks/qa.jsonl`), so 80.0% = **8/10** and one answer moves QA by "
    "**10 pp**."
)

REGENERATED_ARTIFACTS = (
    "Regenerated artifacts: files rebuilt from the GGUF carry a header that "
    "lists `p1_p`…`p12_p`, 22 bytes longer than the Phase 0 header (eight "
    "allowed densities). The same `p1_p` is 2,562,220,548 bytes in the Phase 0 "
    "artifact and 2,562,220,570 bytes when rebuilt; the payload and its "
    "SHA-256 are identical."
)


def notes_block() -> str:
    """Bloque Markdown con las notas que sobreviven a cualquier regeneración."""
    return f"**Notes.** (1) {REGENERATED_ARTIFACTS} (2) {QUALITY_RESOLUTION}\n\n"


def with_notes(table: str) -> str:
    """Antepone las notas a una tabla ya formateada."""
    return notes_block() + table
