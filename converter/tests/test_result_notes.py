"""Las notas metodológicas sobreviven a cualquier regeneración de los `.md`."""

import importlib.util
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "converter"))
sys.path.insert(0, str(ROOT / "converter" / "tools"))

from notes import notes_block  # noqa: E402


def load(module_name, relative):
    path = ROOT / relative
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


class ResultNotesTest(unittest.TestCase):
    def assert_notes(self, text):
        self.assertIn("Regenerated artifacts", text)
        self.assertIn("22 bytes", text)
        self.assertIn("160 tokens", text)
        self.assertIn("10 questions", text)

    def test_notes_block(self):
        self.assert_notes(notes_block())

    def test_conversions_table_keeps_notes(self):
        module = load("check_full_conversions", "scripts/check_full_conversions.py")
        table = module.render_markdown(
            [
                {
                    "density": "p1_p",
                    "file_bytes": 1,
                    "payload_bytes": 1,
                    "tensors": 195,
                    "matches_reference": True,
                    "verified": True,
                    "seconds": 1.0,
                }
            ]
        )
        self.assert_notes(table)
        self.assertIn("| p1_p |", table)

    def test_degradation_table_keeps_notes(self):
        module = load("run_degradation", "scripts/run_degradation.py")
        table = module.render_markdown(
            [
                {
                    "density": "p1_p",
                    "subparameters": 195,
                    "perplexity": 20.87,
                    "perplexity_delta": 0.0,
                    "qa_accuracy": 0.8,
                    "qa_delta": 0.0,
                }
            ]
        )
        self.assert_notes(table)
        self.assertIn("| p1_p |", table)

    def test_q_star_table_keeps_notes(self):
        module = load("bench_public", "converter/tools/bench_public.py")
        table = module.render_markdown(
            [
                {
                    "model": "out/model-p1_p.ukj",
                    "q": "q2_k",
                    "density": "p1_p",
                    "tokens_per_s": 2.7,
                    "ttft_ms": 8300,
                    "latency_ms": 14200,
                    "rss_mb": 2124.3,
                    "perplexity": 20.87,
                    "qa_accuracy": 0.8,
                }
            ]
        )
        self.assert_notes(table)
        self.assertIn("p1_p", table)


if __name__ == "__main__":
    unittest.main()
