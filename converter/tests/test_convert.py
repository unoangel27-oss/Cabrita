from __future__ import annotations

import os
import shutil
import struct
import tempfile
import unittest

from ukj_convert.container import open_subs, open_ukj
from ukj_convert.convert import convert_gguf
from ukj_convert.format import UkjError, parse_density, plan_subs
from ukj_convert.gguf import GGUFReader
from ukj_convert.subs import ensure_subs, extract_subs, subs_path_for, verify_subs

from .fixture import tiny_tensors, write_tiny_gguf


class Base(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp(prefix="ukj-test-")
        self.gguf = write_tiny_gguf(os.path.join(self.tmp, "tiny.gguf"))
        self.expected = {name: data for name, _, _, data in tiny_tensors()}
        self.subs_dir = os.path.join(self.tmp, "cabrita")
        with GGUFReader(self.gguf) as r:
            self.tensor_meta = {t.name: t for t in r.tensors}

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def convert(self, density: str, name: str | None = None) -> tuple[str, str]:
        out = os.path.join(self.tmp, name or f"tiny-{density}.ukj")
        result = convert_gguf(
            self.gguf,
            out,
            density=density,
            subs_dir=self.subs_dir,
            force=True,
        )
        return result["output"], result["subs"]


class TestDensity(unittest.TestCase):
    def test_aliases(self) -> None:
        self.assertEqual(parse_density("p_p"), 1)
        self.assertEqual(parse_density("p1_p"), 1)
        self.assertEqual(parse_density("P5_P"), 5)
        self.assertEqual(parse_density("p9_p"), 9)
        self.assertEqual(parse_density("p12_p"), 12)

    def test_invalid(self) -> None:
        for code in ("p0_p", "p13_p", "q4_k", ""):
            with self.assertRaises(ValueError):
                parse_density(code)

    def test_plan_subs_sums_and_blocks(self) -> None:
        subs = plan_subs(68, 2, 3, block_bytes=34)
        self.assertEqual(len(subs), 2)
        self.assertEqual(sum(s["nbytes"] for s in subs), 68)
        self.assertTrue(all(s["nbytes"] % 34 == 0 for s in subs))

    def test_plan_subs_rows(self) -> None:
        subs = plan_subs(24, 3, 3, block_bytes=2)
        self.assertEqual([s["nbytes"] for s in subs], [8, 8, 8])
        self.assertEqual([s["row_count"] for s in subs], [1, 1, 1])


class TestConvert(Base):
    def _check_roundtrip(self, density: str, expected_subs: list[int]) -> None:
        out, subs_path = self.convert(density)
        ukj = open_ukj(out, verify_payload=True)
        self.assertEqual(ukj.header["model"]["id"].endswith(density), True)
        self.assertEqual(ukj.header["densities"]["default"], density)
        self.assertTrue(os.path.isfile(subs_path))

        with open(out, "rb") as f:
            f.seek(ukj.payload_offset)
            payload = f.read(ukj.payload_len)
        self.assertEqual(len(payload), ukj.payload_len)

        counts = []
        for t in ukj.header["tensors"]:
            got = payload[t["offset"] : t["offset"] + t["nbytes"]]
            self.assertEqual(got, self.expected[t["name"]], t["name"])
            self.assertEqual(t["sha256"], __import__("hashlib").sha256(got).hexdigest())
            counts.append(len(t["subs"]))
        self.assertEqual(sorted(counts), sorted(expected_subs))

        subs = open_subs(subs_path, verify_payload=True)
        by_tensor: dict[str, bytearray] = {name: bytearray() for name in self.expected}
        for e in subs.header["entries"]:
            with open(subs_path, "rb") as f:
                f.seek(subs.payload_offset + e["offset"])
                data = f.read(e["nbytes"])
            by_tensor[e["tensor"]] += data
        for name, data in by_tensor.items():
            self.assertEqual(bytes(data), self.expected[name], name)

        check = verify_subs(out, subs_dir=self.subs_dir)
        self.assertTrue(check["ok"], check)
        self.assertEqual(check["entries"], sum(counts))

    def test_p1_p(self) -> None:
        self._check_roundtrip("p1_p", [1, 1, 1])

    def test_p3_p(self) -> None:
        # norm: 3; matriz 3 filas: 3; q8 con 2 bloques: 2
        self._check_roundtrip("p3_p", [2, 3, 3])

    def test_p5_p(self) -> None:
        # norm: 4 bloques F32 -> 4; matriz 3 filas -> 3; q8 2 bloques -> 2
        self._check_roundtrip("p5_p", [2, 4, 5])

    def test_header_contents(self) -> None:
        out, _ = self.convert("p2_p")
        ukj = open_ukj(out)
        h = ukj.header
        self.assertEqual(h["format"], "UKJ")
        self.assertEqual(h["version"], 1)
        self.assertEqual(h["model"]["architecture"], "tiny")
        self.assertEqual(len(h["routes"]), 2)  # global + capa.0
        self.assertEqual({op["name"] for op in h["operators"]}, {"stop-subparameter", "continue parameter"})
        self.assertEqual(sum(len(t["subs"]) for t in h["tensors"]), 6)
        for t in h["tensors"]:
            self.assertEqual(sum(s["nbytes"] for s in t["subs"]), t["nbytes"])
            self.assertEqual(t["offset"] % 32, 0)

    def test_continue_irregular(self) -> None:
        out = os.path.join(self.tmp, "continued.ukj")
        result = convert_gguf(
            self.gguf,
            out,
            density="p5_p",
            write_subs=False,
            force=True,
            continue_irregular=True,
        )
        self.assertGreaterEqual(result["continued_tensors"], 1)
        ukj = open_ukj(out, verify_payload=True)
        with open(out, "rb") as f:
            f.seek(ukj.payload_offset)
            payload = f.read(ukj.payload_len)

        continued = [t for t in ukj.header["tensors"] if t.get("continued_from")]
        self.assertTrue(continued)
        by_source: dict[str, bytearray] = {}
        for t in continued:
            self.assertTrue(t["independent"])
            self.assertEqual(t["density"], "p1_p")
            self.assertEqual(len(t["subs"]), 1)
            got = payload[t["offset"] : t["offset"] + t["nbytes"]]
            src = self.expected[t["continued_from"]]
            start = t["source_offset"]
            self.assertEqual(got, src[start : start + t["nbytes"]], t["name"])
            by_source.setdefault(t["continued_from"], bytearray()).extend(got)
        for source, data in by_source.items():
            self.assertEqual(bytes(data), self.expected[source], source)
        # Las rutas siguen cubriendo todos los tensores.
        routed = {name for r in ukj.header["routes"] for name in r["tensors"]}
        self.assertEqual(routed, {t["name"] for t in ukj.header["tensors"]})

    def test_no_subs(self) -> None:
        out = os.path.join(self.tmp, "nosubs.ukj")
        convert_gguf(self.gguf, out, density="p1_p", write_subs=False, force=True)
        self.assertFalse(
            os.path.exists(subs_path_for("tiny-p1_p", os.path.join(self.tmp, "cabrita")))
        )


class TestSubsRepair(Base):
    def test_missing_subs_is_created(self) -> None:
        out, subs_path = self.convert("p3_p")
        os.remove(subs_path)
        result = ensure_subs(out, subs_dir=self.subs_dir)
        self.assertTrue(result["ok"], result)
        self.assertTrue(result["repaired"])
        self.assertEqual(result["action"], "creado")
        self.assertTrue(os.path.isfile(subs_path))

    def test_corrupt_subs_is_repaired(self) -> None:
        out, subs_path = self.convert("p3_p")
        subs = open_subs(subs_path, verify_payload=False)
        first = subs.header["entries"][0]
        with open(subs_path, "r+b") as f:
            f.seek(subs.payload_offset + first["offset"])
            b = f.read(1)
            f.seek(subs.payload_offset + first["offset"])
            f.write(bytes([b[0] ^ 0xFF]))
        # El checksum del payload .subs ya no cuadra.
        check = verify_subs(out, subs_dir=self.subs_dir)
        self.assertFalse(check["ok"])
        result = ensure_subs(out, subs_dir=self.subs_dir)
        self.assertTrue(result["ok"], result)
        self.assertEqual(result["action"], "reparado")
        self.assertTrue(verify_subs(out, subs_dir=self.subs_dir)["ok"])

    def test_desynced_subs_is_repaired(self) -> None:
        out, subs_path = self.convert("p3_p")
        other = os.path.join(self.tmp, "other.ukj")
        shutil.copyfile(out, other)
        # Cambiar un byte del payload y rehacer los checksums del .ukj.
        ukj = open_ukj(other)
        with open(other, "r+b") as f:
            pos = ukj.payload_offset + 1
            f.seek(pos)
            b = f.read(1)
            f.seek(pos)
            f.write(bytes([b[0] ^ 0xFF]))
            f.seek(ukj.payload_offset)
            import hashlib

            payload_sha = hashlib.sha256(f.read(ukj.payload_len)).digest()
            f.seek(4 + 4 + 8 + 32)
            f.write(struct.pack("<Q", ukj.payload_len))
            f.seek(4 + 4 + 8 + 32 + 8)
            f.write(payload_sha)
            f.seek(0)
            h = hashlib.sha256()
            remaining = ukj.payload_offset + ukj.payload_len
            while remaining:
                chunk = f.read(min(1 << 20, remaining))
                h.update(chunk)
                remaining -= len(chunk)
            f.seek(ukj.payload_offset + ukj.payload_len)
            f.write(h.digest())
        check = verify_subs(other, subs_path)
        self.assertFalse(check["ok"])
        self.assertEqual(check["status"], "desync")
        result = ensure_subs(other, subs_path)
        self.assertTrue(result["ok"], result)
        self.assertEqual(result["action"], "reparado")


class TestInvalid(Base):
    def test_truncated(self) -> None:
        out, _ = self.convert("p1_p")
        with open(out, "r+b") as f:
            f.truncate(os.path.getsize(out) - 1)
        with self.assertRaises(UkjError):
            open_ukj(out)

    def test_bad_header_checksum(self) -> None:
        out, _ = self.convert("p1_p")
        ukj = open_ukj(out)
        with open(out, "r+b") as f:
            f.seek(88 + 5)
            b = f.read(1)
            f.seek(88 + 5)
            f.write(bytes([b[0] ^ 0xFF]))
        with self.assertRaises(UkjError):
            open_ukj(out)

    def test_bad_payload_checksum(self) -> None:
        out, _ = self.convert("p1_p")
        ukj = open_ukj(out)
        with open(out, "r+b") as f:
            f.seek(ukj.payload_offset + 1)
            b = f.read(1)
            f.seek(ukj.payload_offset + 1)
            f.write(bytes([b[0] ^ 0xFF]))
        with self.assertRaises(UkjError):
            open_ukj(out, verify_payload=True)

    def test_bad_magic(self) -> None:
        out, _ = self.convert("p1_p")
        with open(out, "r+b") as f:
            f.write(b"NOPE")
        with self.assertRaises(UkjError):
            open_ukj(out)


if __name__ == "__main__":
    unittest.main()
