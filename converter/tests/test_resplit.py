from __future__ import annotations

import os
import shutil
import sys
import tempfile
import unittest

from ukj_convert.container import open_ukj
from ukj_convert.convert import convert_gguf

from .fixture import write_tiny_gguf

TOOLS = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if TOOLS not in sys.path:
    sys.path.insert(0, TOOLS)

from tools.resplit_density import resplit  # noqa: E402


class TestResplit(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp(prefix="ukj-resplit-")
        self.gguf = write_tiny_gguf(os.path.join(self.tmp, "tiny.gguf"))
        result = convert_gguf(
            self.gguf,
            os.path.join(self.tmp, "tiny-p1_p.ukj"),
            density="p1_p",
            subs_dir=os.path.join(self.tmp, "cabrita"),
            force=True,
        )
        self.p1 = result["output"]

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_resplit_preserves_payload_and_changes_density(self) -> None:
        out = os.path.join(self.tmp, "tiny-p12_p.ukj")
        report = resplit(self.p1, "p12_p", out)
        assert report["density"] == "p12_p"
        assert report["payload_sha256"] == open_ukj(self.p1).payload_sha256
        p1 = open_ukj(self.p1)
        p12 = open_ukj(out, verify_payload=True, verify_trailer=True)
        assert p12.payload_sha256 == p1.payload_sha256
        assert p12.header["densities"]["default"] == "p12_p"
        assert p12.header["model"]["id"].endswith("-p12_p")
        for tensor in p12.header["tensors"]:
            assert tensor["density"] == f"p{len(tensor['subs'])}_p"
            assert sum(s["nbytes"] for s in tensor["subs"]) == tensor["nbytes"]


if __name__ == "__main__":
    unittest.main()
