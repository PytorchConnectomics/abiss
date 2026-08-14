#!/usr/bin/env python3
"""Exercise the nucleus veto through the compiled mean-edge binary."""

from __future__ import annotations

import struct
import subprocess
import sys
import tempfile
from pathlib import Path


def wire(sid, state, nucleus_id=0, count=0, total=0):
    return struct.pack("<QBIQQ", sid, state, nucleus_id, count, total)


def run_case(binary, records):
    with tempfile.TemporaryDirectory(prefix="abiss-nuc-agg-") as name:
        path = Path(name)
        (path / "input_rg.data").write_bytes(
            struct.pack("<QQfQ", 100, 200, 407.4, 420)
        )
        (path / "frozen.data").write_bytes(b"")
        (path / "ns.data").write_bytes(struct.pack("<QQQQ", 100, 100, 200, 100))
        (path / "ongoing_semantic_labels.data").write_bytes(b"")
        (path / "ongoing_nuclei_labels.data").write_bytes(b"".join(records))
        (path / "ongoing_seg_size.data").write_bytes(
            struct.pack("<QQQQ", 100, 100, 200, 100)
        )
        result = subprocess.run(
            [str(binary), "0.25", "input_rg.data", "frozen.data", "ns.data"],
            cwd=path,
            text=True,
            capture_output=True,
        )
        if result.returncode:
            raise AssertionError(
                f"agg exited {result.returncode}\nstdout:\n{result.stdout}"
                f"\nstderr:\n{result.stderr}"
            )
        cuts = (path / "nuc_cuts.data").read_bytes()
        remaps = (path / "remap.data").read_bytes()
        return cuts, remaps


def main():
    binary = Path(sys.argv[1]).resolve()
    proper_1 = wire(100, 1, 1, 100, 100)
    proper_2 = wire(200, 1, 2, 100, 100)
    same_1 = wire(200, 1, 1, 100, 100)
    conflict = wire(100, 2, 0, 0, 100)

    cuts, remaps = run_case(binary, [proper_1, proper_2])
    assert len(cuts) == 16 and set(struct.unpack("<QQ", cuts)) == {100, 200}
    assert remaps == b""

    cuts, remaps = run_case(binary, [proper_1, same_1])
    assert cuts == b"" and len(remaps) == 16

    cuts, remaps = run_case(binary, [proper_1])
    assert cuts == b"" and len(remaps) == 16

    cuts, remaps = run_case(binary, [conflict, proper_2])
    assert len(cuts) == 16 and remaps == b""

    cuts, remaps = run_case(binary, [])
    assert cuts == b"" and len(remaps) == 16
    print("test_nuc_agg: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
