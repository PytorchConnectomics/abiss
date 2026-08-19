#!/usr/bin/env python3
"""Exercise the nucleus cannot-link through the compiled chunk matcher."""

from __future__ import annotations

import csv
import struct
import subprocess
import sys
import tempfile
from pathlib import Path


MATCHING = struct.Struct("<QQQQ")
PAIR = struct.Struct("<QQ")
WIRE = struct.Struct("<QBIQQ")


def wire(sid, state, nucleus_id=0, count=0, total=0):
    return WIRE.pack(sid, state, nucleus_id, count, total)


def run_case(binary, records):
    with tempfile.TemporaryDirectory(prefix="abiss-nuc-match-") as name:
        path = Path(name)
        tag = "unit"
        (path / "matching_faces.data").write_bytes(
            MATCHING.pack(10, 0, 100, 1000)
            + MATCHING.pack(10, 0, 200, 900)
        )
        empty_files = [
            "o_residual_rg.data",
            f"o_incomplete_edges_{tag}.tmp",
            "vetoed_edges.data",
            "o_boundary_nuclei_labels.data",
            "o_ongoing_supervoxel_counts.data",
            "o_ongoing_semantic_labels.data",
            "o_ongoing_seg_size.data",
        ] + [f"o_boundary_{face}_{tag}.tmp" for face in range(6)]
        for filename in empty_files:
            (path / filename).write_bytes(b"")
        (path / "o_ongoing_nuclei_labels.data").write_bytes(b"".join(records))

        result = subprocess.run(
            [str(binary), tag], cwd=path, text=True, capture_output=True
        )
        if result.returncode:
            raise AssertionError(
                f"match_chunks exited {result.returncode}\nstdout:\n{result.stdout}"
                f"\nstderr:\n{result.stderr}"
            )

        remap_data = (path / "extra_remaps.data").read_bytes()
        assert len(remap_data) % PAIR.size == 0
        remaps = {
            PAIR.unpack_from(remap_data, offset)
            for offset in range(0, len(remap_data), PAIR.size)
        }

        nucleus_data = (path / "ongoing_nuclei_labels.data").read_bytes()
        assert len(nucleus_data) % WIRE.size == 0
        nuclei = [
            WIRE.unpack_from(nucleus_data, offset)
            for offset in range(0, len(nucleus_data), WIRE.size)
        ]

        with (path / "nuc_match_cuts.tsv").open(newline="") as handle:
            cuts = list(csv.DictReader(handle, delimiter="\t"))
        return result.stdout, remaps, nuclei, cuts


def main():
    binary = Path(sys.argv[1]).resolve()
    proper_1 = wire(100, 1, 1, 100, 100)
    proper_2 = wire(200, 1, 2, 100, 100)
    same_1 = wire(200, 1, 1, 100, 100)

    stdout, remaps, nuclei, cuts = run_case(binary, [proper_1, proper_2])
    assert (200, 100) not in remaps
    assert [(row[0], row[1], row[2]) for row in nuclei] == [
        (100, 1, 1),
        (200, 1, 2),
    ]
    assert "nuc: match_rejected_remaps 1" in stdout
    assert "nuc: match_conflict_collisions 0" in stdout
    assert len(cuts) == 1
    assert (cuts[0]["oid_nucleus"], cuts[0]["nid_nucleus"]) == ("1", "2")

    stdout, remaps, nuclei, cuts = run_case(binary, [proper_1, same_1])
    assert (200, 100) in remaps
    assert len(nuclei) == 1
    assert nuclei[0][:3] == (100, 1, 1)
    assert "nuc: match_rejected_remaps 0" in stdout
    assert cuts == []

    stdout, remaps, nuclei, cuts = run_case(binary, [])
    assert (200, 100) in remaps
    assert nuclei == []
    assert "nuc: match_rejected_remaps 0" in stdout
    assert cuts == []

    print("test_nuc_match: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
