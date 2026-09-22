#!/usr/bin/env python3
"""Must-link every supervoxel of one nucleus into a single region-graph node.

This is em_seg's soma "snap" (`seg_pipeline.py:366`: relabel every watershed
fragment overlapping soma i onto one id, BEFORE the region graph is built),
expressed with machinery ABISS already has: `chunkmap.data` is a supervoxel
equivalence table that `atomic_chunk_ME.cpp:65` loads and hands to every
extractor's `output()`, including the affinity and chunked-RG extractors. Two
supervoxels mapped to the same target therefore collapse into one RG node, and
their mutual edge becomes a self-edge that the agglomerator already drops.

Why this is needed on top of the NUC_PATH veto: the veto only decides WHETHER to
merge. Nothing holds a cell body together, so a soma fragments -- measured on the
zebrafinch 173/213 fusion as nucleus-213 dominance 1.000 -> 0.642 (L120). The snap
removes the fragmentation at its source by making each soma one node to begin with.

Runs between cut_chunk_agg.py (which writes seg.raw / nuc.raw) and acme. Opt-in via
ABISS_NUC_SNAP=1 so the verified no-snap behaviour is unchanged by default.

The supervoxel -> nucleus rule is deliberately IDENTICAL to NucExtractor's
(ABISS_NUC_MIN_TAGGED total floor, then ABISS_NUC_DOMINANCE on the winning id), so
the snap and the payload can never disagree about which nucleus owns a supervoxel.
"""
from __future__ import annotations

import os
import sys

import numpy as np


def _params(path="param.txt"):
    with open(path) as f:
        off = [int(v) for v in f.readline().split()]
        dim = [int(v) for v in f.readline().split()]
    return off, dim


def _load_chunkmap(path="chunkmap.data"):
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return {}
    a = np.fromfile(path, dtype=np.uint64)
    return dict(zip(a[0::2].tolist(), a[1::2].tolist()))


def main() -> int:
    if os.environ.get("ABISS_NUC_SNAP", "0") == "0":
        return 0
    if not os.path.exists("nuc.raw"):
        return 0

    _, dim = _params()
    n = dim[0] * dim[1] * dim[2]
    seg = np.memmap("seg.raw", dtype=np.uint64, mode="r", shape=(n,))
    nuc = np.memmap("nuc.raw", dtype=np.uint32, mode="r", shape=(n,))

    floor = int(os.environ.get("ABISS_NUC_MIN_TAGGED", "50"))
    ratio = float(os.environ.get("ABISS_NUC_DOMINANCE", "0.6"))

    sel = (nuc != 0) & (seg != 0)
    if not sel.any():
        print("nuc: snap 0 groups (no tagged voxels)")
        return 0

    # per (supervoxel, nucleus) voxel counts
    sv = np.asarray(seg[sel], dtype=np.uint64)
    nid = np.asarray(nuc[sel], dtype=np.uint64)
    key = np.stack([sv, nid], 1)
    uniq, cnt = np.unique(key, axis=0, return_counts=True)

    # total tagged voxels per supervoxel, and its winning nucleus
    order = np.lexsort((-cnt, uniq[:, 0]))
    uniq, cnt = uniq[order], cnt[order]
    svs, starts = np.unique(uniq[:, 0], return_index=True)
    totals = np.add.reduceat(cnt, starts)
    win_id, win_cnt = uniq[starts, 1], cnt[starts]      # first row per sv = largest count

    keep = (totals >= floor) & (win_cnt * 1000 >= int(round(ratio * 1000)) * totals)
    svs, win_id = svs[keep], win_id[keep]

    cmap = _load_chunkmap()

    def resolve(s):
        return cmap.get(s, s)

    groups: dict[int, set] = {}
    for s, w in zip(svs.tolist(), win_id.tolist()):
        groups.setdefault(w, set()).add(resolve(s))

    merged = 0
    for w, targets in groups.items():
        if len(targets) < 2:
            continue
        rep = min(targets)
        for t in targets:
            if t != rep:
                cmap[t] = rep
                merged += 1

    # chunkmap lookup is ONE level (Utils.hpp:63) -- flatten so no key points at a
    # key that was itself just remapped.
    def final(v, _seen=None):
        while v in cmap and cmap[v] != v:
            v = cmap[v]
        return v

    for k in list(cmap):
        cmap[k] = final(cmap[k])

    out = np.empty(len(cmap) * 2, dtype=np.uint64)
    out[0::2] = np.fromiter(cmap.keys(), dtype=np.uint64, count=len(cmap))
    out[1::2] = np.fromiter(cmap.values(), dtype=np.uint64, count=len(cmap))
    out.tofile("chunkmap.data")

    snapped = sum(1 for _, t in groups.items() if len(t) >= 2)
    print(f"nuc: snap {snapped} nuclei, {merged} supervoxels must-linked, "
          f"{len(cmap)} chunkmap entries")
    return 0


if __name__ == "__main__":
    sys.exit(main())
