"""Apply sparse competitive-growth territories to an ABISS watershed cutout."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def local_manifest_path(value):
    text = str(value)
    if text.startswith("file://"):
        from urllib.parse import unquote
        from urllib.request import url2pathname

        text = url2pathname(unquote(text[len("file://") :]))
    if "://" in text:
        raise ValueError("NUC_COMPETITION_MANIFEST currently requires a shared local path")
    return Path(text)


def intersect_box(first, second):
    start = [max(int(first[i]), int(second[i])) for i in range(3)]
    stop = [min(int(first[i + 3]), int(second[i + 3])) for i in range(3)]
    if any(stop[i] <= start[i] for i in range(3)):
        return None
    return start + stop


def apply_nucleus_competition(seg_cutout, chunk_start_xyz, global_params):
    """Overlay globally flooded watershed territories on one XYZC cutout."""

    value = global_params.get("NUC_COMPETITION_MANIFEST")
    if not value:
        return seg_cutout
    manifest_path = local_manifest_path(value)
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"nucleus competition manifest is missing: {manifest_path}; "
            "run the competitive_nucleus_growth stage before agglomeration"
        )
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("manifest_type") != "abiss_nucleus_competition":
        raise ValueError(f"unexpected nucleus competition manifest: {manifest_path}")
    if str(manifest.get("base_watershed")) != str(global_params["WS_PATH"]):
        raise ValueError("nucleus competition manifest was built from a different WS_PATH")
    if seg_cutout.ndim != 4 or seg_cutout.shape[3] != 1:
        raise ValueError(f"watershed cutout must be single-channel XYZC, got {seg_cutout.shape}")

    chunk_start = [int(v) for v in chunk_start_xyz]
    chunk_stop = [chunk_start[i] + int(seg_cutout.shape[i]) for i in range(3)]
    chunk_box = chunk_start + chunk_stop
    labels = seg_cutout[..., 0]
    changed = 0
    for repair in manifest.get("repairs", []):
        repair_box = [int(v) for v in repair["bbox_xyz"]]
        overlap = intersect_box(chunk_box, repair_box)
        if overlap is None:
            continue
        factor = int(repair["factor"])
        if factor < 1:
            raise ValueError("nucleus competition factor must be positive")
        territory_path = manifest_path.parent / repair["territory_file"]
        with np.load(territory_path, allow_pickle=False) as archive:
            territory = np.asarray(archive["territory"])
            stored_box = [int(v) for v in archive["bbox_xyz"]]
            stored_factor = int(archive["factor"])
        if stored_box != repair_box or stored_factor != factor:
            raise ValueError(f"territory metadata does not match manifest: {territory_path}")

        local = tuple(
            slice(overlap[i] - chunk_start[i], overlap[i + 3] - chunk_start[i]) for i in range(3)
        )
        pooled_indices = [
            (np.arange(overlap[i], overlap[i + 3], dtype=np.int64) - repair_box[i]) // factor
            for i in range(3)
        ]
        pooled = territory[np.ix_(pooled_indices[0], pooled_indices[1], pooled_indices[2])]
        parent_id = int(repair["parent_id"])
        block = labels[local]
        for marker_text, label_text in repair["marker_labels"].items():
            new_label = int(label_text)
            if new_label == parent_id:
                continue
            select = (block == parent_id) & (pooled == int(marker_text))
            changed += int(select.sum())
            block[select] = new_label
    print(f"nucleus competition: overlaid {changed} voxels in this RAG chunk")
    return seg_cutout


__all__ = ["apply_nucleus_competition", "intersect_box", "local_manifest_path"]
