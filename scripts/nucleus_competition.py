#!/usr/bin/env python3
"""Split watershed objects that contain touching nucleus instances.

This is the global stage between ``remap_watershed`` and mean-edge
agglomeration.  It does not rewrite the watershed volume.  Instead it writes a
small manifest plus pooled territory arrays.  ``cut_chunk_agg.py`` overlays
those deterministic labels while constructing each atomic RAG.

Only contact groups are repaired.  Nuclei that share a watershed id but are
far apart are reported as likely neurite bridges and are not used to draw an
arbitrary cut through neuropil.
"""

from __future__ import annotations

import argparse
import collections
import contextlib
import datetime as dt
import hashlib
import json
import math
import os
import re
import sys
import time
from pathlib import Path

import numpy as np

# isort: off
from nucleus_overlay import (
    PUBLICATION_SCHEMA_VERSION,
    build_publication_ledger,
    identity_declaration,
    minted_nucleus_id,
    validate_publication_identity,
    validate_required_capabilities,
)
# isort: on

NEW_ID_BASE = 1 << 60
ABISS_NATIVE_ID_LIMIT = 1 << 57
DEFAULT_MAX_UNITS = 64
RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def _axis_vector(params, name, default, *, positive=False, number_type=int):
    value = params.get(name, default)
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"{name} must be a three-element [z,y,x] list")
    output = []
    for item in value:
        if isinstance(item, bool):
            raise ValueError(f"{name} cannot contain booleans")
        try:
            converted = number_type(item)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} contains a non-numeric value: {item!r}") from exc
        if positive and converted <= 0:
            raise ValueError(f"{name} values must all be positive")
        output.append(converted)
    return tuple(output)


def _volume_bounds_xyz(volume):
    bounds = getattr(volume, "bounds", None)
    if bounds is not None:
        return tuple(int(v) for v in bounds.minpt), tuple(int(v) for v in bounds.maxpt)
    offset = getattr(volume, "voxel_offset", None)
    start = tuple(int(v) for v in offset[:3]) if offset is not None else (0, 0, 0)
    return start, tuple(start[i] + int(volume.shape[i]) for i in range(3))


def _intersect_box(first, second):
    start = tuple(max(int(first[i]), int(second[i])) for i in range(3))
    stop = tuple(min(int(first[i + 3]), int(second[i + 3])) for i in range(3))
    if any(stop[i] <= start[i] for i in range(3)):
        return None
    return start + stop


def _read_scalar(volume, box_xyz):
    x0, y0, z0, x1, y1, z1 = (int(v) for v in box_xyz)
    block = np.asarray(volume[x0:x1, y0:y1, z0:z1])
    if block.ndim == 4:
        if block.shape[3] != 1:
            raise ValueError(f"expected a scalar volume, got cutout shape {block.shape}")
        block = block[..., 0]
    if block.ndim != 3:
        raise ValueError(f"expected an XYZ scalar cutout, got shape {block.shape}")
    return block


def _iter_boxes(box_xyz, block_xyz):
    x0, y0, z0, x1, y1, z1 = (int(v) for v in box_xyz)
    bx, by, bz = (int(v) for v in block_xyz)
    for z in range(z0, z1, bz):
        for y in range(y0, y1, by):
            for x in range(x0, x1, bx):
                yield (x, y, z, min(x + bx, x1), min(y + by, y1), min(z + bz, z1))


def _source_box_for_high_box(high_box_xyz, ratio_xyz, offset_xyz):
    low = []
    high = []
    for axis in range(3):
        start = int(high_box_xyz[axis])
        stop = int(high_box_xyz[axis + 3])
        ratio = int(ratio_xyz[axis])
        offset = int(offset_xyz[axis])
        low.append((start + ratio // 2) // ratio + offset)
        high.append((stop - 1 + ratio // 2) // ratio + offset + 1)
    return tuple(low + high)


def scan_nucleus_geometry(nucleus, high_box_xyz, ratio_zyx, offset_zyx, block_z=16):
    """Return count, centroid sum, and source-grid bbox for every nucleus id."""

    ratio_xyz = tuple(reversed(ratio_zyx))
    offset_xyz = tuple(reversed(offset_zyx))
    source_box = _source_box_for_high_box(high_box_xyz, ratio_xyz, offset_xyz)
    nuc_start, nuc_stop = _volume_bounds_xyz(nucleus)
    volume_box = tuple(nuc_start) + tuple(nuc_stop)
    source_box = _intersect_box(source_box, volume_box)
    if source_box is None:
        raise ValueError("NUC_RATIO/NUC_OFFSET map BBOX outside the nucleus volume")

    stats = {}
    x0, y0, z0, x1, y1, z1 = source_box
    for z in range(z0, z1, int(block_z)):
        box = (x0, y0, z, x1, y1, min(z + int(block_z), z1))
        data = _read_scalar(nucleus, box)
        if not np.issubdtype(data.dtype, np.integer):
            raise TypeError(f"nucleus instances must be integer-valued, got {data.dtype}")
        labels = np.unique(data)
        for raw_label in labels.tolist():
            label = int(raw_label)
            if label == 0:
                continue
            local = np.argwhere(data == raw_label)  # XYZ
            global_xyz = local + np.asarray(box[:3], dtype=np.int64)
            record = stats.setdefault(
                label,
                {
                    "count": 0,
                    "sum_xyz": np.zeros(3, dtype=np.float64),
                    "start_xyz": np.full(3, np.iinfo(np.int64).max, dtype=np.int64),
                    "stop_xyz": np.full(3, np.iinfo(np.int64).min, dtype=np.int64),
                },
            )
            record["count"] += int(global_xyz.shape[0])
            record["sum_xyz"] += global_xyz.sum(axis=0, dtype=np.float64)
            record["start_xyz"] = np.minimum(record["start_xyz"], global_xyz.min(axis=0))
            record["stop_xyz"] = np.maximum(record["stop_xyz"], global_xyz.max(axis=0) + 1)
    return stats


def nucleus_segment_histograms(
    nucleus,
    watershed,
    stats,
    ratio_zyx,
    offset_zyx,
    high_block_xyz=(256, 256, 64),
):
    """Count watershed support over the exact nearest-neighbour nucleus expansion."""

    ratio_xyz = np.asarray(tuple(reversed(ratio_zyx)), dtype=np.int64)
    offset_xyz = np.asarray(tuple(reversed(offset_zyx)), dtype=np.int64)
    ws_start, ws_stop = _volume_bounds_xyz(watershed)
    high_box = tuple(ws_start) + tuple(ws_stop)
    source_box = _source_box_for_high_box(
        high_box,
        tuple(int(value) for value in ratio_xyz),
        tuple(int(value) for value in offset_xyz),
    )
    nuc_start, nuc_stop = _volume_bounds_xyz(nucleus)
    source_box = _intersect_box(source_box, tuple(nuc_start) + tuple(nuc_stop))
    if source_box is None:
        raise ValueError("NUC_RATIO/NUC_OFFSET map the watershed outside the nucleus volume")

    high_block = np.asarray(high_block_xyz, dtype=np.int64)
    if np.any(high_block <= 0):
        raise ValueError("high_block_xyz values must be positive")
    source_block = tuple(max(1, int(high_block[axis]) // int(ratio_xyz[axis])) for axis in range(3))
    counts = {int(nucleus_id): collections.defaultdict(int) for nucleus_id in stats}
    ws_start_a = np.asarray(ws_start, dtype=np.int64)
    ws_stop_a = np.asarray(ws_stop, dtype=np.int64)
    half = ratio_xyz // 2
    for block in _iter_boxes(source_box, source_block):
        low = _read_scalar(nucleus, block)
        labels = np.unique(low)
        labels = labels[labels != 0]
        if labels.size == 0:
            continue

        source_start = np.asarray(block[:3], dtype=np.int64)
        source_stop = np.asarray(block[3:], dtype=np.int64)
        high_start = (source_start - offset_xyz) * ratio_xyz - half
        high_stop = (source_stop - offset_xyz) * ratio_xyz - half
        high_start = np.maximum(high_start, ws_start_a)
        high_stop = np.minimum(high_stop, ws_stop_a)
        if np.any(high_stop <= high_start):
            continue
        block_high = tuple(high_start.tolist() + high_stop.tolist())
        aligned = _aligned_nucleus_xyz(
            nucleus,
            block_high,
            ratio_zyx,
            offset_zyx,
        )
        segments = _read_scalar(watershed, block_high)
        for raw_nucleus_id in labels.tolist():
            nucleus_id = int(raw_nucleus_id)
            if nucleus_id not in counts:
                continue
            selected = segments[aligned == raw_nucleus_id]
            segment_ids, segment_counts = np.unique(selected, return_counts=True)
            for segment_id, count in zip(segment_ids.tolist(), segment_counts.tolist()):
                if int(segment_id) != 0:
                    counts[nucleus_id][int(segment_id)] += int(count)

    tags_per_source_voxel = int(np.prod(ratio_xyz))
    return {
        nucleus_id: {
            segment_id: float(count) / tags_per_source_voxel
            for segment_id, count in sorted(histogram.items())
        }
        for nucleus_id, histogram in sorted(counts.items())
    }


def qualifying_targets(histograms, stats, min_share):
    per_segment = collections.defaultdict(dict)
    for nucleus_id, histogram in histograms.items():
        total = int(stats[nucleus_id]["count"])
        for segment_id, count in histogram.items():
            share = count / max(total, 1)
            if share >= min_share:
                per_segment[int(segment_id)][int(nucleus_id)] = share
    return {
        segment_id: tuple(sorted(shares))
        for segment_id, shares in per_segment.items()
        if len(shares) >= 2
    }, per_segment


def _nucleus_shape(stats, ratio_zyx, offset_zyx, voxel_size_zyx_nm):
    ratio_xyz = np.asarray(tuple(reversed(ratio_zyx)), dtype=np.float64)
    offset_xyz = np.asarray(tuple(reversed(offset_zyx)), dtype=np.float64)
    voxel_xyz = np.asarray(tuple(reversed(voxel_size_zyx_nm)), dtype=np.float64)
    mask_voxel_nm3 = float(np.prod(ratio_xyz * voxel_xyz))
    centers = {}
    radii = {}
    for nucleus_id, record in stats.items():
        center_source = record["sum_xyz"] / max(int(record["count"]), 1)
        center_high = (center_source - offset_xyz) * ratio_xyz
        centers[nucleus_id] = center_high * voxel_xyz
        radii[nucleus_id] = (3.0 * int(record["count"]) * mask_voxel_nm3 / (4.0 * math.pi)) ** (
            1.0 / 3.0
        )
    return centers, radii


def contact_units(targets, stats, ratio_zyx, offset_zyx, voxel_size_zyx_nm, contact_um):
    centers, radii = _nucleus_shape(stats, ratio_zyx, offset_zyx, voxel_size_zyx_nm)
    units = []
    bridges = []
    for segment_id, nuclei in sorted(targets.items()):
        parent = {nucleus_id: nucleus_id for nucleus_id in nuclei}

        def find(value):
            while parent[value] != value:
                parent[value] = parent[parent[value]]
                value = parent[value]
            return value

        pairs = []
        for index, left in enumerate(nuclei):
            for right in nuclei[index + 1 :]:
                gap_um = (
                    float(np.linalg.norm(centers[left] - centers[right]))
                    - radii[left]
                    - radii[right]
                ) / 1000.0
                pairs.append((left, right, gap_um))
                if gap_um < contact_um:
                    root_left, root_right = find(left), find(right)
                    if root_left != root_right:
                        parent[root_left] = root_right
        groups = collections.defaultdict(list)
        for nucleus_id in nuclei:
            groups[find(nucleus_id)].append(nucleus_id)
        contact_groups = [tuple(sorted(group)) for group in groups.values() if len(group) >= 2]
        for group in sorted(contact_groups):
            gaps = [gap for left, right, gap in pairs if left in group and right in group]
            units.append(
                {
                    "parent_id": int(segment_id),
                    "anchor_ids": group,
                    "min_gap_um": min(gaps),
                    "max_gap_um": max(gaps),
                }
            )
        if len(groups) > 1:
            cross = [gap for left, right, gap in pairs if find(left) != find(right)]
            bridges.append(
                {
                    "parent_id": str(segment_id),
                    "groups": [
                        list(group) for group in sorted(tuple(sorted(v)) for v in groups.values())
                    ],
                    "min_cross_gap_um": min(cross) if cross else None,
                }
            )
    return units, bridges


def _aligned_nucleus_xyz(nucleus, box_xyz, ratio_zyx, offset_zyx):
    ratio_xyz = tuple(reversed(ratio_zyx))
    offset_xyz = tuple(reversed(offset_zyx))
    indices = []
    nuc_start, nuc_stop = _volume_bounds_xyz(nucleus)
    for axis in range(3):
        target = np.arange(box_xyz[axis], box_xyz[axis + 3], dtype=np.int64)
        source = (target + ratio_xyz[axis] // 2) // ratio_xyz[axis] + offset_xyz[axis]
        if source.size:
            slack = ratio_xyz[axis]
            if source[0] < nuc_start[axis] - slack or source[-1] >= nuc_stop[axis] + slack:
                raise ValueError(
                    f"NUC_RATIO/NUC_OFFSET map axis {axis} to "
                    f"[{source[0]}, {source[-1]}] outside "
                    f"[{nuc_start[axis]}, {nuc_stop[axis]}) by more than one source voxel"
                )
            np.clip(source, nuc_start[axis], nuc_stop[axis] - 1, out=source)
        indices.append(source)
    source_box = tuple(int(values[0]) for values in indices) + tuple(
        int(values[-1]) + 1 for values in indices
    )
    low = _read_scalar(nucleus, source_box)
    relative = tuple(values - values[0] for values in indices)
    return low[np.ix_(*relative)]


def _pool(array, factor, operation, pad_value):
    if factor == 1:
        return np.asarray(array)
    pad = [(0, (-int(size)) % factor) for size in array.shape]
    padded = np.pad(array, pad, mode="constant", constant_values=pad_value)
    x, y, z = padded.shape
    blocks = padded.reshape(
        x // factor,
        factor,
        y // factor,
        factor,
        z // factor,
        factor,
    )
    return operation(blocks, axis=(1, 3, 5))


def _pool_min(array, factor):
    return _pool(array, factor, np.min, 0)


def _pool_max(array, factor):
    return _pool(array, factor, np.max, 0)


def _pool_unanimous(array, factor):
    low = _pool(array, factor, np.min, 0)
    high = _pool(array, factor, np.max, 0)
    return np.where(low == high, low, 0)


def _find_parent_box(watershed, parent_id, candidate_box, block_xyz, factor):
    found_start = np.full(3, np.iinfo(np.int64).max, dtype=np.int64)
    found_stop = np.full(3, np.iinfo(np.int64).min, dtype=np.int64)
    for block_box in _iter_boxes(candidate_box, block_xyz):
        mask = _read_scalar(watershed, block_box) == parent_id
        if not mask.any():
            continue
        coords = np.argwhere(mask) + np.asarray(block_box[:3], dtype=np.int64)
        found_start = np.minimum(found_start, coords.min(axis=0))
        found_stop = np.maximum(found_stop, coords.max(axis=0) + 1)
    if np.any(found_stop <= found_start):
        return None
    candidate_start = np.asarray(candidate_box[:3], dtype=np.int64)
    candidate_stop = np.asarray(candidate_box[3:], dtype=np.int64)
    start = np.maximum(candidate_start, (found_start // factor) * factor)
    stop = np.minimum(candidate_stop, ((found_stop + factor - 1) // factor) * factor)
    return tuple(start.tolist() + stop.tolist())


def _unit_candidate_box(unit, stats, ratio_zyx, offset_zyx, margin_zyx, bounds_xyz):
    ratio_xyz = np.asarray(tuple(reversed(ratio_zyx)), dtype=np.int64)
    offset_xyz = np.asarray(tuple(reversed(offset_zyx)), dtype=np.int64)
    margin_xyz = np.asarray(tuple(reversed(margin_zyx)), dtype=np.int64)
    starts, stops = [], []
    for nucleus_id in unit["anchor_ids"]:
        record = stats[nucleus_id]
        starts.append((np.asarray(record["start_xyz"]) - offset_xyz) * ratio_xyz)
        stops.append((np.asarray(record["stop_xyz"]) - offset_xyz) * ratio_xyz)
    start = np.min(starts, axis=0) - margin_xyz
    stop = np.max(stops, axis=0) + margin_xyz
    bounds_start = np.asarray(bounds_xyz[:3], dtype=np.int64)
    bounds_stop = np.asarray(bounds_xyz[3:], dtype=np.int64)
    start = np.maximum(start, bounds_start)
    stop = np.minimum(stop, bounds_stop)
    return tuple(start.tolist() + stop.tolist())


def _stable_new_id(anchor_id):
    return minted_nucleus_id(anchor_id)


def _stable_territory_id(parent_id, anchor_id):
    payload = f"nucleus-territory:{int(parent_id)}:{int(anchor_id)}".encode("ascii")
    return NEW_ID_BASE + int.from_bytes(hashlib.sha256(payload).digest()[:7], "big")


def qualified_owner_labels(shares, units):
    protected_owners = sorted(
        {int(anchor_id) for unit in units for anchor_id in unit["anchor_ids"]}
    )
    owner_labels = {nucleus_id: _stable_new_id(nucleus_id) for nucleus_id in protected_owners}
    if len(set(owner_labels.values())) != len(owner_labels):
        raise RuntimeError("generated nucleus-owner label collision")

    qualified = {}
    for segment_id, values in sorted(shares.items()):
        labels = {
            int(nucleus_id): owner_labels[int(nucleus_id)]
            for nucleus_id in sorted(values)
            if int(nucleus_id) in owner_labels
        }
        if labels:
            qualified[int(segment_id)] = labels
    return protected_owners, qualified


def flood_unit(
    affinity,
    watershed,
    nucleus,
    unit,
    box_xyz,
    ratio_zyx,
    offset_zyx,
    factor,
    affinity_channels,
    slab_z,
):
    from skimage.segmentation import watershed as seeded_watershed

    parent_id = int(unit["parent_id"])
    anchors = tuple(int(v) for v in unit["anchor_ids"])
    cost_parts = []
    parent_parts = []
    marker_parts = []
    x0, y0, z0, x1, y1, z1 = box_xyz
    step_z = max(int(factor), int(slab_z) // int(factor) * int(factor))
    for z in range(z0, z1, step_z):
        slab = (x0, y0, z, x1, y1, min(z + step_z, z1))
        raw_affinity = np.asarray(affinity[x0:x1, y0:y1, z : slab[5]])
        if raw_affinity.ndim != 4:
            raise ValueError(f"affinity cutout must be XYZC, got {raw_affinity.shape}")
        channels = [int(v) for v in affinity_channels]
        if not channels or min(channels) < 0 or max(channels) >= raw_affinity.shape[3]:
            raise ValueError(
                f"NUC_COMPETITION_AFF_CHANNELS {channels} invalid for "
                f"{raw_affinity.shape[3]} channels"
            )
        selected = raw_affinity[..., channels]
        if np.issubdtype(selected.dtype, np.integer):
            info = np.iinfo(selected.dtype)
            scale = float(info.max - info.min)
            selected = (selected.astype(np.float32) - float(info.min)) / scale
        else:
            selected = selected.astype(np.float32, copy=False)
        similarity = selected.min(axis=3)
        cost_parts.append(1.0 - _pool_min(similarity, factor))
        parent = _read_scalar(watershed, slab) == parent_id
        parent_pooled = _pool_max(parent, factor).astype(bool, copy=False)
        parent_parts.append(parent_pooled)
        aligned = _aligned_nucleus_xyz(nucleus, slab, ratio_zyx, offset_zyx)
        labels = _pool_unanimous(aligned, factor)
        markers = np.zeros(labels.shape, dtype=np.int32)
        for marker, anchor_id in enumerate(anchors, start=1):
            markers[(labels == anchor_id) & parent_pooled] = marker
        marker_parts.append(markers)

    cost = np.concatenate(cost_parts, axis=2)
    parent = np.concatenate(parent_parts, axis=2)
    markers = np.concatenate(marker_parts, axis=2)
    for marker, anchor_id in enumerate(anchors, start=1):
        if not np.any(markers == marker):
            raise ValueError(
                f"nucleus {anchor_id} has no unanimous pooled seed in parent {parent_id}"
            )
    territory = seeded_watershed(cost, markers=markers, mask=parent, connectivity=1)
    counts = {
        anchor_id: int((territory == marker).sum())
        for marker, anchor_id in enumerate(anchors, start=1)
    }
    anchor_labels = {anchor_id: _stable_new_id(anchor_id) for anchor_id in anchors}
    return territory.astype(np.int32, copy=False), counts, anchor_labels


def _manifest_path(params):
    value = params.get("NUC_COMPETITION_MANIFEST")
    if not value:
        raise ValueError("NUC_COMPETITION_MANIFEST is required for competitive growth")
    text = str(value)
    if text.startswith("file://"):
        from urllib.parse import unquote
        from urllib.request import url2pathname

        text = url2pathname(unquote(text[len("file://") :]))
    if "://" in text:
        raise ValueError("NUC_COMPETITION_MANIFEST currently requires a shared local path")
    return Path(text).resolve()


def _local_path(value, name):
    text = str(value)
    if text.startswith("file://"):
        from urllib.parse import unquote
        from urllib.request import url2pathname

        text = url2pathname(unquote(text[len("file://") :]))
    if "://" in text:
        raise ValueError(f"{name} must be a local path for fingerprinting, got {value!r}")
    text = text.split("::", 1)[0]
    return Path(text).resolve()


def _utc_now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _sha256_bytes(payload):
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path, block_size=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            block = handle.read(block_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _json_bytes(payload):
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _atomic_write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = _json_bytes(payload)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("wb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return _sha256_bytes(encoded)


def _atomic_savez(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


class _PhaseTimer:
    def __init__(self):
        self.phases = {}

    @contextlib.contextmanager
    def phase(self, name):
        started = time.perf_counter()
        try:
            yield
        finally:
            self.phases[name] = self.phases.get(name, 0.0) + (time.perf_counter() - started)


def _run_id(value):
    if not value or not RUN_ID_PATTERN.fullmatch(str(value)):
        raise ValueError("NUC_COMPETITION_RUN_ID must match [A-Za-z0-9][A-Za-z0-9_.-]*")
    return str(value)


def _run_dir(params, run_id):
    return _manifest_path(params).parent / ".nuccomp-runs" / _run_id(run_id)


def _python_sources_digest(abiss_home):
    digest = hashlib.sha256()
    sources = sorted((Path(abiss_home) / "scripts").rglob("*.py"))
    if not sources:
        raise RuntimeError(f"no ABISS Python sources found under {abiss_home}")
    for path in sources:
        relative = str(path.relative_to(abiss_home)).encode("utf-8")
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        digest.update(_sha256_file(path).encode("ascii"))
    return digest.hexdigest()


def _abiss_build_id(abiss_home):
    """Use the replay driver's canonical native/runtime build identity helper."""

    repo_scripts = Path(__file__).resolve().parents[3] / "scripts"
    added = str(repo_scripts) not in sys.path
    if added:
        sys.path.insert(0, str(repo_scripts))
    try:
        from run_seuron_provenance import _abiss_build_id as canonical_build_id

        return canonical_build_id(Path(abiss_home))
    finally:
        if added:
            sys.path.remove(str(repo_scripts))


def _store_member_summary(path):
    path = Path(path)
    members = sorted(item for item in path.rglob("*") if item.is_file())
    digest = hashlib.sha256()
    for member in members:
        name = str(member.relative_to(path)).encode("utf-8")
        digest.update(len(name).to_bytes(4, "big"))
        digest.update(name)
        digest.update(int(member.stat().st_size).to_bytes(8, "big"))
    return {"member_count": len(members), "member_sizes_sha256": digest.hexdigest()}


def _nucleus_fingerprint(value):
    path = _local_path(value, "NUC_PATH")
    if path.is_file():
        return {"kind": "file", "sha256": _sha256_file(path), "size": path.stat().st_size}
    if not path.is_dir():
        raise FileNotFoundError(f"nucleus input does not exist: {path}")
    identity = next(
        (
            candidate
            for candidate in (path / "manifest.json", path / "index.json")
            if candidate.is_file()
        ),
        None,
    )
    if identity is None:
        raise ValueError(f"nucleus store lacks manifest.json or index.json: {path}")
    return {
        "kind": "store",
        "identity_file": identity.name,
        "identity_sha256": _sha256_file(identity),
        **_store_member_summary(path),
    }


def _affinity_fingerprint(value):
    path = _local_path(value, "AFF_PATH")
    if path.is_file():
        return {"kind": "file", "sha256": _sha256_file(path), "size": path.stat().st_size}
    if not path.is_dir():
        raise FileNotFoundError(f"affinity input does not exist: {path}")
    candidates = [path / "index.json"]
    if path.name.endswith(".chunks"):
        candidates.append(path.with_name(path.name[: -len(".chunks")] + ".index.json"))
    index = next((candidate for candidate in candidates if candidate.is_file()), None)
    if index is None:
        raise ValueError(f"affinity chunk store lacks its required index.json: {path}")
    chunks = list(path.glob("chunk_*.h5"))
    return {
        "kind": "chunk_store",
        "index_file": str(index),
        "index_sha256": _sha256_file(index),
        "chunk_count": len(chunks),
    }


def _watershed_manifest_path(params):
    explicit = params.get("WS_MANIFEST")
    if explicit:
        path = _local_path(explicit, "WS_MANIFEST")
        if not path.is_file():
            raise FileNotFoundError(f"watershed manifest does not exist: {path}")
        return path
    watershed = _local_path(params["WS_PATH"], "WS_PATH")
    for directory in (watershed, *watershed.parents):
        candidate = directory / "manifest.json"
        if not candidate.is_file():
            continue
        try:
            payload = json.loads(candidate.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if "abiss_build_id" in payload and "provenance_sha" in payload:
            return candidate
    raise ValueError(
        "cannot identify the watershed: set WS_MANIFEST to the authoritative ABISS manifest"
    )


def watershed_fingerprint(params):
    manifest = _watershed_manifest_path(params)
    digest = _sha256_file(manifest)
    expected = params.get("WS_MANIFEST_SHA256")
    if expected is not None and str(expected) != digest:
        raise ValueError(f"watershed manifest digest mismatch: expected {expected}, found {digest}")
    return {"manifest_file": str(manifest), "manifest_sha256": digest}


def input_fingerprints(params, param_path=None):
    abiss_home = Path(__file__).resolve().parents[1]
    if param_path is None:
        param_payload = _json_bytes(params)
    else:
        param_payload = Path(param_path).read_bytes()
    return {
        "param": {"sha256": _sha256_bytes(param_payload)},
        "nucleus": _nucleus_fingerprint(params["NUC_PATH"]),
        "watershed": watershed_fingerprint(params),
        "affinity": _affinity_fingerprint(params["AFF_PATH"]),
        "code": {
            "abiss_build_id": _abiss_build_id(abiss_home),
            "python_sources_sha256": _python_sources_digest(abiss_home),
        },
    }


def _settings(params):
    if not params.get("NUC_PATH"):
        raise ValueError("NUC_PATH is required for competitive growth")
    ratio_zyx = _axis_vector(params, "NUC_RATIO", [1, 1, 1], positive=True)
    offset_zyx = _axis_vector(params, "NUC_OFFSET", [0, 0, 0])
    voxel_zyx = _axis_vector(
        params, "NUC_VOXEL_SIZE_ZYX_NM", None, positive=True, number_type=float
    )
    bbox_xyz = tuple(int(v) for v in params["BBOX"])
    if len(bbox_xyz) != 6:
        raise ValueError("BBOX must contain six XYZ values")
    min_share = float(params.get("NUC_MIN_SHARE", 0.02))
    if not 0.0 < min_share <= 1.0:
        raise ValueError("NUC_MIN_SHARE must lie in (0, 1]")
    contact_um = float(params.get("NUC_CONTACT_UM", 8.0))
    margin_um = float(params.get("NUC_COMPETITION_MARGIN_UM", 5.0))
    factor = int(params.get("NUC_COMPETITION_FACTOR", 4))
    if contact_um <= 0 or margin_um < 0 or factor < 1:
        raise ValueError("contact distance and factor must be positive; margin must be nonnegative")
    if "NUC_COMPETITION_MARGIN_ZYX" in params:
        margin_zyx = _axis_vector(params, "NUC_COMPETITION_MARGIN_ZYX", [0, 0, 0])
        if any(value < 0 for value in margin_zyx):
            raise ValueError("NUC_COMPETITION_MARGIN_ZYX values must be nonnegative")
    else:
        margin_zyx = tuple(
            int(math.ceil(margin_um * 1000.0 / voxel_zyx[axis])) for axis in range(3)
        )
    block_xyz = tuple(
        reversed(_axis_vector(params, "NUC_COMPETITION_BLOCK_ZYX", [64, 256, 256], positive=True))
    )
    slab_z = int(params.get("NUC_COMPETITION_SLAB_Z", 64))
    affinity_channels = params.get("NUC_COMPETITION_AFF_CHANNELS", [0, 1, 2])
    if isinstance(affinity_channels, (int, str)):
        affinity_channels = [int(affinity_channels)]

    max_units = int(params.get("NUC_MAX_UNITS", DEFAULT_MAX_UNITS))
    if max_units < 1:
        raise ValueError("NUC_MAX_UNITS must be positive")
    return {
        "ratio_zyx": ratio_zyx,
        "offset_zyx": offset_zyx,
        "voxel_zyx": voxel_zyx,
        "bbox_xyz": bbox_xyz,
        "min_share": min_share,
        "contact_um": contact_um,
        "margin_um": margin_um,
        "margin_zyx": margin_zyx,
        "factor": factor,
        "block_xyz": block_xyz,
        "slab_z": slab_z,
        "affinity_channels": tuple(int(value) for value in affinity_channels),
        "max_units": max_units,
    }


def _open_inputs(params, *, affinity=True):
    from volume_backends import open_volume

    affinity_volume = None
    if affinity:
        affinity_volume = open_volume(
            params["AFF_PATH"],
            mip=params.get("AFF_RESOLUTION", 0),
            fill_missing=params.get("AFF_FILL_MISSING", False),
        )
    watershed = open_volume(
        params["WS_PATH"],
        mip=params.get("AFF_RESOLUTION", 0),
        fill_missing=params.get("WS_FILL_MISSING", False),
    )
    nucleus = open_volume(
        params["NUC_PATH"],
        mip=params.get("NUC_MIP", params.get("AFF_RESOLUTION", 0)),
        fill_missing=params.get("NUC_FILL_MISSING", False),
    )
    return affinity_volume, watershed, nucleus


def _prepare_units(units, stats, shares, watershed, settings, bounds):
    protected_owners, qualified_segment_labels = qualified_owner_labels(shares, units)
    prepared = []
    used_boxes = collections.defaultdict(list)
    owner_ids = {
        int(nucleus_id): int(label)
        for values in qualified_segment_labels.values()
        for nucleus_id, label in values.items()
    }
    if len(set(owner_ids.values())) != len(owner_ids):
        raise RuntimeError("generated nucleus-owner label collision")
    generated = set(owner_ids.values())

    for index, unit in enumerate(units):
        parent_id = int(unit["parent_id"])
        if parent_id >= ABISS_NATIVE_ID_LIMIT:
            raise RuntimeError(
                f"watershed id {parent_id} is outside ABISS's native id namespace; "
                "competitive labels cannot be proven collision-free"
            )
        candidate = _unit_candidate_box(
            unit,
            stats,
            settings["ratio_zyx"],
            settings["offset_zyx"],
            settings["margin_zyx"],
            bounds,
        )
        repair_box = _find_parent_box(
            watershed,
            parent_id,
            candidate,
            settings["block_xyz"],
            settings["factor"],
        )
        if repair_box is None:
            raise RuntimeError(f"parent {parent_id} disappeared before competitive growth")
        for existing in used_boxes[parent_id]:
            if _intersect_box(repair_box, existing) is not None:
                raise RuntimeError(
                    f"competitive scopes overlap for parent {parent_id}; increase NUC_CONTACT_UM "
                    "so they form one contact unit or reduce NUC_COMPETITION_MARGIN_UM"
                )
        used_boxes[parent_id].append(repair_box)

        territories = []
        for anchor_id in unit["anchor_ids"]:
            internal_id = _stable_territory_id(parent_id, anchor_id)
            if internal_id in generated:
                raise RuntimeError("generated competitive territory id collision")
            generated.add(internal_id)
            territories.append(
                {
                    "anchor_id": str(int(anchor_id)),
                    "internal_territory_id": str(internal_id),
                }
            )
        prepared.append(
            {
                "index": index,
                "parent_id": str(parent_id),
                "anchor_ids": [str(int(value)) for value in unit["anchor_ids"]],
                "bbox_xyz": list(repair_box),
                "factor": settings["factor"],
                "territories": territories,
                "min_gap_um": float(unit["min_gap_um"]),
                "max_gap_um": float(unit["max_gap_um"]),
                "separation_claim": "local_only",
            }
        )
    return protected_owners, qualified_segment_labels, prepared


def scan_stage(params, param_path, run_id):
    settings = _settings(params)
    run_dir = _run_dir(params, run_id)
    run_dir.mkdir(parents=True, exist_ok=False)
    timer = _PhaseTimer()

    with timer.phase("fingerprint_inputs"):
        fingerprints = input_fingerprints(params, param_path)
    with timer.phase("open_inputs"):
        _affinity, watershed, nucleus = _open_inputs(params, affinity=False)

    ws_start, ws_stop = _volume_bounds_xyz(watershed)
    bounds = _intersect_box(settings["bbox_xyz"], tuple(ws_start) + tuple(ws_stop))
    if bounds is None:
        raise ValueError("BBOX does not intersect WS_PATH")
    print(f"[{_utc_now()}] nucleus competition: scan instance geometry", flush=True)
    with timer.phase("scan_geometry"):
        stats = scan_nucleus_geometry(
            nucleus,
            bounds,
            settings["ratio_zyx"],
            settings["offset_zyx"],
            block_z=int(params.get("NUC_SCAN_BLOCK_Z", 16)),
        )
    print(
        f"[{_utc_now()}] nucleus competition: map {len(stats)} nuclei to watershed ids",
        flush=True,
    )
    with timer.phase("map_to_watershed"):
        histograms = nucleus_segment_histograms(
            nucleus,
            watershed,
            stats,
            settings["ratio_zyx"],
            settings["offset_zyx"],
        )
    with timer.phase("contact_detection"):
        targets, shares = qualifying_targets(histograms, stats, settings["min_share"])
        units, bridges = contact_units(
            targets,
            stats,
            settings["ratio_zyx"],
            settings["offset_zyx"],
            settings["voxel_zyx"],
            settings["contact_um"],
        )
    print(
        f"[{_utc_now()}] nucleus competition: {len(targets)} multi-nucleus watershed ids -> "
        f"{len(units)} contact units; {len(bridges)} bridge cases left untouched",
        flush=True,
    )
    print(
        f"[{_utc_now()}] nucleus competition: array capacity "
        f"{len(units)}/{settings['max_units']} units observed/configured",
        flush=True,
    )
    if len(units) > settings["max_units"]:
        raise RuntimeError(
            f"scan found {len(units)} units, exceeding NUC_MAX_UNITS={settings['max_units']}"
        )
    with timer.phase("plan_units"):
        protected_owners, qualified_segment_labels, prepared_units = _prepare_units(
            units, stats, shares, watershed, settings, bounds
        )
        plan = {
            "schema_version": PUBLICATION_SCHEMA_VERSION,
            "required_capabilities": [],
            "plan_type": "abiss_nucleus_competition_units",
            "run_id": _run_id(run_id),
            "created_at": _utc_now(),
            "max_units": settings["max_units"],
            "fingerprints": fingerprints,
            "base_watershed": str(params["WS_PATH"]),
            "nucleus_instances": str(params["NUC_PATH"]),
            "bbox_xyz": list(bounds),
            "ratio_zyx": list(settings["ratio_zyx"]),
            "offset_zyx": list(settings["offset_zyx"]),
            "voxel_size_zyx_nm": list(settings["voxel_zyx"]),
            "min_nucleus_share": settings["min_share"],
            "contact_um": settings["contact_um"],
            "margin_um": settings["margin_um"],
            "margin_zyx": list(settings["margin_zyx"]),
            "pooling_factor": settings["factor"],
            "affinity_channels": list(settings["affinity_channels"]),
            "slab_z": settings["slab_z"],
            "nucleus_histogram_samples_per_voxel": int(np.prod(settings["ratio_zyx"])),
            "multi_nucleus_watershed_ids": len(targets),
            "protected_nucleus_owners": protected_owners,
            "identity": identity_declaration(),
            "units": prepared_units,
            "bridges_left_untouched": bridges,
            "qualified_segment_owners": {
                str(segment_id): [int(nucleus_id) for nucleus_id in sorted(values)]
                for segment_id, values in sorted(shares.items())
            },
            "qualified_segment_labels": {
                str(segment_id): {
                    str(nucleus_id): str(label) for nucleus_id, label in sorted(values.items())
                }
                for segment_id, values in sorted(qualified_segment_labels.items())
            },
            "qualified_segment_shares": {
                str(segment_id): {
                    str(nucleus_id): float(share) for nucleus_id, share in sorted(values.items())
                }
                for segment_id, values in sorted(shares.items())
            },
            "target_shares": {
                str(segment_id): {
                    str(nucleus_id): float(share) for nucleus_id, share in sorted(values.items())
                }
                for segment_id, values in sorted(shares.items())
                if len(values) >= 2
            },
        }
        plan_path = run_dir / "units.json"
        plan_digest = _atomic_write_json(plan_path, plan)

    scan_report = {
        "status": "complete",
        "stage": "scan",
        "run_id": _run_id(run_id),
        "plan_digest": plan_digest,
        "unit_count": len(prepared_units),
        "unit_capacity": settings["max_units"],
        "phase_seconds": timer.phases,
        "completed_at": _utc_now(),
    }
    _atomic_write_json(run_dir / "scan_report.json", scan_report)
    if not prepared_units:
        print(
            "WARNING nucleus competition: ZERO REPAIRS planned; acceptance must judge "
            "zero_repairs after merge",
            flush=True,
        )
    print(
        f"[{_utc_now()}] nucleus competition: wrote {plan_path} "
        f"({len(prepared_units)} units, digest {plan_digest})",
        flush=True,
    )
    return plan, plan_digest


def _load_plan(params, param_path, run_id):
    path = _run_dir(params, run_id) / "units.json"
    encoded = path.read_bytes()
    plan = json.loads(encoded)
    digest = _sha256_bytes(encoded)
    if plan.get("plan_type") != "abiss_nucleus_competition_units":
        raise ValueError(f"unexpected unit plan: {path}")
    if plan.get("run_id") != _run_id(run_id):
        raise ValueError("unit plan run_id mismatch")
    expected_param = plan.get("fingerprints", {}).get("param", {}).get("sha256")
    actual_param = _sha256_file(param_path)
    if expected_param != actual_param:
        raise ValueError(
            "parameter fingerprint changed after scan: "
            f"expected {expected_param}, found {actual_param}"
        )
    return plan, digest


def flood_stage(params, param_path, run_id, unit_index):
    plan, plan_digest = _load_plan(params, param_path, run_id)
    index = int(unit_index)
    if index < 0 or index >= int(plan["max_units"]):
        raise ValueError(f"unit index {index} is outside [0, {plan['max_units']})")
    run_dir = _run_dir(params, run_id)
    report_path = run_dir / f"flood_{index:05d}.report.json"
    if index >= len(plan["units"]):
        _atomic_write_json(
            report_path,
            {
                "status": "unused",
                "stage": "flood",
                "unit_index": index,
                "plan_digest": plan_digest,
                "elapsed_seconds": 0.0,
                "completed_at": _utc_now(),
            },
        )
        print(f"nucleus competition: unit {index} unused; exiting 0", flush=True)
        return None

    unit = plan["units"][index]
    timer = _PhaseTimer()
    with timer.phase("open_inputs"):
        affinity, watershed, nucleus = _open_inputs(params, affinity=True)
    print(
        f"[{_utc_now()}] nucleus competition: flood unit {index + 1}/{len(plan['units'])} "
        f"parent {unit['parent_id']} anchors {unit['anchor_ids']} box {unit['bbox_xyz']}",
        flush=True,
    )
    flood_input = {
        "parent_id": int(unit["parent_id"]),
        "anchor_ids": tuple(int(value) for value in unit["anchor_ids"]),
    }
    with timer.phase("flood"):
        markers, counts, owner_labels = flood_unit(
            affinity,
            watershed,
            nucleus,
            flood_input,
            tuple(int(value) for value in unit["bbox_xyz"]),
            tuple(int(value) for value in plan["ratio_zyx"]),
            tuple(int(value) for value in plan["offset_zyx"]),
            int(unit["factor"]),
            tuple(int(value) for value in plan["affinity_channels"]),
            int(plan["slab_z"]),
        )
        expected_owner_labels = {
            int(anchor): int(plan["qualified_segment_labels"][unit["parent_id"]][anchor])
            for anchor in unit["anchor_ids"]
        }
        if owner_labels != expected_owner_labels:
            raise RuntimeError("competitive labels disagree with qualified owner labels")
        territory = np.zeros(markers.shape, dtype=np.uint64)
        for marker, item in enumerate(unit["territories"], start=1):
            territory[markers == marker] = int(item["internal_territory_id"])
        actual_ids = {int(value) for value in np.unique(territory) if int(value) != 0}
        expected_ids = {int(item["internal_territory_id"]) for item in unit["territories"]}
        if actual_ids != expected_ids:
            raise RuntimeError(
                f"unit {index} territory ids differ: expected {expected_ids}, found {actual_ids}"
            )

    territory_name = f"terr_{index:05d}_{unit['parent_id']}.npz"
    territory_path = run_dir / territory_name
    _atomic_savez(
        territory_path,
        territory=territory,
        bbox_xyz=np.asarray(unit["bbox_xyz"], dtype=np.int64),
        factor=np.asarray(unit["factor"], dtype=np.int64),
        plan_digest=np.asarray(plan_digest),
    )
    record = {
        "schema_version": PUBLICATION_SCHEMA_VERSION,
        "required_capabilities": [],
        "record_type": "abiss_nucleus_competition_unit",
        "status": "complete",
        "unit_index": index,
        "plan_digest": plan_digest,
        "territory_file": territory_name,
        "territory_sha256": _sha256_file(territory_path),
        "pooled_voxels": {str(anchor): int(count) for anchor, count in sorted(counts.items())},
        "territory_ids": [str(value) for value in sorted(actual_ids)],
        "phase_seconds": timer.phases,
        "completed_at": _utc_now(),
    }
    _atomic_write_json(run_dir / f"unit_{index:05d}.json", record)
    _atomic_write_json(
        report_path,
        {
            "status": "complete",
            "stage": "flood",
            "unit_index": index,
            "plan_digest": plan_digest,
            "elapsed_seconds": sum(timer.phases.values()),
            "phase_seconds": timer.phases,
            "completed_at": _utc_now(),
        },
    )
    return record


def _validated_unit_record(plan, plan_digest, run_dir, unit):
    index = int(unit["index"])
    record_path = run_dir / f"unit_{index:05d}.json"
    record = json.loads(record_path.read_text())
    if record.get("status") != "complete" or record.get("plan_digest") != plan_digest:
        raise ValueError(f"unit {index} is incomplete or belongs to another plan")
    if int(record.get("unit_index", -1)) != index:
        raise ValueError(f"unit record index mismatch: {record_path}")
    territory_path = (run_dir / record["territory_file"]).resolve()
    if territory_path.parent != run_dir.resolve():
        raise ValueError(f"unit {index} territory escapes its run directory")
    if _sha256_file(territory_path) != record.get("territory_sha256"):
        raise ValueError(f"unit {index} territory fingerprint mismatch")
    with np.load(territory_path, allow_pickle=False) as archive:
        territory = np.asarray(archive["territory"])
        stored_box = [int(value) for value in archive["bbox_xyz"]]
        stored_factor = int(archive["factor"])
        stored_digest = str(archive["plan_digest"])
    if stored_box != unit["bbox_xyz"] or stored_factor != int(unit["factor"]):
        raise ValueError(f"unit {index} territory metadata disagrees with units.json")
    if stored_digest != plan_digest:
        raise ValueError(f"unit {index} territory belongs to another plan")
    actual_ids = {int(value) for value in np.unique(territory) if int(value) != 0}
    expected_ids = {int(item["internal_territory_id"]) for item in unit["territories"]}
    record_ids = {int(value) for value in record.get("territory_ids", [])}
    if actual_ids != expected_ids or record_ids != expected_ids:
        raise ValueError(
            f"unit {index} territory-id domain must exactly equal {sorted(expected_ids)}"
        )
    counts = {int(anchor): int(count) for anchor, count in record["pooled_voxels"].items()}
    expected_anchors = {int(value) for value in unit["anchor_ids"]}
    if set(counts) != expected_anchors or any(value <= 0 for value in counts.values()):
        raise ValueError(f"unit {index} has an invalid pooled-voxel count domain")
    return record, territory_path, actual_ids, counts


def _efficiency_recommendation(scan_seconds, flood_seconds):
    geometry = float(scan_seconds.get("scan_geometry", 0.0))
    mapping = float(scan_seconds.get("map_to_watershed", 0.0))
    flood = float(sum(flood_seconds))
    total = geometry + mapping + flood
    fractions = {
        "scan_geometry": geometry / total if total else 0.0,
        "map_to_watershed": mapping / total if total else 0.0,
        "flood": flood / total if total else 0.0,
    }
    dominant = max(fractions, key=fractions.get)
    if dominant == "map_to_watershed" and fractions[dominant] >= 0.4:
        recommendation = "shard_map_to_watershed"
    elif dominant == "scan_geometry" and fractions[dominant] >= 0.4:
        recommendation = "shard_scan_geometry"
    elif dominant == "flood":
        recommendation = "flood_array_addresses_dominant_phase"
    else:
        recommendation = "no_phase_dominates_do_not_build"
    return {"serial_phase_fractions": fractions, "selection": recommendation}


def merge_stage(params, param_path, run_id):
    merge_started = time.perf_counter()
    plan, plan_digest = _load_plan(params, param_path, run_id)
    run_dir = _run_dir(params, run_id)
    manifest_path = _manifest_path(params)
    repairs = []
    all_internal_ids = set()
    emitted_owners = {}
    unit_seconds = []
    flood_compute_seconds = []
    for unit in plan["units"]:
        record, territory_path, actual_ids, counts = _validated_unit_record(
            plan, plan_digest, run_dir, unit
        )
        mappings = []
        for item in unit["territories"]:
            anchor_id = int(item["anchor_id"])
            internal_id = int(item["internal_territory_id"])
            emitted_id = int(plan["qualified_segment_labels"][unit["parent_id"]][str(anchor_id)])
            mappings.append(
                {
                    "anchor_id": str(anchor_id),
                    "internal_territory_id": str(internal_id),
                    "emitted_id": str(emitted_id),
                    "pooled_voxels": counts[anchor_id],
                }
            )
        overlap = all_internal_ids.intersection(actual_ids)
        if overlap:
            raise ValueError(f"territory ids collide across units: {sorted(overlap)}")
        all_internal_ids.update(actual_ids)
        for item in mappings:
            emitted_id = int(item["emitted_id"])
            anchor_id = int(item["anchor_id"])
            if emitted_owners.setdefault(emitted_id, anchor_id) != anchor_id:
                raise ValueError("one emitted territory id maps to multiple nuclei")
        relative_territory = os.path.relpath(territory_path, manifest_path.parent)
        repairs.append(
            {
                **unit,
                "territory_file": relative_territory,
                "territory_sha256": record["territory_sha256"],
                "territory_encoding": "internal_uint64_id",
                "territories": mappings,
            }
        )
        phase_seconds = record.get("phase_seconds", {})
        unit_seconds.append(sum(float(value) for value in phase_seconds.values()))
        flood_compute_seconds.append(float(phase_seconds.get("flood", 0.0)))

    plan_relative = os.path.relpath(run_dir / "units.json", manifest_path.parent)
    fingerprints = dict(plan["fingerprints"])
    fingerprints["units"] = {"sha256": plan_digest}
    manifest = {
        "schema_version": PUBLICATION_SCHEMA_VERSION,
        "required_capabilities": [],
        "manifest_type": "abiss_nucleus_competition",
        "coordinate_order": "xyz",
        "run_id": _run_id(run_id),
        "plan_digest": plan_digest,
        "plan_file": plan_relative,
        "completion": {"state": "complete", "plan_digest": plan_digest},
        "fingerprints": fingerprints,
        "base_watershed": plan["base_watershed"],
        "nucleus_instances": plan["nucleus_instances"],
        "bbox_xyz": plan["bbox_xyz"],
        "ratio_zyx": plan["ratio_zyx"],
        "offset_zyx": plan["offset_zyx"],
        "voxel_size_zyx_nm": plan["voxel_size_zyx_nm"],
        "min_nucleus_share": plan["min_nucleus_share"],
        "contact_um": plan["contact_um"],
        "margin_um": plan["margin_um"],
        "margin_zyx": plan["margin_zyx"],
        "pooling_factor": plan["pooling_factor"],
        "nucleus_histogram_samples_per_voxel": plan["nucleus_histogram_samples_per_voxel"],
        "multi_nucleus_watershed_ids": plan["multi_nucleus_watershed_ids"],
        "protected_nucleus_owners": plan["protected_nucleus_owners"],
        "identity": plan["identity"],
        "repairs": repairs,
        "zero_repairs": len(repairs) == 0,
        "reason": ("competitive_repairs_completed" if repairs else "no_competitive_contact_units"),
        "separation_claim": "local_only" if repairs else "none",
        "bridges_left_untouched": plan["bridges_left_untouched"],
        "qualified_segment_owners": plan["qualified_segment_owners"],
        "qualified_segment_labels": plan["qualified_segment_labels"],
        "qualified_segment_shares": plan["qualified_segment_shares"],
        "target_shares": plan["target_shares"],
        "stage_report_file": os.path.relpath(run_dir / "stage_report.json", manifest_path.parent),
    }
    manifest["ledger"] = build_publication_ledger(
        manifest["repairs"], manifest["qualified_segment_labels"]
    )
    validate_required_capabilities(manifest)
    validate_publication_identity(manifest)

    scan_report = json.loads((run_dir / "scan_report.json").read_text())
    merge_seconds = time.perf_counter() - merge_started
    nonflood_seconds = sum(float(value) for value in scan_report["phase_seconds"].values())
    nonflood_seconds += merge_seconds
    serial_seconds = nonflood_seconds + sum(unit_seconds)
    array_seconds = nonflood_seconds + (max(unit_seconds) if unit_seconds else 0.0)
    report = {
        "schema_version": PUBLICATION_SCHEMA_VERSION,
        "required_capabilities": [],
        "report_type": "abiss_nucleus_competition_stage",
        "status": "complete",
        "run_id": _run_id(run_id),
        "plan_digest": plan_digest,
        "unit_count": len(repairs),
        "unit_capacity": int(plan["max_units"]),
        "zero_repairs": len(repairs) == 0,
        "reason": manifest["reason"],
        "phase_seconds": {
            **scan_report["phase_seconds"],
            "flood_units": unit_seconds,
            "flood_compute_units": flood_compute_seconds,
            "flood_sum": sum(unit_seconds),
            "flood_max": max(unit_seconds) if unit_seconds else 0.0,
            "merge": merge_seconds,
        },
        "critical_path_model": {
            "nonflood_seconds": nonflood_seconds,
            "serial_seconds": serial_seconds,
            "array_seconds_before_scheduler_overhead": array_seconds,
            "predicted_speedup_before_scheduler_overhead": (
                serial_seconds / array_seconds if array_seconds else 1.0
            ),
        },
        "next_optimization": _efficiency_recommendation(scan_report["phase_seconds"], unit_seconds),
        "completed_at": _utc_now(),
    }
    _atomic_write_json(run_dir / "stage_report.json", report)
    _atomic_write_json(manifest_path, manifest)
    if not repairs:
        print(
            "WARNING nucleus competition: ZERO REPAIRS completed; acceptance result "
            "zero_repairs=FAIL for an intervention run",
            flush=True,
        )
    print(
        f"nucleus competition: published {manifest_path} ({len(repairs)} repairs, "
        f"plan {plan_digest})",
        flush=True,
    )
    return manifest


def _record_failure(params, run_id, stage, error, unit_index=None):
    try:
        run_dir = _run_dir(params, run_id)
        run_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "status": "failed",
            "stage": stage,
            "unit_index": unit_index,
            "error_type": type(error).__name__,
            "error": str(error),
            "failed_at": _utc_now(),
            "canonical_manifest_preserved": _manifest_path(params).is_file(),
        }
        _atomic_write_json(run_dir / "stage_report.partial.json", payload)
        _atomic_write_json(
            run_dir / f"failure_{stage}_{unit_index if unit_index is not None else 'stage'}.json",
            payload,
        )
    except Exception as report_error:  # pragma: no cover - preserve the original failure
        print(
            f"could not preserve nucleus competition failure report: {report_error}",
            file=sys.stderr,
        )


def run(params, param_path, run_id=None):
    if run_id is None:
        run_id = f"serial-{dt.datetime.now(dt.timezone.utc):%Y%m%dT%H%M%SZ}-{os.getpid()}"
    plan, _digest = scan_stage(params, param_path, run_id)
    for index in range(len(plan["units"])):
        flood_stage(params, param_path, run_id, index)
    return merge_stage(params, param_path, run_id)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("param", type=Path, help="ABISS pipeline JSON")
    parser.add_argument(
        "action",
        nargs="?",
        choices=("run", "scan", "flood", "merge", "fingerprint"),
        default="run",
    )
    parser.add_argument("--run-id", default=os.environ.get("NUC_COMPETITION_RUN_ID"))
    parser.add_argument(
        "--unit-index",
        type=int,
        default=(
            int(os.environ["SLURM_ARRAY_TASK_ID"])
            if os.environ.get("SLURM_ARRAY_TASK_ID") is not None
            else None
        ),
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    param_path = args.param.resolve()
    os.environ["PARAM_JSON"] = str(param_path)
    with param_path.open() as handle:
        params = json.load(handle)
    if args.action == "fingerprint":
        print(json.dumps(input_fingerprints(params, param_path), indent=2, sort_keys=True))
        return 0
    run_id = args.run_id
    if args.action == "run" and run_id is None:
        run_id = f"serial-{dt.datetime.now(dt.timezone.utc):%Y%m%dT%H%M%SZ}-{os.getpid()}"
    run_id = _run_id(run_id)
    try:
        if args.action == "run":
            run(params, param_path, run_id)
        elif args.action == "scan":
            scan_stage(params, param_path, run_id)
        elif args.action == "flood":
            if args.unit_index is None:
                raise ValueError("flood requires --unit-index or SLURM_ARRAY_TASK_ID")
            flood_stage(params, param_path, run_id, args.unit_index)
        else:
            merge_stage(params, param_path, run_id)
    except Exception as error:
        _record_failure(params, run_id, args.action, error, args.unit_index)
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
