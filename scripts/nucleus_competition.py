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
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np

NEW_ID_BASE = 1 << 60
ABISS_NATIVE_ID_LIMIT = 1 << 57


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


def _sample_scalar(volume, coords_xyz, block_xyz=(256, 256, 64)):
    """Sample integer coordinates without reading their full bounding cuboid."""

    coords = np.asarray(coords_xyz, dtype=np.int64)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError("sample coordinates must have shape (N, 3)")
    if coords.shape[0] == 0:
        return np.empty((0,), dtype=volume.dtype)
    origin, stop = _volume_bounds_xyz(volume)
    origin_a = np.asarray(origin, dtype=np.int64)
    stop_a = np.asarray(stop, dtype=np.int64)
    if np.any(coords < origin_a) or np.any(coords >= stop_a):
        raise ValueError("sample coordinates lie outside the watershed volume")

    block = np.asarray(block_xyz, dtype=np.int64)
    grid = (coords - origin_a) // block
    grid_shape = np.maximum(1, (stop_a - origin_a + block - 1) // block)
    keys = (grid[:, 0] * grid_shape[1] + grid[:, 1]) * grid_shape[2] + grid[:, 2]
    order = np.argsort(keys, kind="stable")
    sorted_keys = keys[order]
    boundaries = np.flatnonzero(np.diff(sorted_keys)) + 1
    groups = np.split(order, boundaries)
    output = np.empty(coords.shape[0], dtype=volume.dtype)
    for indices in groups:
        point = coords[indices[0]]
        start = origin_a + ((point - origin_a) // block) * block
        end = np.minimum(start + block, stop_a)
        box = tuple(start.tolist() + end.tolist())
        data = _read_scalar(volume, box)
        local = coords[indices] - start
        output[indices] = data[tuple(local.T)]
    return output


def nucleus_segment_histograms(nucleus, watershed, stats, ratio_zyx, offset_zyx):
    ratio_xyz = np.asarray(tuple(reversed(ratio_zyx)), dtype=np.int64)
    offset_xyz = np.asarray(tuple(reversed(offset_zyx)), dtype=np.int64)
    ws_start, ws_stop = (np.asarray(v, dtype=np.int64) for v in _volume_bounds_xyz(watershed))
    histograms = {}
    for nucleus_id in sorted(stats):
        record = stats[nucleus_id]
        start = np.asarray(record["start_xyz"], dtype=np.int64)
        stop = np.asarray(record["stop_xyz"], dtype=np.int64)
        data = _read_scalar(nucleus, tuple(start.tolist() + stop.tolist()))
        local = np.argwhere(data == nucleus_id)
        source_xyz = local + start
        high_xyz = (source_xyz - offset_xyz) * ratio_xyz
        valid = np.all((high_xyz >= ws_start) & (high_xyz < ws_stop), axis=1)
        high_xyz = high_xyz[valid]
        labels = _sample_scalar(watershed, high_xyz)
        ids, counts = np.unique(labels, return_counts=True)
        histograms[nucleus_id] = {
            int(seg_id): int(count)
            for seg_id, count in zip(ids.tolist(), counts.tolist())
            if int(seg_id) != 0
        }
    return histograms


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


def _stable_new_id(parent_id, anchor_id):
    payload = f"{int(parent_id)}:{int(anchor_id)}".encode("ascii")
    return NEW_ID_BASE + int.from_bytes(hashlib.sha256(payload).digest()[:7], "big")


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
    keeper = min(anchors, key=lambda anchor_id: (-counts[anchor_id], anchor_id))
    anchor_labels = {
        anchor_id: (parent_id if anchor_id == keeper else _stable_new_id(parent_id, anchor_id))
        for anchor_id in anchors
    }
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


def run(params):
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

    from volume_backends import open_volume

    affinity = open_volume(
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

    ws_start, ws_stop = _volume_bounds_xyz(watershed)
    bounds = _intersect_box(bbox_xyz, tuple(ws_start) + tuple(ws_stop))
    if bounds is None:
        raise ValueError("BBOX does not intersect WS_PATH")
    print("nucleus competition: scan instance geometry", flush=True)
    stats = scan_nucleus_geometry(
        nucleus,
        bounds,
        ratio_zyx,
        offset_zyx,
        block_z=int(params.get("NUC_SCAN_BLOCK_Z", 16)),
    )
    print(f"nucleus competition: map {len(stats)} nuclei to watershed ids", flush=True)
    histograms = nucleus_segment_histograms(nucleus, watershed, stats, ratio_zyx, offset_zyx)
    targets, shares = qualifying_targets(histograms, stats, min_share)
    units, bridges = contact_units(
        targets,
        stats,
        ratio_zyx,
        offset_zyx,
        voxel_zyx,
        contact_um,
    )
    print(
        f"nucleus competition: {len(targets)} multi-nucleus watershed ids -> "
        f"{len(units)} contact units; {len(bridges)} bridge cases left untouched",
        flush=True,
    )

    manifest_path = _manifest_path(params)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    repairs = []
    used_ids = set()
    used_boxes = collections.defaultdict(list)
    for index, unit in enumerate(units):
        parent_id = int(unit["parent_id"])
        if parent_id >= ABISS_NATIVE_ID_LIMIT:
            raise RuntimeError(
                f"watershed id {parent_id} is outside ABISS's native id namespace; "
                "competitive labels cannot be proven collision-free"
            )
        candidate = _unit_candidate_box(unit, stats, ratio_zyx, offset_zyx, margin_zyx, bounds)
        repair_box = _find_parent_box(watershed, parent_id, candidate, block_xyz, factor)
        if repair_box is None:
            raise RuntimeError(f"parent {parent_id} disappeared before competitive growth")
        for existing in used_boxes[parent_id]:
            if _intersect_box(repair_box, existing) is not None:
                raise RuntimeError(
                    f"competitive scopes overlap for parent {parent_id}; increase NUC_CONTACT_UM "
                    "so they form one contact unit or reduce NUC_COMPETITION_MARGIN_UM"
                )
        used_boxes[parent_id].append(repair_box)
        print(
            f"nucleus competition: flood {index + 1}/{len(units)} parent {parent_id} "
            f"anchors {list(unit['anchor_ids'])} box {repair_box}",
            flush=True,
        )
        territory, counts, anchor_labels = flood_unit(
            affinity,
            watershed,
            nucleus,
            unit,
            repair_box,
            ratio_zyx,
            offset_zyx,
            factor,
            affinity_channels,
            slab_z,
        )
        for label in anchor_labels.values():
            if label != parent_id and (label < NEW_ID_BASE or label in used_ids):
                raise RuntimeError(f"generated competitive label collision: {label}")
            used_ids.add(label)
        name = f"territory_{index:05d}_{parent_id}.npz"
        np.savez_compressed(
            manifest_path.parent / name,
            territory=territory,
            bbox_xyz=np.asarray(repair_box, dtype=np.int64),
            factor=np.asarray(factor, dtype=np.int64),
        )
        repairs.append(
            {
                "parent_id": str(parent_id),
                "anchor_ids": [str(v) for v in unit["anchor_ids"]],
                "bbox_xyz": list(repair_box),
                "factor": factor,
                "territory_file": name,
                "marker_labels": {
                    str(marker): str(anchor_labels[anchor_id])
                    for marker, anchor_id in enumerate(unit["anchor_ids"], start=1)
                },
                "anchor_labels": {
                    str(anchor_id): str(label) for anchor_id, label in anchor_labels.items()
                },
                "pooled_voxels": {
                    str(anchor_id): int(count) for anchor_id, count in counts.items()
                },
                "min_gap_um": unit["min_gap_um"],
                "max_gap_um": unit["max_gap_um"],
            }
        )

    manifest = {
        "schema_version": "1.0",
        "manifest_type": "abiss_nucleus_competition",
        "coordinate_order": "xyz",
        "base_watershed": str(params["WS_PATH"]),
        "nucleus_instances": str(params["NUC_PATH"]),
        "bbox_xyz": list(bounds),
        "ratio_zyx": list(ratio_zyx),
        "offset_zyx": list(offset_zyx),
        "voxel_size_zyx_nm": list(voxel_zyx),
        "min_nucleus_share": min_share,
        "contact_um": contact_um,
        "margin_um": margin_um,
        "margin_zyx": list(margin_zyx),
        "pooling_factor": factor,
        "multi_nucleus_watershed_ids": len(targets),
        "repairs": repairs,
        "bridges_left_untouched": bridges,
        "target_shares": {
            str(segment_id): {str(nucleus_id): share for nucleus_id, share in values.items()}
            for segment_id, values in shares.items()
            if len(values) >= 2
        },
    }
    temporary = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, manifest_path)
    print(
        f"nucleus competition: wrote {manifest_path} ({len(repairs)} repairs)",
        flush=True,
    )
    return manifest


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("param", type=Path, help="ABISS pipeline JSON")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    os.environ["PARAM_JSON"] = str(args.param.resolve())
    with args.param.open() as handle:
        params = json.load(handle)
    run(params)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
