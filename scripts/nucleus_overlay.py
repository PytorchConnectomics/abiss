"""Apply sparse competitive-growth territories to an ABISS watershed cutout."""

from __future__ import annotations

import hashlib
import json
import shlex
from pathlib import Path

import numpy as np

NEW_ID_BASE = 1 << 60
PUBLICATION_SCHEMA_VERSION = "3.0"
IMPLEMENTED_CAPABILITIES: frozenset[str] = frozenset()
MINT_DESCRIPTOR = {
    "scheme": "sha256_prefix_v1",
    "deterministic": True,
    "key_template": "nucleus-owner:{nucleus_id}",
    "digest": "sha256",
    "prefix_bytes": 7,
    "namespace_base": str(NEW_ID_BASE),
    "id_encoding": "decimal_string",
}

_MISSING = object()


def identity_declaration():
    return {
        "scope": "nucleus",
        "parent_disposition": "retired",
        "residue_disposition": "parent_retained",
        "mint": dict(MINT_DESCRIPTOR),
    }


def minted_nucleus_id(nucleus_id, mint=None):
    descriptor = MINT_DESCRIPTOR if mint is None else mint
    if descriptor != MINT_DESCRIPTOR:
        raise ValueError("unsupported nucleus competition mint descriptor")
    nucleus = int(nucleus_id)
    if nucleus <= 0:
        raise ValueError("nucleus ids must be positive")
    key = descriptor["key_template"].format(nucleus_id=nucleus).encode("ascii")
    prefix = int.from_bytes(hashlib.sha256(key).digest()[: descriptor["prefix_bytes"]], "big")
    return int(descriptor["namespace_base"]) + prefix


def _decimal_id(value, field):
    if not isinstance(value, str) or not value.isascii() or not value.isdecimal():
        raise ValueError(f"{field} must be a canonical decimal string")
    parsed = int(value)
    if parsed <= 0 or str(parsed) != value:
        raise ValueError(f"{field} must be a canonical positive decimal string")
    return parsed


def build_publication_ledger(repairs, qualified_segment_labels):
    """Describe every renamed source and every many-to-one owner consolidation."""

    retirements = []
    owner_sources = {}
    minted_ids = set()
    for segment_text, raw_owners in sorted(
        qualified_segment_labels.items(), key=lambda item: int(item[0])
    ):
        for emitted_text in raw_owners.values():
            minted_ids.add(_decimal_id(emitted_text, "qualified emitted id"))
        if len(raw_owners) != 1:
            continue
        source_id = _decimal_id(str(segment_text), "qualified source id")
        nucleus_text, emitted_text = next(iter(raw_owners.items()))
        nucleus_id = _decimal_id(str(nucleus_text), "qualified nucleus id")
        emitted_id = _decimal_id(emitted_text, "qualified emitted id")
        retirements.append(
            {
                "source_id": str(source_id),
                "reason": "owner_canonicalization",
                "scope": "all_source_voxels",
                "emitted_ids": [str(emitted_id)],
            }
        )
        owner_sources.setdefault((nucleus_id, emitted_id), []).append(source_id)

    for repair in repairs:
        parent_id = _decimal_id(repair["parent_id"], "repair parent id")
        emitted_ids = sorted(
            {_decimal_id(item["emitted_id"], "repair emitted id") for item in repair["territories"]}
        )
        retirements.append(
            {
                "source_id": str(parent_id),
                "reason": "competitive_split",
                "scope": "adjudicated_voxels",
                "bbox_xyz": [int(value) for value in repair["bbox_xyz"]],
                "emitted_ids": [str(value) for value in emitted_ids],
                "residue_disposition": "parent_retained",
            }
        )
        minted_ids.update(emitted_ids)

    consolidations = [
        {
            "nucleus_id": str(nucleus_id),
            "emitted_id": str(emitted_id),
            "sources": [str(value) for value in sorted(sources)],
        }
        for (nucleus_id, emitted_id), sources in sorted(owner_sources.items())
    ]
    return {
        "retirements": retirements,
        "consolidations": consolidations,
        "emitted_id_space": {
            "minted_ids": [str(value) for value in sorted(minted_ids)],
            "otherwise": "untouched_base_id",
        },
    }


def _first_identity_difference(expected, found, path="identity"):
    if isinstance(expected, dict) and isinstance(found, dict):
        for key, expected_value in expected.items():
            found_value = found.get(key, _MISSING)
            difference = _first_identity_difference(expected_value, found_value, f"{path}.{key}")
            if difference is not None:
                return difference
        for key, found_value in found.items():
            if key not in expected:
                return f"{path}.{key}", _MISSING, found_value
        return None
    if expected != found:
        return path, expected, found
    return None


def _identity_value(value):
    return "<missing>" if value is _MISSING else repr(value)


def validate_required_capabilities(manifest):
    if "required_capabilities" not in manifest:
        raise ValueError("nucleus competition manifest lacks required_capabilities")
    capabilities = manifest["required_capabilities"]
    if not isinstance(capabilities, list) or any(
        not isinstance(value, str) for value in capabilities
    ):
        raise ValueError("nucleus competition required_capabilities must be a list of strings")
    unsupported = sorted(set(capabilities).difference(IMPLEMENTED_CAPABILITIES))
    if unsupported:
        raise ValueError(
            f"nucleus competition manifest requires unsupported capabilities: {unsupported}"
        )
    return capabilities


def validate_publication_identity(manifest):
    """Enforce only the identity and retirement claims declared by this artifact."""

    identity = manifest.get("identity")
    difference = _first_identity_difference(identity_declaration(), identity)
    if difference is not None:
        field, expected, found = difference
        raise ValueError(
            f"unsupported nucleus competition identity field {field}: "
            f"expected {_identity_value(expected)}, found {_identity_value(found)}"
        )
    mint = identity["mint"]
    nucleus_to_emitted = {}
    emitted_to_nucleus = {}

    def record(nucleus_value, emitted_value, context):
        nucleus_id = _decimal_id(str(nucleus_value), f"{context} nucleus id")
        emitted_id = _decimal_id(emitted_value, f"{context} emitted id")
        expected = minted_nucleus_id(nucleus_id, mint)
        if emitted_id != expected:
            raise ValueError(
                f"{context} emitted id {emitted_id} does not match declared nucleus mint {expected}"
            )
        if nucleus_to_emitted.setdefault(nucleus_id, emitted_id) != emitted_id:
            raise ValueError("one nucleus maps to multiple emitted ids")
        if emitted_to_nucleus.setdefault(emitted_id, nucleus_id) != nucleus_id:
            raise ValueError("one emitted id maps to multiple nuclei")

    qualified = manifest.get("qualified_segment_labels")
    if not isinstance(qualified, dict):
        raise ValueError("competition manifest lacks qualified_segment_labels")
    for segment_text, raw_owners in qualified.items():
        _decimal_id(str(segment_text), "qualified source id")
        if not isinstance(raw_owners, dict):
            raise ValueError("qualified segment labels must map nuclei to emitted ids")
        for nucleus_text, emitted_text in raw_owners.items():
            record(nucleus_text, emitted_text, "qualified segment")

    repairs = manifest.get("repairs")
    if not isinstance(repairs, list):
        raise ValueError("competition manifest repairs must be a list")
    for repair in repairs:
        parent_id = _decimal_id(repair["parent_id"], "repair parent id")
        mappings = repair.get("territories")
        if not isinstance(mappings, list) or not mappings:
            raise ValueError("competition repair lacks explicit territory emission mappings")
        for mapping in mappings:
            emitted_id = _decimal_id(mapping["emitted_id"], "repair emitted id")
            if emitted_id == parent_id:
                raise ValueError(
                    "retired repair parents cannot be emitted by adjudicated territory"
                )
            record(mapping["anchor_id"], mapping["emitted_id"], "repair territory")

    expected_ledger = build_publication_ledger(repairs, qualified)
    if manifest.get("ledger") != expected_ledger:
        raise ValueError("nucleus competition publication ledger is incomplete or inconsistent")
    return identity


def local_manifest_path(value):
    text = str(value)
    if text.startswith("file://"):
        from urllib.parse import unquote
        from urllib.request import url2pathname

        text = url2pathname(unquote(text[len("file://") :]))
    if "://" in text:
        raise ValueError("NUC_COMPETITION_MANIFEST currently requires a shared local path")
    text = text.split("::", 1)[0]
    return Path(text)


def intersect_box(first, second):
    start = [max(int(first[i]), int(second[i])) for i in range(3)]
    stop = [min(int(first[i + 3]), int(second[i + 3])) for i in range(3)]
    if any(stop[i] <= start[i] for i in range(3)):
        return None
    return start + stop


def _sha256_file(path, block_size=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            block = handle.read(block_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _watershed_manifest_path(global_params):
    explicit = global_params.get("WS_MANIFEST")
    if explicit:
        path = local_manifest_path(explicit)
        if not path.is_file():
            raise FileNotFoundError(f"watershed manifest does not exist: {path}")
        return path
    watershed = local_manifest_path(global_params["WS_PATH"])
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


def load_validated_manifest(manifest_path, global_params):
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("manifest_type") != "abiss_nucleus_competition":
        raise ValueError(f"unexpected nucleus competition manifest: {manifest_path}")
    schema_version = str(manifest.get("schema_version"))
    if schema_version != PUBLICATION_SCHEMA_VERSION:
        if schema_version == "2.0":
            raise ValueError(
                "nucleus competition manifest schema '2.0' is a withdrawn pre-release "
                "contract that inverted the parent-id rule and is not accepted by the "
                "production overlay"
            )
        if schema_version == "1.2":
            command = (
                "python dev/zebrafinch/migrate_nucleus_competition.py --manifest "
                f"{shlex.quote(str(manifest_path.resolve()))}"
            )
            raise ValueError(
                f"nucleus competition manifest schema {schema_version!r} is not accepted by the "
                "production overlay; migrate the completed publication without recomputation by "
                f"running exactly: {command}"
            )
        raise ValueError(
            f"nucleus competition manifest schema {schema_version!r} is not accepted by the "
            "production overlay and has no lossless production migration; use the read-only "
            "comparison harness"
        )
    validate_required_capabilities(manifest)
    completion = manifest.get("completion", {})
    plan_digest = manifest.get("plan_digest")
    if completion.get("state") != "complete" or completion.get("plan_digest") != plan_digest:
        raise ValueError("nucleus competition manifest lacks a valid completion marker")
    configured_digest = global_params.get("NUC_COMPETITION_PLAN_DIGEST")
    if configured_digest is not None and str(configured_digest) != str(plan_digest):
        raise ValueError("nucleus competition manifest does not match the expected plan_digest")
    plan_file = manifest.get("plan_file")
    if not plan_file:
        raise ValueError("nucleus competition manifest lacks plan_file")
    plan_path = (manifest_path.parent / plan_file).resolve()
    if not plan_path.is_file() or _sha256_file(plan_path) != plan_digest:
        raise ValueError("nucleus competition units.json does not match plan_digest")
    expected_ws = manifest.get("fingerprints", {}).get("watershed", {}).get("manifest_sha256")
    actual_ws = _sha256_file(_watershed_manifest_path(global_params))
    if not expected_ws or expected_ws != actual_ws:
        raise ValueError(
            "nucleus competition manifest was built from another watershed identity; "
            "schema-1.2 migration must select the authoritative manifest with "
            "--watershed-manifest"
        )
    validate_publication_identity(manifest)
    return manifest


def validate_emitted_mapping(identity, parent_id, territory_ids, mappings):
    domain = {int(value) for value in territory_ids}
    mapped = {
        _decimal_id(item["internal_territory_id"], "internal territory id") for item in mappings
    }
    if mapped != domain or len(mapped) != len(mappings):
        raise ValueError("emitted-id mapping domain must equal the exact territory-id set")
    parent = int(parent_id)
    emitted = []
    anchors = []
    for item in mappings:
        anchor_id = _decimal_id(item["anchor_id"], "territory anchor id")
        emitted_id = _decimal_id(item["emitted_id"], "territory emitted id")
        expected = minted_nucleus_id(anchor_id, identity["mint"])
        if emitted_id != expected:
            raise ValueError("territory emitted id does not match the declared nucleus mint")
        if identity["parent_disposition"] == "retired" and emitted_id == parent:
            raise ValueError("retired repair parents cannot be emitted by adjudicated territory")
        anchors.append(anchor_id)
        emitted.append(emitted_id)
    if len(set(anchors)) != len(anchors) or len(set(emitted)) != len(emitted):
        raise ValueError("territory anchors and nucleus-scoped emitted ids must be unique per unit")


def protected_nucleus_owners(manifest):
    values = manifest.get("protected_nucleus_owners")
    if values is None:
        values = {
            int(owner)
            for repair in manifest.get("repairs", [])
            for owner in repair.get("marker_nucleus_ids", {}).values()
        }
    owners = {int(value) for value in values}
    if any(owner <= 0 for owner in owners):
        raise ValueError("protected nucleus ids must be positive")
    return owners


def filter_sparse_nucleus_ownership(labels, nucleus_labels, manifest, block_voxels=8_000_000):
    """Keep only watershed/nucleus pairs qualified by global mask support."""

    protected = protected_nucleus_owners(manifest)
    if not protected:
        print("nucleus competition: removed 0 sub-share sparse ownership voxels")
        return 0
    qualified = manifest.get("qualified_segment_owners")
    if qualified is None:
        raise ValueError("competition manifest lacks qualified_segment_owners")
    allowed_by_nucleus = {}
    for segment_text, owners in qualified.items():
        segment_id = int(segment_text)
        if segment_id < 0:
            raise ValueError("qualified watershed ids must be nonnegative")
        for owner in owners:
            nucleus_id = int(owner)
            if nucleus_id <= 0:
                raise ValueError("qualified nucleus ids must be positive")
            allowed_by_nucleus.setdefault(nucleus_id, []).append(segment_id)
    allowed_by_nucleus = {
        nucleus_id: np.asarray(sorted(set(segment_ids)), dtype=labels.dtype)
        for nucleus_id, segment_ids in allowed_by_nucleus.items()
    }

    flat_labels = labels.reshape(-1)
    flat_nuclei = nucleus_labels.reshape(-1)
    removed = 0
    for start in range(0, flat_nuclei.size, int(block_voxels)):
        stop = min(start + int(block_voxels), flat_nuclei.size)
        nucleus_block = flat_nuclei[start:stop]
        tagged = np.flatnonzero(nucleus_block)
        if tagged.size == 0:
            continue
        owners = nucleus_block[tagged]
        segment_block = flat_labels[start:stop]
        for raw_owner in np.unique(owners).tolist():
            owner = int(raw_owner)
            if owner not in protected:
                continue
            owner_positions = tagged[owners == raw_owner]
            allowed = allowed_by_nucleus.get(owner)
            if allowed is None:
                keep = np.zeros(owner_positions.shape, dtype=bool)
            else:
                keep = np.isin(segment_block[owner_positions], allowed)
            rejected = owner_positions[~keep]
            nucleus_block[rejected] = 0
            removed += int(rejected.size)
    print(f"nucleus competition: removed {removed} sub-share sparse ownership voxels")
    return removed


def canonicalize_qualified_segments(labels, manifest, block_voxels=8_000_000):
    """Give every globally single-owner watershed object a stable owner-specific id."""

    qualified = manifest.get("qualified_segment_labels")
    if qualified is None:
        raise ValueError("competition manifest lacks qualified_segment_labels")
    remap = {}
    for segment_text, owner_labels in qualified.items():
        if len(owner_labels) != 1:
            continue
        segment_id = int(segment_text)
        new_label = int(next(iter(owner_labels.values())))
        if segment_id < 0 or new_label < 0 or new_label > np.iinfo(labels.dtype).max:
            raise ValueError("qualified segment remap is outside the segmentation dtype")
        remap[segment_id] = new_label
    if not remap:
        print("nucleus competition: canonicalized 0 single-owner segmentation voxels")
        return 0

    source_ids = np.asarray(sorted(remap), dtype=labels.dtype)
    target_ids = np.asarray([remap[int(source)] for source in source_ids], dtype=labels.dtype)
    flat = labels.reshape(-1)
    changed = 0
    for start in range(0, flat.size, int(block_voxels)):
        stop = min(start + int(block_voxels), flat.size)
        block = flat[start:stop]
        positions = np.searchsorted(source_ids, block)
        in_range = positions < source_ids.size
        matched = np.zeros(block.shape, dtype=bool)
        matched[in_range] = source_ids[positions[in_range]] == block[in_range]
        if np.any(matched):
            block[matched] = target_ids[positions[matched]]
            changed += int(matched.sum())
    print(f"nucleus competition: canonicalized {changed} single-owner segmentation voxels")
    return changed


def apply_nucleus_competition_state(
    seg_cutout,
    nucleus_cutout,
    chunk_start_xyz,
    global_params,
):
    """Overlay flooded labels and their owning nucleus on one XYZC cutout.

    The ownership overlay is essential: an ABISS atomic chunk may intersect a
    competitive territory without containing its sparse nucleus core.  Without
    a dense owner tag that piece has ``NUC_STATE_NONE`` and can merge back into
    the competing soma before the hierarchy ever sees the core tag.
    """

    value = global_params.get("NUC_COMPETITION_MANIFEST")
    if not value:
        return seg_cutout, nucleus_cutout
    manifest_path = local_manifest_path(value)
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"nucleus competition manifest is missing: {manifest_path}; "
            "run the competitive_nucleus_growth stage before agglomeration"
        )
    manifest = load_validated_manifest(manifest_path, global_params)
    if seg_cutout.ndim != 4 or seg_cutout.shape[3] != 1:
        raise ValueError(f"watershed cutout must be single-channel XYZC, got {seg_cutout.shape}")
    if nucleus_cutout is not None:
        if nucleus_cutout.shape != seg_cutout.shape:
            raise ValueError(
                "nucleus ownership cutout must match watershed cutout, got "
                f"{nucleus_cutout.shape} versus {seg_cutout.shape}"
            )
        if not np.issubdtype(nucleus_cutout.dtype, np.integer):
            raise TypeError(f"nucleus ownership cutout must be integer, got {nucleus_cutout.dtype}")
    repairs = manifest.get("repairs", [])

    chunk_start = [int(v) for v in chunk_start_xyz]
    chunk_stop = [chunk_start[i] + int(seg_cutout.shape[i]) for i in range(3)]
    chunk_box = chunk_start + chunk_stop
    # CloudVolume/ABISS cutouts are commonly Fortran ordered. Flattening a
    # non-C-contiguous channel view with the default reshape order can create a
    # silent copy, so mutate explicit working arrays and always write them back.
    labels = np.array(seg_cutout[..., 0], copy=True, order="C")
    nucleus_labels = (
        np.array(nucleus_cutout[..., 0], copy=True, order="C")
        if nucleus_cutout is not None
        else None
    )
    if nucleus_labels is not None:
        filter_sparse_nucleus_ownership(labels, nucleus_labels, manifest)
    canonicalize_qualified_segments(labels, manifest)
    if not repairs:
        print("nucleus competition: no repair territories; canonicalization complete")
    changed = 0
    ownership_tagged = 0
    for repair in repairs:
        repair_box = [int(v) for v in repair["bbox_xyz"]]
        overlap = intersect_box(chunk_box, repair_box)
        if overlap is None:
            continue
        factor = int(repair["factor"])
        if factor < 1:
            raise ValueError("nucleus competition factor must be positive")
        territory_path = (manifest_path.parent / repair["territory_file"]).resolve()
        if manifest_path.parent.resolve() not in territory_path.parents:
            raise ValueError("competition territory escapes the manifest directory")
        with np.load(territory_path, allow_pickle=False) as archive:
            territory = np.asarray(archive["territory"])
            stored_box = [int(v) for v in archive["bbox_xyz"]]
            stored_factor = int(archive["factor"])
        if stored_box != repair_box or stored_factor != factor:
            raise ValueError(f"territory metadata does not match manifest: {territory_path}")
        expected_sha = repair.get("territory_sha256")
        if not expected_sha or _sha256_file(territory_path) != expected_sha:
            raise ValueError(f"territory fingerprint does not match manifest: {territory_path}")
        if not np.issubdtype(territory.dtype, np.integer):
            raise TypeError(f"territory ids must be integer-valued: {territory_path}")
        mappings = repair.get("territories")
        if not mappings:
            raise ValueError("competition repair lacks explicit territory emission mappings")
        territory_ids = {int(value) for value in np.unique(territory) if int(value) != 0}
        parent_id = int(repair["parent_id"])
        validate_emitted_mapping(manifest["identity"], parent_id, territory_ids, mappings)

        local = tuple(
            slice(overlap[i] - chunk_start[i], overlap[i + 3] - chunk_start[i]) for i in range(3)
        )
        pooled_indices = [
            (np.arange(overlap[i], overlap[i + 3], dtype=np.int64) - repair_box[i]) // factor
            for i in range(3)
        ]
        pooled = territory[np.ix_(pooled_indices[0], pooled_indices[1], pooled_indices[2])]
        block = labels[local]
        parent = block == parent_id
        nucleus_block = nucleus_labels[local] if nucleus_labels is not None else None
        for mapping in mappings:
            internal_id = int(mapping["internal_territory_id"])
            new_label = int(mapping["emitted_id"])
            select = parent & (pooled == internal_id)
            if new_label != parent_id:
                changed += int(select.sum())
                block[select] = new_label
            if nucleus_block is not None:
                nucleus_id = int(mapping["anchor_id"])
                if nucleus_id < 0 or nucleus_id > np.iinfo(nucleus_block.dtype).max:
                    raise ValueError(
                        f"competition nucleus id {nucleus_id} is outside {nucleus_block.dtype}"
                    )
                ownership_tagged += int(select.sum())
                nucleus_block[select] = nucleus_id
    print(
        f"nucleus competition: overlaid {changed} segmentation voxels and "
        f"{ownership_tagged} ownership voxels in this RAG chunk"
    )
    seg_cutout[..., 0] = labels
    if nucleus_labels is not None:
        nucleus_cutout[..., 0] = nucleus_labels
    return seg_cutout, nucleus_cutout


def apply_nucleus_competition(seg_cutout, chunk_start_xyz, global_params):
    """Compatibility wrapper for callers that need only the label overlay."""

    result, _ = apply_nucleus_competition_state(
        seg_cutout,
        None,
        chunk_start_xyz,
        global_params,
    )
    return result


__all__ = [
    "PUBLICATION_SCHEMA_VERSION",
    "apply_nucleus_competition",
    "apply_nucleus_competition_state",
    "build_publication_ledger",
    "canonicalize_qualified_segments",
    "filter_sparse_nucleus_ownership",
    "identity_declaration",
    "intersect_box",
    "load_validated_manifest",
    "local_manifest_path",
    "minted_nucleus_id",
    "protected_nucleus_owners",
    "validate_emitted_mapping",
    "validate_publication_identity",
    "validate_required_capabilities",
]
