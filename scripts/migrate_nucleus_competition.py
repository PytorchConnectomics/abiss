#!/usr/bin/env python3
"""Upgrade a completed schema-1.2 nucleus publication without recomputing floods.

The legacy territory arrays are marker-index arrays and remain byte-for-byte
unchanged. The command writes a plan sidecar, preserves the original manifest,
and atomically replaces only ``manifest.json`` with a completed schema-3.0
publication whose identity and retirement claims are explicit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any
from urllib.parse import unquote
from urllib.request import url2pathname

import numpy as np

ABISS_SCRIPTS = Path(__file__).resolve().parent
if str(ABISS_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(ABISS_SCRIPTS))

# isort: off
from nucleus_overlay import (  # noqa: E402
    PUBLICATION_SCHEMA_VERSION,
    build_publication_ledger,
    identity_declaration,
    validate_publication_identity,
    validate_required_capabilities,
)

# isort: on


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def _json_bytes(payload: dict[str, Any]) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _atomic_write(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("xb") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _preserve_once(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError:
        if path.read_bytes() != value:
            raise FileExistsError(f"legacy backup exists with different content: {path}")


def _local_path(value: str) -> Path:
    text = str(value)
    if text.startswith("file://"):
        text = url2pathname(unquote(text[len("file://") :]))
    if "://" in text:
        raise ValueError(f"migration requires a local watershed publication, found {value!r}")
    return Path(text.split("::", 1)[0])


def _infer_watershed_manifest(legacy: dict[str, Any], explicit: Path | None) -> Path:
    if explicit is not None:
        candidate = explicit.resolve()
        if not candidate.is_file():
            raise FileNotFoundError(f"watershed manifest does not exist: {candidate}")
        return candidate
    watershed = _local_path(legacy["base_watershed"]).resolve()
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
        "cannot identify the watershed publication; pass --watershed-manifest explicitly"
    )


def _territory_mappings(
    manifest_path: Path, repair: dict[str, Any]
) -> tuple[list[dict[str, Any]], str]:
    territory_path = (manifest_path.parent / str(repair["territory_file"])).resolve()
    if manifest_path.parent.resolve() not in territory_path.parents:
        raise ValueError("legacy territory escapes the publication directory")
    with np.load(territory_path, allow_pickle=False) as archive:
        territory = np.asarray(archive["territory"])
        stored_box = [int(value) for value in archive["bbox_xyz"]]
        stored_factor = int(archive["factor"])
    if territory.dtype != np.int32:
        raise ValueError(f"legacy territory must retain int32 marker indices: {territory_path}")
    if stored_box != [int(value) for value in repair["bbox_xyz"]]:
        raise ValueError(f"legacy territory bbox differs from its manifest: {territory_path}")
    if stored_factor != int(repair["factor"]):
        raise ValueError(f"legacy territory factor differs from its manifest: {territory_path}")
    marker_ids = {int(value) for value in np.unique(territory) if int(value) != 0}
    marker_labels = repair.get("marker_labels")
    marker_nuclei = repair.get("marker_nucleus_ids")
    if not isinstance(marker_labels, dict) or not isinstance(marker_nuclei, dict):
        raise ValueError("schema-1.2 repair lacks marker_labels or marker_nucleus_ids")
    if {int(value) for value in marker_labels} != marker_ids:
        raise ValueError("legacy marker-label domain differs from the territory marker indices")
    if {int(value) for value in marker_nuclei} != marker_ids:
        raise ValueError("legacy marker-nucleus domain differs from the territory marker indices")
    pooled_voxels = repair.get("pooled_voxels", {})
    mappings = []
    for marker in sorted(marker_ids):
        nucleus_text = str(marker_nuclei[str(marker)])
        mappings.append(
            {
                "anchor_id": nucleus_text,
                "internal_territory_id": str(marker),
                "emitted_id": str(marker_labels[str(marker)]),
                "pooled_voxels": int(pooled_voxels[nucleus_text]),
            }
        )
    return mappings, _sha256_file(territory_path)


def migrate_manifest(
    manifest_path: Path | str, watershed_manifest: Path | str | None = None
) -> dict[str, Any]:
    """Atomically upgrade one schema-1.2 manifest and return the published payload."""

    path = Path(manifest_path).resolve()
    source_bytes = path.read_bytes()
    legacy = json.loads(source_bytes)
    if legacy.get("manifest_type") != "abiss_nucleus_competition":
        raise ValueError(f"unexpected nucleus competition manifest: {path}")
    schema = str(legacy.get("schema_version"))
    if schema != "1.2":
        raise ValueError(
            f"migration supports schema '1.2' only, found {schema!r}; schema 1.0 ids are opaque"
        )
    source_sha = _sha256_bytes(source_bytes)
    explicit_ws = Path(watershed_manifest) if watershed_manifest is not None else None
    ws_manifest = _infer_watershed_manifest(legacy, explicit_ws)

    repairs = []
    for repair in legacy.get("repairs", []):
        mappings, territory_sha = _territory_mappings(path, repair)
        migrated = {
            key: value
            for key, value in repair.items()
            if key not in {"anchor_labels", "marker_labels", "marker_nucleus_ids"}
        }
        migrated.update(
            {
                "parent_id": str(repair["parent_id"]),
                "anchor_ids": [str(value) for value in repair["anchor_ids"]],
                "territory_sha256": territory_sha,
                "territory_encoding": "marker_index",
                "territories": mappings,
            }
        )
        repairs.append(migrated)

    migration_dir = path.parent / ".nuccomp-migrations" / source_sha[:16]
    plan_path = migration_dir / "units.json"
    plan = {
        "schema_version": PUBLICATION_SCHEMA_VERSION,
        "required_capabilities": [],
        "plan_type": "abiss_nucleus_competition_migration",
        "source_schema_version": schema,
        "source_manifest_sha256": source_sha,
        "territory_recomputed": False,
        "territory_encoding": "marker_index",
        "repairs": repairs,
    }
    plan_bytes = _json_bytes(plan)
    plan_digest = _sha256_bytes(plan_bytes)
    _atomic_write(plan_path, plan_bytes)

    backup_path = path.with_name(f"manifest.schema-{schema}.{source_sha[:16]}.json")
    _preserve_once(backup_path, source_bytes)
    fingerprints = dict(legacy.get("fingerprints", {}))
    fingerprints.update(
        {
            "watershed": {
                "manifest_file": str(ws_manifest),
                "manifest_sha256": _sha256_file(ws_manifest),
            },
            "units": {"sha256": plan_digest},
            "legacy_manifest": {"schema_version": schema, "sha256": source_sha},
        }
    )
    manifest = dict(legacy)
    manifest.update(
        {
            "schema_version": PUBLICATION_SCHEMA_VERSION,
            "required_capabilities": [],
            "run_id": f"migration-{source_sha[:16]}",
            "plan_digest": plan_digest,
            "plan_file": os.path.relpath(plan_path, path.parent),
            "completion": {"state": "complete", "plan_digest": plan_digest},
            "fingerprints": fingerprints,
            "identity": identity_declaration(),
            "repairs": repairs,
            "zero_repairs": len(repairs) == 0,
            "reason": (
                "migrated_legacy_publication_without_recomputation"
                if repairs
                else "migrated_legacy_publication_with_no_repairs"
            ),
            "separation_claim": "local_only" if repairs else "none",
            "migration": {
                "source_schema_version": schema,
                "source_manifest_sha256": source_sha,
                "source_manifest_backup": backup_path.name,
                "territory_recomputed": False,
            },
        }
    )
    manifest["ledger"] = build_publication_ledger(
        manifest["repairs"], manifest.get("qualified_segment_labels", {})
    )
    validate_required_capabilities(manifest)
    validate_publication_identity(manifest)
    _atomic_write(path, _json_bytes(manifest))
    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--watershed-manifest", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = migrate_manifest(args.manifest, args.watershed_manifest)
    print(
        f"migrated {args.manifest.resolve()} to schema {manifest['schema_version']} "
        "without recomputing territory arrays"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
