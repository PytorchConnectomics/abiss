import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from volume_backends import _ChunkedH5Array, open_volume  # noqa: E402


def _write_chunk(path, value):
    with h5py.File(path, "w") as handle:
        handle.create_dataset(
            "main",
            data=np.full((3, 2, 2, 2), value, dtype=np.float16),
        )


def _production_param(tmp_path, chunk_store):
    path = tmp_path / "param.json"
    path.write_text(
        json.dumps(
            {
                "AFF_PATH": str(chunk_store),
                "BBOX": [0, 0, 0, 4, 2, 2],
            }
        )
    )
    return path


def test_production_grid_does_not_depend_on_directory_listing(tmp_path, monkeypatch):
    store = tmp_path / "affinity.h5.chunks"
    store.mkdir()
    _write_chunk(store / "chunk_z0_y0_x0.h5", 1)
    _write_chunk(store / "chunk_z0_y0_x1.h5", 2)
    monkeypatch.setenv("PARAM_JSON", str(_production_param(tmp_path, store)))

    def reject_readdir(path):
        if Path(path) == store:
            raise AssertionError("production grid must not use readdir")
        return []

    monkeypatch.setattr("volume_backends.os.listdir", reject_readdir)
    array = _ChunkedH5Array(str(store))
    volume = open_volume(str(store))

    assert array.shape == (3, 2, 2, 4)
    assert volume.shape == (4, 2, 2, 3)
    seam = array[:, :, :, 1:3]
    np.testing.assert_array_equal(seam[..., 0], 1)
    np.testing.assert_array_equal(seam[..., 1], 2)


def test_required_missing_chunk_fails_instead_of_zero_filling(tmp_path, monkeypatch):
    store = tmp_path / "affinity.h5.chunks"
    store.mkdir()
    _write_chunk(store / "chunk_z0_y0_x0.h5", 1)
    monkeypatch.setenv("PARAM_JSON", str(_production_param(tmp_path, store)))

    array = _ChunkedH5Array(str(store))
    with pytest.raises((FileNotFoundError, OSError)):
        array[:, :, :, 2:4]
