"""Tests for scripts/volume_backends.py (run: pytest scripts/test_volume_backends.py).

These exercise the ABISS-side adapters directly. A previous version of this work cited
tests that lived in pytorch_connectomics and only covered an equivalent transform
there -- they never imported this module, so none of the behaviour below was actually
guarded.
"""
import json
import os
import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cut_chunk_common import convert_and_scale_integer_data  # noqa: E402
from volume_backends import (  # noqa: E402
    is_h5_chunkstore_path,
    is_h5_path,
    is_zarr_path,
    open_volume,
)

# NOT importorskip: h5py and zarr are hard requirements of these backends (they are
# installed in docker/Dockerfile). Skipping on ImportError would turn "the ABISS image
# forgot a dependency" -- exactly the review_v1 finding -- into a green test run.
import h5py  # noqa: E402
import zarr  # noqa: E402


def _write_h5(path, arr, dataset="main"):
    with h5py.File(path, "w") as f:
        f.create_dataset(dataset, data=arr)
    return str(path)


def _banis_reference(a):
    """Port of dev/zebrafinch/upload_affinity_full_masked.py (edge shift + channel flip)."""
    out = np.empty_like(a)
    for c in range(3):
        s = a[c]
        dst = np.empty_like(s)
        hi = [slice(None)] * 3
        lo = [slice(None)] * 3
        hi[c] = slice(1, None)
        lo[c] = slice(0, -1)
        dst[tuple(hi)] = s[tuple(lo)]
        face = [slice(None)] * 3
        face[c] = 0
        dst[tuple(face)] = 0
        out[c] = dst
    return np.transpose(np.clip(out[::-1], 0, 1), (3, 2, 1, 0))


def _param(tmp_path, **kw):
    p = tmp_path / "param"
    p.write_text(json.dumps(kw))
    os.environ["PARAM_JSON"] = str(p)
    return p


# --- dtype / C++ ABI -------------------------------------------------------------

def test_float16_affinity_is_materialized_as_float32():
    """aff.raw is mmapped as aff_t (float32); a float16 passthrough halves the file."""
    a16 = np.random.default_rng(0).random((4, 4, 4, 3)).astype("float16")
    out = convert_and_scale_integer_data(a16, "float32")
    assert out.dtype == np.float32
    assert out.nbytes == a16.size * 4


def test_float32_affinity_is_not_copied_or_rescaled():
    a32 = np.random.default_rng(1).random((4, 4, 4, 3)).astype("float32")
    out = convert_and_scale_integer_data(a32, "float32")
    assert out.dtype == np.float32
    assert np.array_equal(out, a32)


def test_h5_float16_read_then_convert_matches_source(tmp_path):
    """End of the float16 chain: HDF5 -> backend -> cut_data's conversion -> aff_t."""
    src = np.random.default_rng(2).random((3, 5, 6, 7)).astype("float16")
    v = open_volume(_write_h5(tmp_path / "aff.h5", src))
    block = np.asarray(v[0:7, 0:6, 0:5])
    assert block.dtype == np.float16                      # adapter preserves storage dtype
    conv = convert_and_scale_integer_data(block, "float32")
    assert conv.dtype == np.float32
    assert np.allclose(conv, np.transpose(src, (3, 2, 1, 0)).astype("float32"))


# --- read correctness ------------------------------------------------------------

def test_h5_read_matches_czyx_to_xyzc(tmp_path):
    src = np.random.default_rng(3).random((3, 5, 6, 7)).astype("float32")
    v = open_volume(_write_h5(tmp_path / "a.h5", src))
    assert v.shape == (7, 6, 5, 3)
    assert np.array_equal(np.asarray(v[0:7, 0:6, 0:5]), np.transpose(src, (3, 2, 1, 0)))
    assert np.array_equal(np.asarray(v[1:4, 2:5, 1:3]),
                          np.transpose(src, (3, 2, 1, 0))[1:4, 2:5, 1:3])


def test_h5_dataset_selection(tmp_path):
    p = tmp_path / "two.h5"
    with h5py.File(p, "w") as f:
        f.create_dataset("a", data=np.zeros((3, 2, 2, 2), "float32"))
        f.create_dataset("b", data=np.ones((3, 2, 2, 2), "float32"))
    with pytest.raises(ValueError, match="specify one"):
        open_volume(str(p))
    assert np.asarray(open_volume(f"{p}::b")[0:2, 0:2, 0:2]).max() == 1.0


def test_h5_is_read_only(tmp_path):
    src = np.zeros((3, 2, 2, 2), "float32")
    v = open_volume(_write_h5(tmp_path / "ro.h5", src))
    with pytest.raises(NotImplementedError, match="read-only"):
        v[0:1, 0:1, 0:1] = np.ones((1, 1, 1, 3), "float32")


# --- BANIS conversion ------------------------------------------------------------

def test_banis_full_and_chunked_reads_match_reference(tmp_path):
    src = np.random.default_rng(4).random((3, 9, 8, 7)).astype("float32")
    path = _write_h5(tmp_path / "aff.h5", src)
    _param(tmp_path, AFF_PATH=path, AFF_CONVENTION="banis")
    v = open_volume(path)
    ref = _banis_reference(src)
    assert np.allclose(np.asarray(v[0:7, 0:8, 0:9]), ref)
    # Sub-reads must equal the same whole-volume answer: the adapter reads one extra
    # voxel on each low face so a chunk seam pulls the true neighbour, not a zero.
    for x0 in (0, 3, 5):
        for y0 in (0, 4):
            for z0 in (0, 5):
                got = np.asarray(v[x0:x0 + 2, y0:y0 + 3, z0:z0 + 4])
                assert np.allclose(got, ref[x0:x0 + 2, y0:y0 + 3, z0:z0 + 4]), (x0, y0, z0)


def test_banis_only_applies_to_aff_path(tmp_path):
    src = np.random.default_rng(5).random((3, 4, 4, 4)).astype("float32")
    aff = _write_h5(tmp_path / "aff.h5", src)
    other = _write_h5(tmp_path / "seg.h5", src)
    _param(tmp_path, AFF_PATH=aff, AFF_CONVENTION="banis")
    assert np.array_equal(np.asarray(open_volume(other)[0:4, 0:4, 0:4]),
                          np.transpose(src, (3, 2, 1, 0)))


def test_banis_rejects_non_three_channel(tmp_path):
    """4-channel affinity+myelin would be reordered into nonsense; refuse instead."""
    src = np.random.default_rng(6).random((4, 4, 4, 4)).astype("float32")
    path = _write_h5(tmp_path / "aff4.h5", src)
    _param(tmp_path, AFF_PATH=path, AFF_CONVENTION="banis")
    with pytest.raises(ValueError, match="exactly 3 channels"):
        open_volume(path)


def test_restore_sigmoid(tmp_path):
    src = np.random.default_rng(7).random((3, 4, 4, 4)).astype("float32")
    path = _write_h5(tmp_path / "aff.h5", src)
    _param(tmp_path, AFF_PATH=path, AFF_CONVENTION="banis", AFF_RESTORE_SIGMOID=0.2)
    p = np.clip(src, 1e-6, 1 - 1e-6)
    logit = np.log(p) - np.log1p(-p)
    ref = _banis_reference((1.0 / (1.0 + np.exp(-logit / 0.2))).astype("float32"))
    assert np.allclose(np.asarray(open_volume(path)[0:4, 0:4, 0:4]), ref, atol=1e-6)


# --- dispatch / guards -----------------------------------------------------------

def test_nonzero_mip_is_rejected(tmp_path):
    """Single-scale backends must not silently read scale 0 for a mip>0 request."""
    path = _write_h5(tmp_path / "a.h5", np.zeros((3, 2, 2, 2), "float32"))
    os.environ.pop("PARAM_JSON", None)
    with pytest.raises(ValueError, match="single-scale"):
        open_volume(path, mip=2)
    open_volume(path, mip=0)          # mip 0 is fine
    open_volume(path, mip=[9, 9, 20])  # a resolution triple means scale 0 here


def test_path_detection_is_suffix_based(tmp_path):
    assert is_h5_path("/x/a.h5") and is_h5_path("/x/a.h5::main") and is_h5_path("/x/a.hdf5")
    assert is_zarr_path("/x/a.zarr") and is_zarr_path("/x/a.zarr/")
    # a precomputed layer that merely lives under a *.zarr directory is NOT zarr
    assert not is_zarr_path("/x/a.zarr/inner/precomputed_layer")
    assert not is_zarr_path("/x/plain") and not is_h5_path("/x/plain")


def test_zarr_roundtrip_and_default_read_only(tmp_path):
    src = np.random.default_rng(8).random((3, 4, 5, 6)).astype("float32")
    p = tmp_path / "a.zarr"
    z = zarr.open(str(p), mode="w", shape=src.shape, chunks=(3, 2, 2, 2), dtype="float32")
    z[:] = src
    os.environ.pop("PARAM_JSON", None)
    v = open_volume(str(p))
    assert np.array_equal(np.asarray(v[0:6, 0:5, 0:4]), np.transpose(src, (3, 2, 1, 0)))
    # default is read-only, so a typo cannot silently create a store
    missing = tmp_path / "nope.zarr"
    with pytest.raises(Exception):
        open_volume(str(missing))
    assert not missing.exists()
    w = open_volume(str(p), writable=True)
    w[0:2, 0:2, 0:2] = np.full((2, 2, 2, 3), 0.25, "float32")
    assert np.allclose(np.asarray(w[0:2, 0:2, 0:2]), 0.25)


# --------------------------------------------------------------------------------
# Production write paths.
#
# review_v2 [P1]: making zarr read-only by default broke every real writer, and the
# adapter-level test missed it because it passed writable=True itself. These assert
# on the actual writer modules.
# --------------------------------------------------------------------------------

WRITER_CALLS = [
    ("cut_chunk_ws.py", "ADJUSTED_AFF_PATH"),
    ("upload_chunk.py", "sys.argv[2]"),
    ("upload_size.py", "sys.argv[2]"),
]


@pytest.mark.parametrize("module_name,target", WRITER_CALLS)
def test_production_writers_open_volumes_writable(module_name, target):
    """Each writer must opt in to `writable=True` or its first assignment raises."""
    import re

    src = (pathlib.Path(__file__).parent / module_name).read_text()
    calls = [m for m in re.findall(r"open_volume\((?:[^()]|\([^()]*\))*\)", src)]
    assert calls, f"{module_name} no longer calls open_volume"
    for call in calls:
        assert "writable=True" in call, (
            f"{module_name}: {call} opens a WRITE target read-only; the following "
            f"assignment raises NotImplementedError (target {target})."
        )


def test_writable_zarr_accepts_assignment(tmp_path):
    """End-to-end of the writer contract: open writable, assign, read back."""
    path = tmp_path / "out.zarr"
    zarr.open(str(path), mode="w", shape=(4, 4, 4), chunks=(2, 2, 2), dtype="uint32")

    vol = open_volume(str(path), writable=True)
    vol[0:2, 0:2, 0:2] = np.full((2, 2, 2), 7, dtype=np.uint32)

    assert np.all(open_volume(str(path))[0:2, 0:2, 0:2] == 7)


def test_writable_is_not_forwarded_to_cloudvolume():
    """`writable` is ours; reaching the CloudVolume constructor is a TypeError."""
    import inspect

    import volume_backends

    src = inspect.getsource(volume_backends.open_volume)
    pop_at = src.index('kwargs.pop("writable"')
    # It must be consumed before either dispatch branch, not inside the zarr branch.
    assert pop_at < src.index("ZarrVolume(")
    assert pop_at < src.index("CloudVolume(")


# --------------------------------------------------------------------------------
# aff_t ABI (review_v2 [P2]).
# --------------------------------------------------------------------------------


def test_affinity_dtype_follows_compiled_aff_t(monkeypatch):
    from cut_chunk_common import affinity_dtype

    monkeypatch.delenv("ABISS_AFF_DTYPE", raising=False)
    assert affinity_dtype() == "float32"

    monkeypatch.setenv("ABISS_AFF_DTYPE", "float64")  # -DDOUBLE build
    assert affinity_dtype() == "float64"

    monkeypatch.setenv("ABISS_AFF_DTYPE", "float16")
    with pytest.raises(ValueError, match="float32 or float64"):
        affinity_dtype()


def test_save_raw_data_writes_the_full_element_width(tmp_path, monkeypatch):
    """A float16 array must never reach aff.raw at half the expected byte length."""
    from cut_chunk_common import affinity_dtype, save_raw_data

    monkeypatch.delenv("ABISS_AFF_DTYPE", raising=False)
    data = np.arange(3 * 4 * 5, dtype="float16").reshape(3, 4, 5)
    converted = convert_and_scale_integer_data(data, affinity_dtype())

    fn = tmp_path / "aff.raw"
    save_raw_data(str(fn), converted)
    assert fn.stat().st_size == data.size * 4
    assert np.allclose(
        np.fromfile(fn, dtype="float32").reshape(data.shape, order="F"),
        data.astype("float32"),
    )


def test_misaligned_writable_zarr_write_is_rejected(tmp_path):
    """The concurrency precondition is enforced, not just documented."""
    p = tmp_path / "aligned.zarr"
    zarr.open(str(p), mode="w", shape=(8, 8, 8), chunks=(4, 4, 4), dtype="uint32")
    v = open_volume(str(p), writable=True)
    block = np.ones((4, 4, 4), dtype="uint32")

    v[0:4, 0:4, 0:4] = block  # aligned
    v[4:8, 4:8, 4:8] = block  # aligned, last block

    with pytest.raises(ValueError, match="not aligned to the storage chunk"):
        v[1:5, 0:4, 0:4] = block


def test_ragged_final_block_is_allowed(tmp_path):
    """A volume whose extent is not a chunk multiple must still be writable."""
    p = tmp_path / "ragged.zarr"
    zarr.open(str(p), mode="w", shape=(6, 6, 6), chunks=(4, 4, 4), dtype="uint32")
    v = open_volume(str(p), writable=True)

    v[4:6, 4:6, 4:6] = np.ones((2, 2, 2), dtype="uint32")  # stops at the volume edge
    assert np.all(np.asarray(open_volume(str(p))[4:6, 4:6, 4:6]) == 1)


# --------------------------------------------------------------------------------
# Multi-file chunk store: the 726-file grid affinity presented as one volume.
# --------------------------------------------------------------------------------


def _make_chunkstore(tmp_path, grid=(2, 2, 2), cshape=(6, 6, 6), seed=3):
    """Write a grid of chunk_z*_y*_x*.h5 and return (dir, monolithic array)."""
    rng = np.random.default_rng(seed)
    full = rng.random((3,) + tuple(g * c for g, c in zip(grid, cshape))).astype("float32")
    d = tmp_path / "store.h5.chunks"
    d.mkdir()
    for gz in range(grid[0]):
        for gy in range(grid[1]):
            for gx in range(grid[2]):
                sl = (
                    slice(None),
                    slice(gz * cshape[0], (gz + 1) * cshape[0]),
                    slice(gy * cshape[1], (gy + 1) * cshape[1]),
                    slice(gx * cshape[2], (gx + 1) * cshape[2]),
                )
                with h5py.File(d / f"chunk_z{gz}_y{gy}_x{gx}.h5", "w") as f:
                    ds = f.create_dataset("main", data=full[sl])
                    ds.attrs["chunk_start_zyx"] = str(
                        [gz * cshape[0], gy * cshape[1], gx * cshape[2]]
                    )
    return d, full


def test_chunkstore_is_detected_and_shaped_like_the_whole_grid(tmp_path):
    d, full = _make_chunkstore(tmp_path)
    assert is_h5_chunkstore_path(str(d))
    assert not is_h5_path(str(d))  # a directory is not a single-file h5

    vol = open_volume(str(d))
    zdim, ydim, xdim = full.shape[1:]
    assert vol.shape == (xdim, ydim, zdim, 3)


def test_chunkstore_read_matches_the_monolithic_array(tmp_path):
    """Assembly must be transparent, including boxes spanning several files."""
    d, full = _make_chunkstore(tmp_path)
    vol = open_volume(str(d))
    reference = np.transpose(full, (3, 2, 1, 0))  # (C,Z,Y,X) -> (X,Y,Z,C)

    for box in [
        (slice(0, 6), slice(0, 6), slice(0, 6)),      # exactly one chunk
        (slice(3, 9), slice(3, 9), slice(3, 9)),      # straddles all three seams
        (slice(0, 12), slice(0, 12), slice(0, 12)),   # whole grid
        (slice(5, 7), slice(5, 7), slice(5, 7)),      # 1 voxel either side of a seam
    ]:
        got = np.asarray(vol[box])
        assert np.allclose(got, reference[box]), f"mismatch on {box}"


def test_chunkstore_banis_conversion_crosses_file_seams(tmp_path):
    """The BANIS 1-voxel low margin may live in the NEIGHBOURING file."""
    d, full = _make_chunkstore(tmp_path)
    store = open_volume(str(d), convention="banis")
    monolithic = open_volume_array_for_test(full)

    box = (slice(6, 10), slice(6, 10), slice(6, 10))  # starts exactly on a seam
    assert np.allclose(np.asarray(store[box]), np.asarray(monolithic[box]))


def open_volume_array_for_test(array):
    import volume_backends

    return volume_backends._ArrayVolume(
        array, writable=False, label="monolithic", convention="banis"
    )


def test_chunkstore_rejects_a_grid_that_contradicts_the_attrs(tmp_path):
    """Filename grid index and chunk_start_zyx must agree, or placement is wrong."""
    d, _ = _make_chunkstore(tmp_path)
    with h5py.File(d / "chunk_z0_y0_x0.h5", "a") as f:
        f["main"].attrs["chunk_start_zyx"] = str([999, 0, 0])
    with pytest.raises(ValueError, match="does not match grid index"):
        open_volume(str(d))
