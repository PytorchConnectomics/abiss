"""Pluggable volume backends for ABISS: precomputed (CloudVolume), zarr, HDF5.

ABISS talks to volumes through a very small slice of the CloudVolume API:

    v.shape        # (X, Y, Z, C)
    v.dtype
    v[x0:x1, y0:y1, z0:z1]        # -> (X, Y, Z, C)
    v[x0:x1, y0:y1, z0:z1] = arr  # writes

so any store that can present that can be dropped in. `open_volume()` dispatches on
the path and returns either a real CloudVolume (unchanged default) or one of the
adapters below.

Why this exists: affinity produced by inference is naturally (C, Z, Y, X) HDF5 or
zarr. Without a backend it has to be converted into a precomputed layer first --
a full second copy of the volume (3.1 TB for zebrafinch) purely for format.

Storage convention for the adapters: arrays are (C, Z, Y, X) for multi-channel or
(Z, Y, X) for single-channel, i.e. what the rest of this codebase writes. The
adapter transposes to CloudVolume's (X, Y, Z, C) view on read and back on write.

Concurrency: ABISS runs many chunk workers at once.
  - zarr: safe to write concurrently as long as workers write DISJOINT,
    chunk-aligned regions, which is how ABISS writes (whole logical chunks).
  - HDF5: READ-ONLY here. HDF5 has no multi-process writer support (SWMR is
    single-writer), so concurrent chunk writes would silently corrupt the file.
    Attempting to write raises instead of quietly producing garbage.
"""
import os

import numpy as np

__all__ = ["open_volume", "is_zarr_path", "is_h5_path"]

_ZARR_SUFFIXES = (".zarr",)
_H5_SUFFIXES = (".h5", ".hdf5")


def _strip_scheme(path):
    for scheme in ("zarr://", "h5://", "hdf5://", "file://"):
        if path.startswith(scheme):
            return path[len(scheme) :]
    return path


def is_zarr_path(path):
    p = _strip_scheme(str(path))
    return str(path).startswith("zarr://") or any(s in p for s in _ZARR_SUFFIXES)


def is_h5_path(path):
    p = _strip_scheme(str(path))
    return str(path).startswith(("h5://", "hdf5://")) or any(
        p.split("::")[0].endswith(s) for s in _H5_SUFFIXES
    )


class _ArrayVolume(object):
    """Common (X, Y, Z, C) view over an array stored as (C, Z, Y, X) or (Z, Y, X)."""

    def __init__(self, array, writable=False, label=""):
        self._a = array
        self._writable = writable
        self._label = label
        if array.ndim == 4:
            self._channels = int(array.shape[0])
        elif array.ndim == 3:
            self._channels = 1
        else:
            raise ValueError(
                "%s: expected (C, Z, Y, X) or (Z, Y, X), got shape %r"
                % (label, tuple(array.shape))
            )

    @property
    def shape(self):
        z, y, x = self._a.shape[-3:]
        return (int(x), int(y), int(z), self._channels)

    @property
    def dtype(self):
        return self._a.dtype

    @property
    def num_channels(self):
        return self._channels

    @staticmethod
    def _xyz_slices(key):
        """CloudVolume-style [x, y, z] or [x, y, z, c] -> (xs, ys, zs, cs)."""
        if not isinstance(key, tuple):
            key = (key,)
        xs, ys, zs = (key + (slice(None),) * 3)[:3]
        cs = key[3] if len(key) > 3 else slice(None)
        return xs, ys, zs, cs

    def __getitem__(self, key):
        xs, ys, zs, cs = self._xyz_slices(key)
        if self._a.ndim == 4:
            block = self._a[cs, zs, ys, xs]  # (C, Z, Y, X)
            block = np.asarray(block)
            if block.ndim == 3:  # a single channel was selected by an int
                block = block[np.newaxis, ...]
            return np.transpose(block, (3, 2, 1, 0))  # -> (X, Y, Z, C)
        block = np.asarray(self._a[zs, ys, xs])  # (Z, Y, X)
        return np.transpose(block, (2, 1, 0))[..., np.newaxis]

    def __setitem__(self, key, value):
        if not self._writable:
            raise NotImplementedError(
                "%s is read-only. HDF5 has no multi-process writer support, and ABISS "
                "writes chunk outputs in parallel; use zarr or precomputed for outputs."
                % self._label
            )
        xs, ys, zs, cs = self._xyz_slices(key)
        value = np.asarray(value)
        if self._a.ndim == 4:
            if value.ndim == 3:
                value = value[..., np.newaxis]
            self._a[cs, zs, ys, xs] = np.transpose(value, (3, 2, 1, 0))
        else:
            if value.ndim == 4:
                value = value[..., 0]
            self._a[zs, ys, xs] = np.transpose(value, (2, 1, 0))


class H5Volume(_ArrayVolume):
    """Read-only HDF5 volume. Path may be ``file.h5`` or ``file.h5::dataset``."""

    def __init__(self, path):
        import h5py

        # HDF5 file locking breaks on many networked filesystems; readers only.
        os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
        raw = _strip_scheme(str(path))
        dataset = None
        if "::" in raw:
            raw, dataset = raw.split("::", 1)
        self._handle = h5py.File(raw, "r")
        if dataset is None:
            names = []
            self._handle.visititems(
                lambda name, obj: names.append(name)
                if isinstance(obj, h5py.Dataset)
                else None
            )
            if len(names) != 1:
                raise ValueError(
                    "%s contains %d datasets %r; specify one as path.h5::dataset"
                    % (raw, len(names), names)
                )
            dataset = names[0]
        super(H5Volume, self).__init__(
            self._handle[dataset], writable=False, label="HDF5 %s::%s" % (raw, dataset)
        )


class ZarrVolume(_ArrayVolume):
    """Zarr volume. Concurrent writes are safe only for disjoint chunk-aligned regions."""

    def __init__(self, path, writable=True):
        import zarr

        raw = _strip_scheme(str(path))
        self._array = zarr.open(raw, mode="a" if writable else "r")
        super(ZarrVolume, self).__init__(
            self._array, writable=writable, label="zarr %s" % raw
        )


def open_volume(path, **kwargs):
    """Open ``path`` with the backend its name implies.

    Unrecognised paths fall through to CloudVolume, so precomputed/gs:// behaviour is
    unchanged. CloudVolume-only kwargs (mip, fill_missing, bounded, ...) are accepted
    and ignored by the single-scale adapters.
    """
    text = str(path)
    if is_h5_path(text):
        return H5Volume(text)
    if is_zarr_path(text):
        return ZarrVolume(text, writable=kwargs.pop("writable", True))
    from cloudvolume import CloudVolume

    return CloudVolume(text, **kwargs)
