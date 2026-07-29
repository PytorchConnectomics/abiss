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
  - zarr: safe to write concurrently only for DISJOINT, storage-chunk-aligned
    regions. This is ENFORCED on write (see _require_storage_chunk_alignment), not
    merely assumed -- a misaligned write raises rather than racing under load.
  - HDF5: READ-ONLY here. HDF5 has no multi-process writer support (SWMR is
    single-writer), so concurrent chunk writes would silently corrupt the file.
    Attempting to write raises instead of quietly producing garbage.
"""
import collections
import json
import re
import os

# HDF5 reads this during library initialization, so it must be set before h5py (and
# therefore libhdf5) is imported anywhere -- setting it inside H5Volume.__init__ is
# already too late. File locking breaks on many networked filesystems; these adapters
# only ever read HDF5.
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

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
    p = _strip_scheme(str(path)).rstrip("/")
    return str(path).startswith("zarr://") or p.endswith(_ZARR_SUFFIXES)


def is_h5_path(path):
    p = _strip_scheme(str(path))
    return str(path).startswith(("h5://", "hdf5://")) or any(
        p.split("::")[0].endswith(s) for s in _H5_SUFFIXES
    )


def _restore_sigmoid(a, scale):
    """sigmoid(scale*x) -> sigmoid(x), i.e. sigmoid(logit(p)/scale). Elementwise."""
    p = np.clip(np.asarray(a, dtype=np.float32), 1e-6, 1.0 - 1e-6)
    logit = np.log(p) - np.log1p(-p)
    return (1.0 / (1.0 + np.exp(-logit / float(scale)))).astype(np.float32)


class _ArrayVolume(object):
    """Common (X, Y, Z, C) view over an array stored as (C, Z, Y, X) or (Z, Y, X).

    ``convention='banis'`` additionally converts BANIS-native affinity into what
    ABISS expects, mirroring dev/zebrafinch/upload_affinity_full_masked.py, which
    used to do this as a separate whole-volume pass before uploading:

      * edge shift ``dst[c, v] = src[c, v-1]`` along spatial axis ``c`` -- the model
        stores an edge on its SOURCE voxel (v -> v+1), ABISS reads it on the
        DESTINATION. Implemented by reading one extra voxel on each low face, so a
        chunk boundary pulls the true neighbour; only a real volume edge is
        zero-filled.
      * ``restore_sigmoid`` (optional): sigmoid(scale*x) -> sigmoid(x), needed only
        for affinity written with BANIS' `scale_sigmoid` (scale 0.2). Inference that
        emits a plain sigmoid needs no restore, so this defaults to off.
      * channel reversal ``[z,y,x] -> [x,y,z]``: ABISS reads channel 0 as
        x-affinity, the model emits channel 0 as z-affinity.
      * clip to [0, 1].

    The FFN tissue/border keep-mask that script also applied is deliberately NOT
    reproduced: it was measured to be ~inert for reconstruction quality and it
    needs volumes (tissue mask, raw EM) that are not available here.
    """

    def __init__(self, array, writable=False, label="", convention=None,
                 restore_sigmoid_scale=None):
        self._a = array
        self._writable = writable
        self._label = label
        self._convention = (convention or "none").lower()
        if self._convention not in ("none", "banis"):
            raise ValueError(
                "%s: unknown affinity convention %r (expected 'none' or 'banis')"
                % (label, convention)
            )
        self._restore_scale = restore_sigmoid_scale
        if array.ndim == 4:
            self._channels = int(array.shape[0])
            if self._convention == "banis" and self._channels != 3:
                # The conversion reverses the whole channel axis, so a 4-channel
                # affinity+myelin volume (which ABISS supports) would come out with
                # myelin in channel 0 and the affinities shifted -- plausible-looking
                # but wrong. Refuse rather than corrupt.
                raise ValueError(
                    "%s: AFF_CONVENTION='banis' requires exactly 3 channels, got %d. "
                    "Convert the affinity separately, or drop the auxiliary channel."
                    % (label, self._channels)
                )
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
            if self._convention == "banis":
                return self._read_banis(xs, ys, zs, cs)
            block = self._a[cs, zs, ys, xs]  # (C, Z, Y, X)
            block = np.asarray(block)
            if block.ndim == 3:  # a single channel was selected by an int
                block = block[np.newaxis, ...]
            return np.transpose(block, (3, 2, 1, 0))  # -> (X, Y, Z, C)
        block = np.asarray(self._a[zs, ys, xs])  # (Z, Y, X)
        return np.transpose(block, (2, 1, 0))[..., np.newaxis]

    def _read_banis(self, xs, ys, zs, cs):
        """Read with a 1-voxel low-side margin, apply the BANIS->ABISS conversion."""
        zdim, ydim, xdim = (int(v) for v in self._a.shape[-3:])
        starts, stops, pads = [], [], []
        for sl, dim in ((zs, zdim), (ys, ydim), (xs, xdim)):
            start, stop, step = sl.indices(dim) if isinstance(sl, slice) else (sl, sl + 1, 1)
            if step != 1:
                raise ValueError("%s: strided reads are not supported" % self._label)
            # One extra voxel on the low side feeds dst[v] = src[v-1]; at v == 0
            # there is no source, so pad with a zero plane instead (volume edge).
            margin = 1 if start > 0 else 0
            starts.append(start - margin)
            stops.append(stop)
            pads.append(1 - margin)  # zero-plane needed when we could not read one

        block = np.asarray(self._a[:, starts[0]:stops[0], starts[1]:stops[1],
                                   starts[2]:stops[2]]).astype(np.float32)
        if any(pads):
            block = np.pad(block, ((0, 0),) + tuple((p, 0) for p in pads), mode="constant")

        if self._restore_scale is not None:
            block = _restore_sigmoid(block, self._restore_scale)

        shifted = np.zeros_like(block)
        for c in range(min(3, block.shape[0])):
            dst = [slice(None)] * 4
            src = [slice(None)] * 4
            dst[0] = src[0] = c
            dst[c + 1] = slice(1, None)
            src[c + 1] = slice(0, -1)
            shifted[tuple(dst)] = block[tuple(src)]
        # Drop the margin plane we added for the shift.
        shifted = shifted[:, 1:, 1:, 1:]
        shifted = np.clip(shifted[::-1], 0.0, 1.0)  # channels [z,y,x] -> [x,y,z]
        out = np.transpose(shifted, (3, 2, 1, 0))  # -> (X, Y, Z, C)
        if not isinstance(cs, slice) or cs != slice(None):
            out = out[..., cs]
            if out.ndim == 3:
                out = out[..., np.newaxis]
        return out

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

    def __init__(self, path, convention=None, restore_sigmoid_scale=None):
        import h5py

        raw = _strip_scheme(str(path))
        dataset = None
        if "::" in raw:
            raw, dataset = raw.split("::", 1)
        # locking=False is the RELIABLE form. The HDF5_USE_FILE_LOCKING env var set at
        # the top of this module only works if nothing imported h5py first, since the
        # C library reads it once at initialization; this kwarg is per-open and cannot
        # be defeated by import order. Requires h5py >= 3.5.
        try:
            self._handle = h5py.File(raw, "r", locking=False)
        except TypeError:  # h5py < 3.5: fall back to the env var set on import
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
            self._handle[dataset], writable=False,
            label="HDF5 %s::%s" % (raw, dataset),
            convention=convention, restore_sigmoid_scale=restore_sigmoid_scale,
        )


class ZarrVolume(_ArrayVolume):
    """Zarr volume. Concurrent writes are safe only for disjoint chunk-aligned regions."""

    def __init__(self, path, writable=False, convention=None, restore_sigmoid_scale=None):
        import zarr

        raw = _strip_scheme(str(path))
        # mode="a" creates a store on a typo; readers get "r" so a wrong path fails loudly.
        self._array = zarr.open(raw, mode="a" if writable else "r")
        super(ZarrVolume, self).__init__(
            self._array, writable=writable, label="zarr %s" % raw,
            convention=convention, restore_sigmoid_scale=restore_sigmoid_scale,
        )

    def __setitem__(self, key, value):
        if self._writable:
            self._require_storage_chunk_alignment(key)
        super(ZarrVolume, self).__setitem__(key, value)

    def _require_storage_chunk_alignment(self, key):
        """Enforce the concurrency precondition instead of only documenting it.

        Two workers writing DIFFERENT logical chunks that land in the SAME zarr
        storage chunk read-modify-write that chunk concurrently, and one write is
        lost -- silently, and only under load. Alignment is what makes the writes
        independent, so check it rather than assert it in a docstring.
        """
        chunks = getattr(self._array, "chunks", None)
        if not chunks:
            return
        # Storage is (C,Z,Y,X) or (Z,Y,X); the incoming key is (X,Y,Z[,C]).
        spatial = list(chunks[-3:])  # (Z, Y, X)
        xs, ys, zs, _ = self._xyz_slices(key)
        shape = self._array.shape[-3:]  # (Z, Y, X)
        for axis, (sl, csize, extent) in enumerate(
            zip((zs, ys, xs), spatial, shape)
        ):
            start = sl.start or 0
            stop = extent if sl.stop is None else sl.stop
            # A ragged final block legitimately stops at the volume edge.
            if start % csize or (stop % csize and stop != extent):
                name = "ZYX"[axis]
                raise ValueError(
                    "%s: write [%d:%d) on axis %s is not aligned to the storage "
                    "chunk size %d. Concurrent ABISS chunk writes would share a "
                    "storage chunk and lose data. Recreate the array with chunks "
                    "that divide the ABISS CHUNK_SIZE."
                    % (self._label, start, stop, name, csize)
                )



class _ChunkedH5Array(object):
    """The 726-file grid-1008 affinity store, presented as one (C, Z, Y, X) array.

    Chunked inference writes one HDF5 per grid cell (``chunk_z2_y3_x3.h5``), each
    holding the halo-trimmed interior. ABISS wants a single volume. Building one
    would duplicate 2.7 TB -- the exact copy this backend exists to avoid -- so
    assemble reads on the fly instead.

    Only basic slicing is supported, which is all _ArrayVolume needs; because this
    sits UNDER _ArrayVolume, the BANIS conversion (including its 1-voxel low-side
    margin, which may fall in the neighbouring file) works across file seams for
    free.
    """

    _NAME_RE = re.compile(r"^chunk_z(\d+)_y(\d+)_x(\d+)\.h5$")
    _MAX_OPEN = 8

    def __init__(self, directory, dataset=None):
        import h5py

        self._dir = directory
        self._dataset = dataset
        self._h5py = h5py
        self._open = collections.OrderedDict()

        self._index = {}
        for name in os.listdir(directory):
            m = self._NAME_RE.match(name)
            if m:
                self._index[tuple(int(v) for v in m.groups())] = os.path.join(directory, name)
        if not self._index:
            raise ValueError("%s: no chunk_z*_y*_x*.h5 files found" % directory)

        probe_key = min(self._index)
        with h5py.File(self._index[probe_key], "r") as f:
            if self._dataset is None:
                names = [k for k in f if isinstance(f[k], h5py.Dataset)]
                if len(names) != 1:
                    raise ValueError(
                        "%s: expected exactly one dataset per chunk, got %r" % (directory, names)
                    )
                self._dataset = names[0]
            ds = f[self._dataset]
            if ds.ndim != 4:
                raise ValueError(
                    "%s: expected (C, Z, Y, X) chunks, got %r" % (directory, ds.shape)
                )
            self._channels = int(ds.shape[0])
            self._cshape = tuple(int(v) for v in ds.shape[1:])  # (Z, Y, X)
            self.dtype = ds.dtype
            # Placement is derived from the filename grid index; the attrs are
            # authoritative, so verify the two agree rather than trusting the name.
            start = ds.attrs.get("chunk_start_zyx")
            if start is not None:
                expected = [probe_key[i] * self._cshape[i] for i in range(3)]
                got = json.loads(str(start))
                if list(got) != expected:
                    raise ValueError(
                        "%s: chunk_start_zyx %r does not match grid index %r x chunk "
                        "shape %r; the filename grid is not the storage layout."
                        % (directory, got, probe_key, self._cshape)
                    )

        grid = [max(k[i] for k in self._index) + 1 for i in range(3)]
        self._grid = tuple(grid)
        self.shape = (self._channels,) + tuple(
            grid[i] * self._cshape[i] for i in range(3)
        )
        self.ndim = 4

    def _handle(self, key):
        handle = self._open.pop(key, None)
        if handle is None:
            handle = self._h5py.File(self._index[key], "r", locking=False)
            if len(self._open) >= self._MAX_OPEN:
                _, old = self._open.popitem(last=False)
                old.close()
        self._open[key] = handle
        return handle

    @staticmethod
    def _bounds(sl, dim):
        if isinstance(sl, slice):
            start, stop, step = sl.indices(dim)
            if step != 1:
                raise ValueError("strided reads are not supported")
            return start, stop
        return sl, sl + 1

    def __getitem__(self, key):
        if not isinstance(key, tuple):
            key = (key,)
        key = key + (slice(None),) * (4 - len(key))
        c0, c1 = self._bounds(key[0], self.shape[0])
        box = [self._bounds(key[i + 1], self.shape[i + 1]) for i in range(3)]

        out = np.zeros(
            (c1 - c0,) + tuple(hi - lo for lo, hi in box), dtype=self.dtype
        )
        ranges = []
        for axis, (lo, hi) in enumerate(box):
            first = lo // self._cshape[axis]
            last = max(first, (hi - 1) // self._cshape[axis])
            ranges.append(range(first, last + 1))

        for gz in ranges[0]:
            for gy in ranges[1]:
                for gx in ranges[2]:
                    gk = (gz, gy, gx)
                    if gk not in self._index:
                        continue  # absent cell stays zero, like a missing tile
                    src, dst = [], []
                    for axis, gi in enumerate(gk):
                        base = gi * self._cshape[axis]
                        lo, hi = box[axis]
                        s0 = max(lo, base)
                        s1 = min(hi, base + self._cshape[axis])
                        src.append(slice(s0 - base, s1 - base))
                        dst.append(slice(s0 - lo, s1 - lo))
                    ds = self._handle(gk)[self._dataset]
                    out[:, dst[0], dst[1], dst[2]] = ds[
                        c0:c1, src[0], src[1], src[2]
                    ]
        return out


class H5ChunkStoreVolume(_ArrayVolume):
    """Read-only view over a directory of grid-indexed affinity HDF5 chunks."""

    def __init__(self, path, convention=None, restore_sigmoid_scale=None):
        raw = _strip_scheme(str(path))
        dataset = None
        if "::" in raw:
            raw, dataset = raw.split("::", 1)
        array = _ChunkedH5Array(raw, dataset=dataset)
        super(H5ChunkStoreVolume, self).__init__(
            array, writable=False, label="h5-chunkstore %s" % raw,
            convention=convention, restore_sigmoid_scale=restore_sigmoid_scale,
        )


def is_h5_chunkstore_path(path):
    """A directory holding chunk_z*_y*_x*.h5 files (chunked-inference output)."""
    raw = _strip_scheme(str(path)).split("::", 1)[0]
    if not os.path.isdir(raw):
        return False
    try:
        return any(_ChunkedH5Array._NAME_RE.match(n) for n in os.listdir(raw))
    except OSError:
        return False


def _affinity_conversion_for(path):
    """Convention/restore-scale to apply, from the ABISS param file.

    Only the volume named by ``AFF_PATH`` is converted -- watershed/segmentation
    volumes read through the same backend must not be touched. Opt in with:

        "AFF_CONVENTION": "banis"          # edge v->v-1 + channels [z,y,x]->[x,y,z]
        "AFF_RESTORE_SIGMOID": 0.2         # optional, only for scale_sigmoid output
    """
    param_json = os.environ.get("PARAM_JSON")
    if not param_json or not os.path.exists(param_json):
        return None, None
    try:
        import json

        with open(param_json) as handle:
            param = json.load(handle)
    except Exception:
        return None, None
    aff_path = param.get("AFF_PATH")
    if not aff_path or str(aff_path) != str(path):
        return None, None
    convention = param.get("AFF_CONVENTION")
    scale = param.get("AFF_RESTORE_SIGMOID")
    return convention, (float(scale) if scale is not None else None)


def open_volume(path, **kwargs):
    """Open ``path`` with the backend its name implies.

    Unrecognised paths fall through to CloudVolume, so precomputed/gs:// behaviour is
    unchanged. CloudVolume-only kwargs are REJECTED rather than ignored by the
    single-scale adapters when they would change what gets read (see below).
    """
    text = str(path)
    convention = kwargs.pop("convention", None)
    restore = kwargs.pop("restore_sigmoid_scale", None)
    # Pop before the CloudVolume fallback: `writable` is ours, and forwarding it to
    # the CloudVolume constructor is a TypeError.
    writable = bool(kwargs.pop("writable", False))
    if convention is None:
        convention, restore = _affinity_conversion_for(text)
    if is_h5_path(text) or is_zarr_path(text) or is_h5_chunkstore_path(text):
        # These backends expose a single scale. Silently ignoring a mip would read
        # full-resolution voxels while the chunk coordinates refer to another scale,
        # which degrades the segmentation without ever crashing.
        mip = kwargs.get("mip", 0)
        if isinstance(mip, (list, tuple)):
            mip = 0 if len(mip) == 3 else mip  # a resolution triple selects scale 0 here
        if mip not in (0, None):
            raise ValueError(
                "%s: mip=%r requested but HDF5/zarr backends are single-scale. "
                "Set AFF_RESOLUTION to 0 or use a precomputed layer." % (text, mip)
            )
        # fill_missing / bounded change what a read RETURNS, so ignoring them is the
        # same silent-corruption class as ignoring mip: fill_missing=True asks for
        # zeros outside the stored data, and these adapters would instead raise or
        # clip. HDF5/zarr here are dense and fully materialized, so the only safe
        # values are the defaults.
        for name, default in (("fill_missing", False), ("bounded", True)):
            value = kwargs.get(name, default)
            if bool(value) != default:
                raise ValueError(
                    "%s: %s=%r is a CloudVolume behaviour the HDF5/zarr backends do "
                    "not implement. Remove it, or use a precomputed layer."
                    % (text, name, value)
                )
    if is_h5_chunkstore_path(text):
        return H5ChunkStoreVolume(
            text, convention=convention, restore_sigmoid_scale=restore
        )
    if is_h5_path(text):
        return H5Volume(text, convention=convention, restore_sigmoid_scale=restore)
    if is_zarr_path(text):
        return ZarrVolume(
            text,
            writable=writable,
            convention=convention,
            restore_sigmoid_scale=restore,
        )
    from cloudvolume import CloudVolume

    return CloudVolume(text, **kwargs)
