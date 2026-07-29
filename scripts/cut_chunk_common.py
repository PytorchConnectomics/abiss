from cloudvolume import CloudVolume
from volume_backends import open_volume
import chunk_utils as cu
import numpy
import os


def load_data(url, **kwargs):
    print("volume url: ", url)

    # Dispatches on the path: *.h5/*.hdf5 and *.zarr use the local adapters, anything
    # else (precomputed, gs://, ...) falls through to CloudVolume unchanged. This lets
    # ABISS read affinity straight out of inference output instead of requiring a
    # precomputed copy of the whole volume.
    return open_volume(url, cache=False, **kwargs)
    #return CloudVolumeGSUtil(url, fill_missing=True)

def load_gt_data(url, mip=0):
    print("volume url: ", url)
    print("mip level: ", mip)

    return open_volume(url, fill_missing=True, bounded=False, mip=mip)

def affinity_dtype():
    """numpy dtype matching the compiled `aff_t`.

    src/global_types.h: `aff_t` is float, or double under -DDOUBLE. save_raw_data()
    writes data.dtype verbatim and the binaries mmap aff.raw as aff_t, so the two
    must agree or the file has the wrong element width and is silently misread.
    Set ABISS_AFF_DTYPE=float64 when running binaries built with -DDOUBLE.
    """
    name = os.environ.get("ABISS_AFF_DTYPE", "float32")
    if name not in ("float32", "float64"):
        raise ValueError(
            f"ABISS_AFF_DTYPE must be float32 or float64, got {name!r}")
    return name


def save_raw_data(fn, data):
    f = numpy.memmap(fn, dtype=data.dtype, mode='w+', order='F', shape=data.shape)
    f[:] = data[:]
    del f
    # Catch an element-width mismatch here rather than as garbage inside the binary.
    expected = int(numpy.prod(data.shape)) * numpy.dtype(data.dtype).itemsize
    actual = os.path.getsize(fn)
    if actual != expected:
        raise RuntimeError(
            f"{fn}: wrote {actual} bytes, expected {expected} "
            f"({data.shape} of {data.dtype})")

def pad_data(data, padding):
    pad = [[padding[i], padding[i+3]] for i in range(3)]
    if len(data.shape) == 3:
        return numpy.pad(data, pad, 'constant', constant_values=0)
    elif len(data.shape) == 4:
        return numpy.pad(data, pad+[[0,0]], 'constant', constant_values=0)
    else:
        raise RuntimeError("encountered array of dimension " + str(len(data.shape)))

def convert_and_scale_integer_data(data, dtype_out):
    if numpy.issubdtype(data.dtype, numpy.integer):
        print(f"convert {data.dtype} to {dtype_out}")
        info = numpy.iinfo(data.dtype)
        return (data.astype(dtype_out, order='F') - info.min)/(info.max - info.min)
    if data.dtype != numpy.dtype(dtype_out):
        # Float input still has to MATCH the C++ ABI: save_raw_data() writes
        # data.dtype verbatim and the binaries mmap aff.raw as aff_t (float, or
        # double under -DDOUBLE). A float16 affinity -- what pytc inference writes --
        # would otherwise produce a half-length file that is silently misread.
        print(f"convert {data.dtype} to {dtype_out}")
        return data.astype(dtype_out, order='F', copy=False)
    return data

def cut_data(data, start_coord, end_coord, padding):
    bb = tuple(slice(start_coord[i], end_coord[i]) for i in range(3))
    global_param = cu.read_inputs(os.environ['PARAM_JSON'])
    if data.shape[3] == 1:
        if numpy.issubdtype(data.dtype, numpy.floating):
            pmap = numpy.squeeze(data[bb])
            affinity = [numpy.minimum(numpy.roll(pmap, shift=1, axis=axis), pmap) for axis in range(3)]
            stacked = numpy.stack(affinity, axis=-1)
            return pad_data(convert_and_scale_integer_data(stacked, affinity_dtype()), padding)
        else:
            return pad_data(data[bb], padding)
    elif data.shape[3] == 3:
        return pad_data(convert_and_scale_integer_data(data[bb+(slice(0,3),)], affinity_dtype()), padding)
    elif data.shape[3] == 4: #0-2 affinity, 3 myelin
        cutout = data[bb+(slice(0,4),)]
        if numpy.issubdtype(cutout.dtype, numpy.floating):
            cutout = convert_and_scale_integer_data(cutout, affinity_dtype())
        return pad_data(cutout, padding)
    else:
        aff_channels = global_param.get('AFF_CHANNELS', 3)
        if data.shape[3] >= aff_channels:
            return pad_data(convert_and_scale_integer_data(data[bb+(slice(0,aff_channels),)], affinity_dtype()), padding)
        raise RuntimeError("encountered array of dimension " + str(len(data.shape)))


