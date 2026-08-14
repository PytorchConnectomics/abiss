import sys
from chunk_utils import read_inputs
from cut_chunk_common import load_data, cut_data, pad_data, save_raw_data
from augment_affinity import adjust_affinitymap, warp_z
import os
import numpy
from nucleus_overlay import apply_nucleus_competition

def chunk_origin(bbox):
    offset = bbox[0:3]
    for i in range(3):
        if boundary_flags[i] == 1:
            offset[i] -= 1
    return offset

def write_metadata(fn, offset, size, ac_offset):
    with open(fn, "w") as f:
        f.write(" ".join([str(x) for x in offset]))
        f.write("\n")
        f.write(" ".join([str(x) for x in size]))
        f.write("\n")
        f.write(str(ac_offset))

def validate_nucleus_cutout(data, expected_shape):
    if tuple(data.shape[:3]) != tuple(expected_shape):
        raise ValueError(
            f"nucleus cutout shape {data.shape[:3]} does not match "
            f"segmentation shape {tuple(expected_shape)}")
    if not numpy.issubdtype(data.dtype, numpy.integer):
        raise TypeError(f"nucleus data must have an integer dtype, got {data.dtype}")
    if numpy.issubdtype(data.dtype, numpy.signedinteger) and numpy.any(data < 0):
        raise ValueError("nucleus data contains a negative instance id")
    if data.size and numpy.max(data) > 0xFFFFFFFF:
        raise ValueError("nucleus data contains an instance id greater than 0xFFFFFFFF")
    return data.astype(numpy.uint32, order='F', copy=False)

def nucleus_axis_vector(params, name, default, positive=False):
    value = params.get(name, default)
    if (not isinstance(value, (list, tuple)) or len(value) != 3
            or any(isinstance(item, bool) or not isinstance(item, int)
                   for item in value)):
        raise ValueError(f"{name} must be a three-element integer [z,y,x] list")
    if positive and any(item <= 0 for item in value):
        raise ValueError(f"{name} values must all be positive")
    return tuple(value)

def cut_nucleus_data(data, start_coord, end_coord, padding, ratio_zyx,
                     offset_zyx):
    if len(data.shape) != 4 or data.shape[3] != 1:
        raise ValueError(
            f"nucleus volume must have one channel, got shape {tuple(data.shape)}")

    if ratio_zyx == (1, 1, 1) and offset_zyx == (0, 0, 0):
        return cut_data(data, start_coord, end_coord, padding)

    # Chunk coordinates and volume adapters use [x,y,z], while the public
    # transform follows the microscopy convention [z,y,x].
    ratio_xyz = tuple(reversed(ratio_zyx))
    offset_xyz = tuple(reversed(offset_zyx))
    source_indices = []
    for axis in range(3):
        target = numpy.arange(start_coord[axis], end_coord[axis], dtype=numpy.int64)
        source = numpy.floor_divide(
            target + ratio_xyz[axis] // 2, ratio_xyz[axis])
        source_indices.append(source + offset_xyz[axis])

    # Nearest-neighbour rounding overshoots by up to ratio//2 at the FAR edge of the
    # volume: mip0 z=5698 with ratio 4 maps to (5698+2)//4 = 1425, one past a 1425-long
    # axis. Clamping is the correct boundary behaviour for nearest-neighbour sampling and
    # is what the whole-volume run needs; a genuinely wrong NUC_OFFSET still raises,
    # because that overshoots by far more than the rounding slack.
    for axis, indices in enumerate(source_indices):
        if not indices.size:
            continue
        limit = data.shape[axis] - 1
        slack = ratio_xyz[axis]          # tolerate at most one source voxel of rounding
        if indices[0] < -slack or indices[-1] > limit + slack:
            raise ValueError(
                f"NUC_RATIO/NUC_OFFSET map axis {axis} to "
                f"[{int(indices[0])}, {int(indices[-1])}], outside nucleus "
                f"volume shape {tuple(data.shape[:3])} by more than one source voxel")
        numpy.clip(indices, 0, limit, out=indices)

    source_slices = tuple(
        slice(int(indices[0]), int(indices[-1]) + 1)
        for indices in source_indices)
    source_cutout = numpy.asarray(data[source_slices])
    relative_indices = tuple(
        indices - indices[0] for indices in source_indices)
    upsampled = source_cutout[numpy.ix_(*relative_indices)]
    return pad_data(upsampled, padding)

param = read_inputs(sys.argv[1])
global_param = read_inputs(os.environ['PARAM_JSON'])
bbox = param["bbox"]
aff_bbox = bbox[:]
aff_bbox[2] = warp_z(bbox[2])
aff_bbox[5] = aff_bbox[2] + (bbox[5] - bbox[2])
print(bbox)
print(aff_bbox)
ac_offset = param["ac_offset"]
boundary_flags = param["boundary_flags"]

aff = load_data(global_param['AFF_PATH'], mip=global_param['AFF_RESOLUTION'], fill_missing=global_param.get('AFF_FILL_MISSING', False))
aff_cutout = adjust_affinitymap(aff, aff_bbox, boundary_flags, 0, 1)

save_raw_data("aff.raw", aff_cutout)
del aff_cutout

start_coord = bbox[0:3]
end_coord = [bbox[i+3]+1-boundary_flags[i+3] for i in range(3)]

seg = load_data(os.environ['WS_PATH'], mip=global_param['AFF_RESOLUTION'], fill_missing=global_param.get('WS_FILL_MISSING', False))
seg_cutout = cut_data(seg, start_coord, end_coord, boundary_flags)
seg_cutout = apply_nucleus_competition(
    seg_cutout,
    [start_coord[i] - boundary_flags[i] for i in range(3)],
    global_param)
save_raw_data("seg.raw", seg_cutout)

if "SEM_PATH" in global_param:
    sem = load_data(global_param['SEM_PATH'], mip=global_param['AFF_RESOLUTION'], fill_missing=global_param.get('SEM_FILL_MISSING', False))
    sem_cutout = cut_data(sem, start_coord, end_coord, boundary_flags)
    save_raw_data("sem.raw", sem_cutout)

if "NUC_PATH" in global_param:
    nuc_ratio = nucleus_axis_vector(
        global_param, "NUC_RATIO", [1, 1, 1], positive=True)
    nuc_offset = nucleus_axis_vector(
        global_param, "NUC_OFFSET", [0, 0, 0])
    nuc = load_data(global_param['NUC_PATH'],
                    mip=global_param.get('NUC_MIP', global_param['AFF_RESOLUTION']),
                    fill_missing=global_param.get('NUC_FILL_MISSING', False))
    nuc_cutout = cut_nucleus_data(
        nuc, start_coord, end_coord, boundary_flags, nuc_ratio, nuc_offset)
    nuc_cutout = validate_nucleus_cutout(nuc_cutout, seg_cutout.shape[:3])
    save_raw_data("nuc.raw", nuc_cutout)

#save_data("aff.h5", aff_cutout)
#save_data("seg.h5", seg_cutout)

write_metadata("param.txt", chunk_origin(bbox), seg_cutout.shape[0:3], ac_offset)
with open("chunk_offset.txt", "w") as f:
    f.write(str(ac_offset))
