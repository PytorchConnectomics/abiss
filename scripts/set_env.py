import os
import re
import sys
import json
import cloudfiles.paths


def sanitize_runname(runname):
    sanitized = re.sub(r'[^0-9a-zA-Z]+', '_', runname)
    return sanitized.strip('_')


def default_io_cmd(path):
    try:
        d = cloudfiles.paths.extract(path)
    except:
        return "cloudfiles cp -r"

    if d.protocol == "gs":
        return "gsutil -m cp -r"
    elif d.protocol == "s3":
        if d.host:
            return f"s5cmd --endpoint-url {d.host} cp"
        else:
            return "s5cmd cp"
    else:
        return "cloudfiles cp -r"


env = ["SCRATCH_PATH", "CHUNKMAP_INPUT", "CHUNKMAP_OUTPUT", "AFF_PATH", "AFF_MIP", "SEM_PATH", "SEM_MIP", "WS_PATH", "SEG_PATH", "WS_HIGH_THRESHOLD", "WS_LOW_THRESHOLD", "WS_SIZE_THRESHOLD", "WS_DUST_THRESHOLD", "AGG_THRESHOLD", "GT_PATH", "CLEFT_PATH", "MYELIN_THRESHOLD", "ADJUSTED_AFF_PATH", "CHUNKED_AGG_OUTPUT", "CHUNKED_SEG_PATH", "REDIS_SERVER", "REDIS_DB", "STATSD_HOST", "STATSD_PORT", "STATSD_PREFIX", "PARANOID", "BOTO_CONFIG", "UPLOAD_CMD", "DOWNLOAD_CMD", "IO_SCRATCH_PATH", "REMAP_SIZE_MAP_THRESHOLD"]

# Agglomeration heuristics. Until now these were compile-time literals in
# src/agg/mean_aggl.cpp, so tuning one meant editing the source and rebuilding; only
# AGG_THRESHOLD (-> input_aff_threshold, argv[1]) was reachable from the param JSON.
# Each is OPTIONAL: the export loop below skips keys absent from the JSON, and the binary
# falls back to its original literal, so an existing param JSON produces an identical run.
env += [
    "AGG_SIZE_AFF_THRESHOLD", "AGG_SMALL_VOXEL_THRESHOLD", "AGG_LARGE_VOXEL_THRESHOLD",
    "AGG_SEM_AFF_THRESHOLD", "AGG_SEM_TOTAL_SIGNAL_THRESHOLD", "AGG_SEM_DOMINANT_SIGNAL_RATIO",
    "AGG_TWIG_AFF_THRESHOLD_DELTA", "AGG_TWIG_VOXEL_THRESHOLD", "AGG_TWIG_AREA_THRESHOLD",
    "AGG_HEURISTICS_AFF_THRESHOLD", "AGG_STARTING_AFF_THRESHOLD", "AGG_STEP",
    "AGG_MIN_EDGES", "AGG_NUM_PARTITIONS",
]

with open(sys.argv[1]) as f:
    data = json.load(f)

data["STATSD_PREFIX"] = sanitize_runname(data["NAME"])

for s in ["SCRATCH", "WS", "SEG"]:
    prefix = "{}_PREFIX".format(s)
    path = "{}_PATH".format(s)
    if path not in data:
        data[path] = data[prefix]+data["NAME"]

if "UPLOAD_CMD" not in data:
    data["UPLOAD_CMD"] = data.get("DOWNLOAD_CMD", default_io_cmd(data["SCRATCH_PATH"]))

if "DOWNLOAD_CMD" not in data:
    data["DOWNLOAD_CMD"] = data.get("UPLOAD_CMD", default_io_cmd(data["SCRATCH_PATH"]))

if data["UPLOAD_CMD"].startswith("cloudfiles") and data["DOWNLOAD_CMD"].startswith("cloudfiles"):
    data["IO_SCRATCH_PATH"] = data["SCRATCH_PATH"]
else:
    extracted_path = cloudfiles.paths.extract(data["SCRATCH_PATH"])
    if extracted_path.alias or extracted_path.host:
        data["IO_SCRATCH_PATH"] = cloudfiles.paths.asprotocolpath(extracted_path._replace(alias=None, host=None))
    else:
        data["IO_SCRATCH_PATH"] = data["SCRATCH_PATH"]

if "CHUNKMAP_OUTPUT" in data:
    d = cloudfiles.paths.extract(data["CHUNKMAP_OUTPUT"])
    data["CHUNKMAP_OUTPUT"] = cloudfiles.paths.asprotocolpath(d._replace(alias=None, host=None))

if "CHUNKMAP_INPUT" not in data:
    data["CHUNKMAP_INPUT"] = os.path.join(data["SCRATCH_PATH"], "ws", "chunkmap")

if data.get("CHUNKED_AGG_OUTPUT", False):
    data["CHUNKED_AGG_OUTPUT"] = 1
else:
    data["CHUNKED_AGG_OUTPUT"] = 0

if data.get("PARANOID", False):
    data["PARANOID"] = 1
else:
    data["PARANOID"] = 0

if "WS_DUST_THRESHOLD" not in data:
    data["WS_DUST_THRESHOLD"] = data["WS_SIZE_THRESHOLD"]

if "REMAP_SIZE_MAP_THRESHOLD" not in data:
    data["REMAP_SIZE_MAP_THRESHOLD"] = 100000

if "gsutil-secret.json" in data.get("MOUNT_SECRETS", []):
    data["BOTO_CONFIG"] = "~/gsutil.boto"

for e in env:
    if e in data:
        print('export {}="{}"'.format(e, data[e]))
