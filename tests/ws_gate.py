#!/usr/bin/env python3
"""Bit-exact ws gate and compute-node memory driver. Never writes stock builds.

Local: --mode local --python-repo PATH
Benchmark: submit ws_benchmark.sbatch (each measurement is a separate srun step).
All generated data and logs live below build_mem/gate_scratch by default.
"""

import argparse
import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path(
    "/projects/weilab/weidf/lib/pytorch_connectomics/outputs/"
    "liconn_final_banis_plus_tube/20260728_032436/test_step=00200000/"
    "val/raw_x1_ch0-1-2.h5"
)
# Stock binaries (92abc91) kept beside the installed memory-lean build/ws, build64/ws64.
STOCK_WS = "build/ws.stock-92abc91"
STOCK_WS64 = "build64/ws64.stock-92abc91"
STOCK_MD5 = "bf4c7343dccf86e60d78f772e583492d"


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def benchmark_thresholds(crop):
    # Float16 percentile index arithmetic overflows on large crops.
    low, high = map(float, np.percentile(np.asarray(crop, dtype=np.float32), [20, 94]))
    require(
        np.isfinite(low) and np.isfinite(high) and low < high,
        f"Invalid benchmark thresholds: low={low}, high={high}",
    )
    return high, low


def driver(repo):
    spec = importlib.util.spec_from_file_location(
        "run_abiss_volume", repo / "scripts/run_abiss_volume.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def digest(path):
    h = hashlib.md5()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def compare(left, right, layout, u32=False):
    a = {p.name: p for p in left.glob("*.data")}
    b = {p.name: p for p in right.glob("*.data")}
    require(
        a.keys() == b.keys(),
        f"file set mismatch: {left} {right}: {a.keys() ^ b.keys()}",
    )
    padding = []
    for name in sorted(a):
        if name.startswith("dend_"):
            require(
                a[name].stat().st_size == b[name].stat().st_size,
                f"{name}: dend size mismatch",
            )
            size = layout["size"]
            require(
                size == 24 and a[name].stat().st_size % size == 0,
                f"{name}: invalid record size",
            )
            av = np.fromfile(a[name], dtype=np.uint8).reshape(-1, size)
            bv = np.fromfile(b[name], dtype=np.uint8).reshape(-1, size)
            for field, width in (("score", 4), ("id1", 8), ("id2", 8)):
                off = layout[field]
                require(
                    np.array_equal(av[:, off : off + width], bv[:, off : off + width]),
                    f"{name}: {field} field bytes mismatch: {left} {right}",
                )
            if not np.array_equal(av, bv):
                padding.append(name)
        elif u32 and re.fullmatch(r"seg_gate(?:_\d+)?\.data", name):
            av = np.fromfile(a[name], dtype=np.uint64)
            bv = np.fromfile(b[name], dtype=np.uint32)
            require(
                np.array_equal(av.astype(np.uint32), bv),
                f"{name}: uint32 values mismatch",
            )
        else:
            require(
                a[name].stat().st_size == b[name].stat().st_size
                and digest(a[name]) == digest(b[name]),
                f"{name}: byte mismatch: {left} {right}",
            )
    return {"files": len(a), "dend_padding_differences": padding}


def fixtures():
    rng = np.random.default_rng(2026)
    shape = (3, 16, 24, 20)
    yield "zero", np.zeros(shape, np.float32), 20, 0
    blocks = np.full(shape, 0.8, np.float32)
    blocks[:, :, :, 10] = 0
    blocks[:, :, 12, :] = 0
    blocks[:, 8, :, :] = 0
    yield "blocks", blocks, 20, 0
    yield "ties", rng.integers(0, 11, shape).astype(np.float32) / 10, 20, 0
    yield "dust", rng.uniform(0, 0.8, shape).astype(np.float32), 12, 8
    yield "heavy", rng.uniform(0.1, 0.9, shape).astype(np.float32), 100000, 0
    yield "rounding", rng.choice(
        np.array([0.1, 0.2, 0.3, 1e-7, 0.7], np.float32), shape
    ), 20, 0


class Gate:
    def __init__(self, args):
        self.args = args
        self.scratch = args.scratch
        self.scratch.mkdir(parents=True, exist_ok=True)
        self.script = driver(args.python_repo)
        self.layout = json.loads(
            subprocess.check_output([ROOT / "build_mem/test_ws_bfs", "--dend-layout"])
        )
        self.results = []

    def record(self, name, **values):
        entry = {"case": name, **values}
        self.results.append(entry)
        print(json.dumps(entry), flush=True)
        (self.scratch / "local_results.json").write_text(
            json.dumps(self.results, indent=2)
        )

    def prepare(self, name, predictions):
        path = self.scratch / name
        path.mkdir(exist_ok=True)
        aff = self.script._to_abiss_affinity(predictions, channels=None)
        shape = self.script._write_affinity_with_halo(path / "aff.raw", aff)
        return path, shape

    def run(
        self,
        binary,
        source,
        shape,
        label,
        flags=(1,) * 6,
        offset=17,
        mode="max",
        thresholds=(0.2,),
        size=20,
        dust=0,
        tokens=(),
        expected=0,
    ):
        out = source / label
        out.mkdir(exist_ok=True)
        self.script._write_abiss_param_file(
            out / "param.txt", shape, list(flags), offset
        )
        cmd = [
            str(binary),
            str(out / "param.txt"),
            str(source / "aff.raw"),
            "0.95",
            "0.05",
            str(size),
            str(dust),
            "gate",
            mode,
            *map(str, thresholds),
            *tokens,
        ]
        with (out / "stdout.log").open("w") as log:
            result = subprocess.run(cmd, cwd=out, stdout=log, stderr=subprocess.STDOUT)
        require(
            result.returncode == expected,
            f"exit {result.returncode}, expected {expected}: {cmd}\n"
            f"{(out / 'stdout.log').read_text()}",
        )
        if expected == 0 and binary.parent == ROOT / "build_mem":
            stdout = (out / "stdout.log").read_text()
            for pattern in (
                r"^num of sv:\d+$",
                r"^size of rg:\d+$",
                r"^finished agglomeration in [\deE.+-]+ seconds$",
                r"^finished writing in [\deE.+-]+ seconds$",
            ):
                require(re.search(pattern, stdout, re.M), f"Missing stdout: {pattern}")
            if len(thresholds) > 1:
                for i, threshold in enumerate(thresholds):
                    pattern = (
                        rf"^merge threshold {i} \({threshold:g}\): sv=\d+ rg=\d+"
                        r" in [\deE.+-]+ seconds$"
                    )
                    require(
                        re.search(pattern, stdout, re.M), f"Missing stdout: {pattern}"
                    )
        return out

    def flags(self):
        caches = []
        for directory in ("build", "build64", "build_ref", "build_mem"):
            target = "ws64" if directory == "build64" else "ws"
            flags = (
                ROOT / directory / f"CMakeFiles/{target}.dir/flags.make"
            ).read_text()
            require(
                not any(
                    f in flags
                    for f in ("-ffast-math", "-fassociative-math", "-Ofast", "-mfma")
                ),
                f"Unsafe FP flags in {directory}: {flags}",
            )
            require("-march=nocona" in flags, f"unexpected arch: {directory}")
            actual = next(
                line for line in flags.splitlines() if line.startswith("CXX_FLAGS =")
            )
            caches.append(actual)
            self.record(f"flags/{directory}", flags=flags)
        require(len(set(caches)) == 1, "stock/reference/modified CXX_FLAGS differ")

    def determinism(self):
        _, data, size, dust = list(fixtures())[2]
        source, shape = self.prepare("determinism", data)
        dirs = [
            self.run(binary, source, shape, name, flags=(0,) * 6, size=size, dust=dust)
            for binary, name in [
                (ROOT / STOCK_WS, "stock_a"),
                (ROOT / STOCK_WS, "stock_b"),
                (ROOT / "build_ref/ws", "reference"),
            ]
        ]
        self.record(
            "determinism/stock_repeat", **compare(dirs[0], dirs[1], self.layout)
        )
        try:
            self.record(
                "determinism/stock_reference", **compare(dirs[0], dirs[2], self.layout)
            )
        except RuntimeError as error:
            self.record("determinism/stock_reference", mismatch=str(error))

    def contract(self, name, data, size, dust):
        source, shape = self.prepare(name, data)
        del data
        for width in ("ws", "ws64"):
            stock = ROOT / (STOCK_WS if width == "ws" else STOCK_WS64)
            for mode, thresholds in [
                ("max", (0.2,)),
                ("max", (0.1, 0.4, 0.8)),
                ("mean", (0.1, 0.4, 0.8)),
                ("p75", (0.1, 0.4, 0.8)),
            ]:
                for flags in ((1,) * 6, (0,) * 6, (0, 1, 0, 1, 0, 1)):
                    key = f"{name}/{width}/{mode}{len(thresholds)}/{''.join(map(str, flags))}"
                    dirs = []
                    for build, binary, tokens in (
                        ("ref", ROOT / "build_ref" / width, ()),
                        ("stock", stock, ()),
                        ("new", ROOT / "build_mem" / width, ()),
                        ("u32", ROOT / "build_mem" / width, ("--seg-dtype=uint32",)),
                    ):
                        dirs.append(
                            self.run(
                                binary,
                                source,
                                shape,
                                build,
                                flags=flags,
                                mode=mode,
                                thresholds=thresholds,
                                size=size,
                                dust=dust,
                                tokens=tokens,
                            )
                        )
                    result = {
                        "reference_new": compare(dirs[0], dirs[2], self.layout),
                        "reference_u32": compare(
                            dirs[0], dirs[3], self.layout, u32=True
                        ),
                    }
                    try:
                        result["stock_new"] = compare(dirs[1], dirs[2], self.layout)
                        result["stock_u32"] = compare(
                            dirs[1], dirs[3], self.layout, u32=True
                        )
                    except RuntimeError as error:
                        result["stock_mismatch"] = str(error)
                    # Keep per-case stdout, but discard the large output volumes after comparison.
                    logs = self.scratch / "logs" / key
                    logs.mkdir(parents=True, exist_ok=True)
                    for directory in dirs:
                        shutil.copyfile(
                            directory / "stdout.log", logs / f"{directory.name}.log"
                        )
                        shutil.rmtree(directory)
                    self.record(key, **result)

    def failures(self):
        _, data, size, dust = list(fixtures())[4]
        source, shape = self.prepare("bounds", data)
        for width in ("ws", "ws64"):
            binary = ROOT / "build_mem" / width
            base = self.run(
                binary,
                source,
                shape,
                width + "_counts",
                size=size,
                thresholds=(0.0, 0.99, 0.5),
                offset=0,
            )
            counts = [
                int(np.fromfile(base / f"meta_gate_{i}.data", np.uint64)[3])
                for i in range(3)
            ]
            require(0 < counts[0] < counts[1], f"heavy fixture must merge: {counts}")
            fit_offset = 2**32 - 1 - counts[0]
            ref = self.run(
                ROOT / "build_ref" / width,
                source,
                shape,
                width + "_bound_ref",
                size=size,
                thresholds=(0.0,),
                offset=fit_offset,
            )
            fit = self.run(
                binary,
                source,
                shape,
                width + "_fit",
                size=size,
                thresholds=(0.0,),
                offset=fit_offset,
                tokens=("--seg-dtype=uint32",),
            )
            self.record(
                width + "/exact_fit",
                compact_counts=counts,
                **compare(ref, fit, self.layout, True),
            )
            for label, offset, thresholds in [
                ("one_past", fit_offset + 1, (0.0,)),
                ("offset_past", 2**32, (0.0,)),
                ("batch_failure", fit_offset, (0.0, 0.99, 0.5)),
            ]:
                out = self.run(
                    binary,
                    source,
                    shape,
                    width + "_" + label,
                    size=size,
                    thresholds=thresholds,
                    offset=offset,
                    tokens=("--seg-dtype=uint32",),
                    expected=3,
                )
                files = sorted(p.name for p in out.glob("*.data"))
                require(
                    not list(out.glob("*.tmp")), f"temporary file after failure: {out}"
                )
                if label == "batch_failure":
                    require(
                        files
                        == [
                            "counts_gate_0.data",
                            "dend_gate_0.data",
                            "meta_gate_0.data",
                            "seg_gate_0.data",
                        ],
                        f"partial batch artifacts: {files}",
                    )
                    require(
                        (out / "seg_gate_0.data").stat().st_size
                        == int(np.prod(np.array(shape) - 2)) * 4,
                        "earlier batch segmentation incomplete",
                    )
                else:
                    require(not files, f"artifacts after overflow: {files}")
                self.record(
                    width + "/" + label,
                    exit=3,
                    files=files,
                    message=(out / "stdout.log").read_text().splitlines()[-1],
                )
            for tokens in [
                ("--seg-dtype=invalid",),
                ("--seg-dtype=uint32", "--seg-dtype=uint64"),
            ]:
                out = self.run(
                    binary,
                    source,
                    shape,
                    width + "_invalid_" + str(len(tokens)),
                    tokens=tokens,
                    expected=2,
                )
                self.record(
                    width + "/invalid_token",
                    tokens=tokens,
                    exit=2,
                    message=(out / "stdout.log").read_text(),
                )
            # Token before function, between function/thresholds, and after thresholds.
            for pos in (8, 9, 10):
                out = source / f"{width}_position{pos}"
                out.mkdir(exist_ok=True)
                self.script._write_abiss_param_file(
                    out / "param.txt", shape, [1] * 6, fit_offset
                )
                cmd = [
                    str(binary),
                    str(out / "param.txt"),
                    str(source / "aff.raw"),
                    "0.95",
                    "0.05",
                    str(size),
                    "0",
                    "gate",
                    "max",
                    "0.0",
                ]
                cmd.insert(pos, "--seg-dtype=uint32")
                with (out / "stdout.log").open("w") as log:
                    subprocess.run(
                        cmd, cwd=out, check=True, stdout=log, stderr=subprocess.STDOUT
                    )
                self.record(
                    f"{width}/token_position/{pos}",
                    **compare(ref, out, self.layout, True),
                )
            seen = []
            try:
                self.script._run_abiss_ws(
                    data,
                    binary,
                    0.95,
                    0.05,
                    size,
                    0,
                    [1] * 6,
                    fit_offset,
                    workdir=source / (width + "_python_failure"),
                    seg_dtype="uint32",
                    ws_merge_thresholds=[0.0, 0.99, 0.5],
                    on_batch_result=lambda *a: seen.append(a),
                )
            except subprocess.CalledProcessError as error:
                require(
                    error.returncode == 3 and not seen,
                    "Python failure callback contract",
                )
            else:
                raise RuntimeError("Python batch overflow did not raise")
            self.record(width + "/python_batch_failure", exit=3, callbacks=len(seen))

    def sizes(self):
        source = self.scratch / "sizes"
        source.mkdir(exist_ok=True)
        for width, high in (("ws", 2**31), ("ws64", 2**63)):
            for label, shape in [
                ("below", (high - 1, 1, 1)),
                ("at", (high, 1, 1)),
                ("above", (high + 1, 1, 1)),
                ("overflow", (2**64 - 1, 2, 2)),
            ]:
                out = self.run(
                    ROOT / "build_mem" / width, source, shape, width + label, expected=2
                )
                log = (out / "stdout.log").read_text()
                require(("Chunk size check passed:" in log) == (label == "below"), log)
                self.record(f"{width}/size/{label}", exit=2, message=log)

    def end_to_end(self):
        source = self.scratch / "e2e"
        source.mkdir(exist_ok=True)
        data = list(fixtures())[2][1]
        with h5py.File(source / "input.h5", "w") as f:
            f["main"] = data
        for batch in (False, True):
            outputs = []
            for dtype in ("stock", "uint64", "uint32", "auto"):
                name = dtype + ("_batch" if batch else "")
                cmd = [
                    sys.executable,
                    str(self.args.python_repo / "scripts/run_abiss_volume.py"),
                    "--input",
                    str(source / "input.h5"),
                    "--output",
                    str(source / f"{name}.h5"),
                    "--ws-binary",
                    str(ROOT / (STOCK_WS if dtype == "stock" else "build_mem/ws")),
                    "--ws-high-threshold",
                    "0.95",
                    "--ws-low-threshold",
                    "0.05",
                    "--ws-size-threshold",
                    "20",
                    "--ws-dust-threshold",
                    "0",
                    "--abiss-offset",
                    "17",
                    "--abiss-workdir",
                    str(source / name),
                ]
                if dtype != "stock":
                    cmd += ["--seg-dtype", dtype]
                if batch:
                    cmd += ["--ws-merge-thresholds", "0.1,0.4,0.8"]
                with (source / f"{name}.log").open("w") as log:
                    subprocess.run(
                        cmd, check=True, stdout=log, stderr=subprocess.STDOUT
                    )
                paths = (
                    [source / f"{name}_mt{i}.h5" for i in range(3)]
                    if batch
                    else [source / f"{name}.h5"]
                )
                arrays = []
                for path in paths:
                    with h5py.File(path) as f:
                        arrays.append(f["main"][:])
                expected = np.uint64 if dtype in ("stock", "uint64") else np.uint32
                require(all(a.dtype == expected for a in arrays), f"e2e dtype: {name}")
                if outputs:
                    require(
                        all(np.array_equal(a, b) for a, b in zip(outputs[0], arrays)),
                        f"e2e values: {name}",
                    )
                outputs.append(arrays)
                self.record(
                    "e2e/" + name,
                    files=len(paths),
                    dtype=str(np.dtype(expected)),
                    equal=True,
                )

    def local(self):
        require(
            digest(ROOT / STOCK_WS) == STOCK_MD5, "stock ws MD5 changed before gate"
        )
        self.flags()
        crop = np.linspace(0, 1, 70000).astype(np.float16)
        high, low = benchmark_thresholds(crop)
        require(0.19 < low < 0.21 and 0.93 < high < 0.95, "percentile regression")
        for invalid in (np.zeros(10), np.full(10, np.nan), np.full(10, np.inf)):
            try:
                benchmark_thresholds(invalid)
            except RuntimeError:
                continue
            raise RuntimeError("Invalid thresholds accepted")
        self.record("benchmark_thresholds", high=high, low=low, rejected=3)
        source, shape = self.prepare("io_failure", np.zeros((3, 4, 4, 4), np.float32))
        for width in ("ws", "ws64"):
            out = source / width
            out.mkdir(exist_ok=True)
            (out / "seg_gate.data").mkdir(exist_ok=True)
            self.run(ROOT / "build_mem" / width, source, shape, width, expected=4)
            require(
                "ws runtime error:" in (out / "stdout.log").read_text(),
                "I/O diagnostic",
            )
            require(not (out / "seg_gate.data.tmp").exists(), "Temporary output leaked")
            self.record(f"{width}/io_failure", exit=4, temporary_removed=True)
        self.determinism()
        for name, data, size, dust in fixtures():
            self.contract(name, data, size, dust)
        self.failures()
        self.sizes()
        self.end_to_end()
        with h5py.File(self.args.source) as f:
            require(
                all(a >= b for a, b in zip(f["main"].shape[1:], (64, 512, 512))),
                "real crop too small",
            )
            data = f["main"][:3, :64, :512, :512]  # Never read the full source.
        self.contract("real_64x512x512", data, 100, 0)
        require(digest(ROOT / STOCK_WS) == STOCK_MD5, "stock ws MD5 changed after gate")
        self.record("stock_md5", md5=STOCK_MD5)


def proc_status(pid):
    try:
        fields = {}
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            key, _, value = line.partition(":")
            if key in ("RssAnon", "RssFile", "VmRSS", "VmHWM"):
                fields[key] = int(value.split()[0]) * 1024
        return fields
    except (OSError, ValueError):
        return {}


def descendants(pid):
    try:
        children = Path(f"/proc/{pid}/task/{pid}/children").read_text().split()
    except OSError:
        return []
    return [int(c) for c in children] + [
        p for c in children for p in descendants(int(c))
    ]


def cgroup_path():
    """Resolve this srun step's v2 cgroup (including cgroup namespace mounts)."""
    try:
        relative = next(
            line[3:]
            for line in Path("/proc/self/cgroup").read_text().splitlines()
            if line.startswith("0::")
        )
        for line in Path("/proc/self/mountinfo").read_text().splitlines():
            before, after = line.split(" - ", 1)
            if after.split()[0] != "cgroup2":
                continue
            fields = before.split()
            mount_root, mount = fields[3], Path(fields[4])
            if relative == mount_root:
                return mount
            if mount_root == "/":
                return mount / relative.lstrip("/")
            if relative.startswith(mount_root + "/"):
                return mount / relative[len(mount_root) + 1 :]
    except (OSError, StopIteration, ValueError):
        pass
    return None


def read_number(path):
    try:
        return int(path.read_text().strip())
    except (OSError, ValueError):
        return None


def memory_stat(path):
    try:
        values = dict(line.split() for line in path.read_text().splitlines())
        return {key: int(values[key]) for key in ("file", "file_dirty")}
    except (OSError, ValueError, KeyError):
        return {}


def measure(args):
    """Called only in an individual srun step; 20 ms samples, flushed marker timestamps."""
    require(
        bool(os.environ.get("SLURM_STEP_ID")),
        "measure must run inside its own srun step",
    )
    spec = json.loads(args.spec.read_text())
    out = Path(spec["output"])
    out.mkdir(parents=True, exist_ok=True)
    group = cgroup_path()
    markers = []
    started = time.monotonic()
    command = ["/usr/bin/time", "-v", "-o", str(out / "time.txt"), *spec["command"]]
    proc = subprocess.Popen(
        command,
        cwd=spec["cwd"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    def output_reader():
        with (out / "stdout.log").open("w") as log:
            for line in proc.stdout:
                log.write(line)
                log.flush()
                if line.startswith("[mem]"):
                    # Last colon separates the label from anon_gb; labels contain colons.
                    markers.append(
                        {
                            "seconds": time.monotonic() - started,
                            "phase": line[6:].split(": anon_gb=")[0],
                            "line": line.strip(),
                        }
                    )

    thread = threading.Thread(target=output_reader)
    thread.start()
    overall = {key: 0 for key in ("RssAnon", "RssFile", "VmRSS", "VmHWM")}
    phases = {}
    sampled_current_peak = None
    next_sample = time.monotonic()
    # Stream samples so the measurement helper's memory does not grow with runtime.
    with (out / "samples.jsonl").open("w") as sample_log:
        while proc.poll() is None:
            pids = descendants(proc.pid)
            per_process = {pid: proc_status(pid) for pid in pids}
            current = read_number(group / "memory.current") if group else None
            stat = memory_stat(group / "memory.stat") if group else {}
            phase_name = markers[-1]["phase"] if markers else "before first marker"
            sample_log.write(
                json.dumps(
                    {
                        "seconds": time.monotonic() - started,
                        "phase": phase_name,
                        "processes": per_process,
                        "cgroup_current": current,
                        "cgroup_stat": stat,
                    }
                )
                + "\n"
            )
            maxima = {
                key: max((v.get(key, 0) for v in per_process.values()), default=0)
                for key in overall
            }
            for key in overall:
                overall[key] = max(overall[key], maxima[key])
            if current is not None:
                sampled_current_peak = max(sampled_current_peak or 0, current)
            if spec["variant"] != "stock":
                phase = phases.setdefault(
                    phase_name,
                    {
                        "RssAnon": 0,
                        "RssFile": 0,
                        "VmRSS": 0,
                        "cgroup_file": 0,
                        "cgroup_file_dirty": 0,
                    },
                )
                for key in ("RssAnon", "RssFile", "VmRSS"):
                    phase[key] = max(phase[key], maxima[key])
                for key in ("file", "file_dirty"):
                    phase["cgroup_" + key] = max(
                        phase["cgroup_" + key], stat.get(key, 0)
                    )
            next_sample += 0.02
            time.sleep(max(0, next_sample - time.monotonic()))
    thread.join()
    wall = time.monotonic() - started
    peak = read_number(group / "memory.peak") if group else None
    (out / "markers.json").write_text(json.dumps(markers, indent=2))
    time_text = (out / "time.txt").read_text()
    rss = re.search(r"Maximum resident set size \(kbytes\): (\d+)", time_text)
    aggregate_method = (
        "memory.peak"
        if peak is not None
        else (
            "sampled memory.current"
            if sampled_current_peak is not None
            else "unverified"
        )
    )
    if peak is None:
        peak = sampled_current_peak
    result = {
        **spec,
        "exit": proc.returncode,
        "wall_seconds": wall,
        "time_max_rss_bytes": int(rss[1]) * 1024 if rss else None,
        "sampled_process_maxima": overall,
        "phases": phases if markers else "unavailable: no C0 markers",
        "aggregate_peak_bytes": peak,
        "aggregate_method": aggregate_method,
        "cgroup": str(group),
        "step": os.environ["SLURM_STEP_ID"],
        "aggregate_scope": "whole srun step including measurement helper",
    }
    (out / "result.json").write_text(json.dumps(result, indent=2))
    require(
        proc.returncode == 0,
        f"measurement command failed ({proc.returncode}): {out / 'stdout.log'}",
    )


def benchmark(args):
    require(
        bool(os.environ.get("SLURM_JOB_ID")),
        "benchmark must run under SLURM; no login-node benchmarks",
    )
    require(digest(ROOT / STOCK_WS) == STOCK_MD5, "stock MD5 changed")
    script = driver(args.python_repo)
    scratch = args.scratch / ("benchmark_" + os.environ["SLURM_JOB_ID"])
    scratch.mkdir(parents=True, exist_ok=True)
    baseline = scratch / "run_abiss_volume_stock.py"
    baseline.write_bytes(
        subprocess.check_output(
            [
                "git",
                "-C",
                str(args.python_repo),
                "show",
                "3cbdf39c06bf42c4b7cf739e1208751403a18bc7:scripts/run_abiss_volume.py",
            ]
        )
    )
    os.environ["PYTHONPATH"] = (
        str(args.python_repo) + os.pathsep + os.environ.get("PYTHONPATH", "")
    )
    rows = []
    for shape in ((128, 1024, 1024), (128, 3072, 3072)):
        name = "x".join(map(str, shape))
        source = scratch / name
        source.mkdir(exist_ok=True)
        with h5py.File(args.source) as f:
            dataset = f["main"]
            require(
                all(a >= b for a, b in zip(dataset.shape[1:], shape)),
                f"source smaller than {shape}",
            )
            starts = [(a - b) // 2 for a, b in zip(dataset.shape[1:], shape)]
            crop = dataset[
                :3, *(slice(start, start + size) for start, size in zip(starts, shape))
            ]
        np.save(source / "input.npy", crop)
        high, low = benchmark_thresholds(crop)
        aff = script._to_abiss_affinity(crop, channels=[2, 1, 0], edge_storage="source")
        del crop
        padded = script._write_affinity_with_halo(source / "aff.raw", aff)
        del aff
        voxels = int(np.prod(padded))
        require(voxels < 2**31, "benchmark exceeds ws capacity")
        script._write_abiss_param_file(source / "param.txt", padded, [1] * 6, 0)
        for width in ("ws", "ws64"):
            for count in (1, 5):
                thresholds = [0.47] if count == 1 else [0.35, 0.41, 0.47, 0.53, 0.59]
                for kind in ("binary", "python"):
                    for variant in ("stock", "uint64", "uint32"):
                        binary = ROOT / (
                            (STOCK_WS if width == "ws" else STOCK_WS64)
                            if variant == "stock"
                            else "build_mem/" + width
                        )
                        out = source / f"{width}_{count}_{kind}_{variant}"
                        out.mkdir(exist_ok=True)
                        if kind == "binary":
                            cmd = [
                                str(binary),
                                str(source / "param.txt"),
                                str(source / "aff.raw"),
                                str(high),
                                str(low),
                                "10000000",
                                "200",
                                "bench",
                                "max",
                                *map(str, thresholds),
                            ]
                            if variant == "uint32":
                                cmd.append("--seg-dtype=uint32")
                        else:
                            runner = (
                                baseline
                                if variant == "stock"
                                else args.python_repo / "scripts/run_abiss_volume.py"
                            )
                            cmd = [
                                sys.executable,
                                str(runner),
                                "--input",
                                str(source / "input.npy"),
                                "--output",
                                str(out / "seg.npy"),
                                "--ws-binary",
                                str(binary),
                                "--ws-high-threshold",
                                str(high),
                                "--ws-low-threshold",
                                str(low),
                                "--ws-size-threshold",
                                "10000000",
                                "--ws-dust-threshold",
                                "200",
                                "--channels",
                                "2,1,0",
                                "--edge-storage",
                                "source",
                                "--abiss-workdir",
                                str(out / "ws"),
                            ]
                            if count == 5:
                                cmd += [
                                    "--ws-merge-thresholds",
                                    ",".join(map(str, thresholds)),
                                ]
                            else:
                                cmd += ["--ws-merge-threshold", str(thresholds[0])]
                            if variant != "stock":
                                cmd += ["--seg-dtype", variant]
                        spec = {
                            "command": cmd,
                            "cwd": str(out),
                            "output": str(out),
                            "shape": shape,
                            "voxels": voxels,
                            "width": width,
                            "thresholds": count,
                            "kind": kind,
                            "variant": variant,
                        }
                        spec_path = out / "spec.json"
                        spec_path.write_text(json.dumps(spec, indent=2))
                        subprocess.run(
                            [
                                "srun",
                                "--exclusive",
                                "--nodes=1",
                                "--ntasks=1",
                                "--cpus-per-task=8",
                                sys.executable,
                                str(Path(__file__).resolve()),
                                "--mode",
                                "measure",
                                "--python-repo",
                                str(args.python_repo),
                                "--spec",
                                str(spec_path),
                            ],
                            check=True,
                        )
                        rows.append(json.loads((out / "result.json").read_text()))
                        (scratch / "results.json").write_text(
                            json.dumps(rows, indent=2)
                        )
                        # Keep logs/samples but bound disk usage between large runs.
                        for path in out.glob("*.data"):
                            path.unlink()
                        for path in out.glob("*.npy"):
                            path.unlink()
                        if (out / "ws").exists():
                            shutil.rmtree(out / "ws")
        (source / "aff.raw").unlink()
        (source / "input.npy").unlink()
    table = [
        "| Crop | Binary | K | Scope | Variant | RSS B/P | Job B/P | Wall s |",
        "|---|---|---|---|---|---|---|---|",
    ]
    acceptance = []
    for row in rows:
        rss = row["time_max_rss_bytes"]
        peak = row["aggregate_peak_bytes"]
        table.append(
            f"| {row['shape']} | {row['width']} | {row['thresholds']} | "
            f"{row['kind']} | {row['variant']} | {rss/row['voxels']:.3f} | "
            f"{peak/row['voxels'] if peak is not None else 'unverified'} | "
            f"{row['wall_seconds']:.3f} |"
        )
        if row["variant"] != "stock" and row["kind"] == "binary":
            base = next(
                b
                for b in rows
                if b["variant"] == "stock"
                and all(
                    b[k] == row[k] for k in ("shape", "width", "thresholds", "kind")
                )
            )
            ratio = rss / base["time_max_rss_bytes"]
            wall_ratio = row["wall_seconds"] / base["wall_seconds"]
            target = (
                0.7
                if row["width"] == "ws64"
                and row["thresholds"] == 5
                and row["voxels"] >= 10**9
                else 1.0
            )
            acceptance.append(
                {
                    "output": row["output"],
                    "rss_ratio": ratio,
                    "rss_target": target,
                    "wall_ratio": wall_ratio,
                    "pass": ratio <= target and wall_ratio <= 1.15,
                }
            )
    (scratch / "memory_table.md").write_text("\n".join(table) + "\n")
    (scratch / "acceptance.json").write_text(json.dumps(acceptance, indent=2))
    accounting = subprocess.run(
        [
            "sacct",
            "-j",
            os.environ["SLURM_JOB_ID"],
            "--format=JobID,MaxRSS,Elapsed",
            "-P",
        ],
        capture_output=True,
        text=True,
    )
    (scratch / "sacct_per_process_only.txt").write_text(
        accounting.stdout + accounting.stderr
    )
    require(digest(ROOT / STOCK_WS) == STOCK_MD5, "stock MD5 changed")
    print(f"Results: {scratch}; sacct MaxRSS is per-process, not aggregate memory.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("local", "benchmark", "measure"), default="local"
    )
    parser.add_argument("--python-repo", type=Path, required=True)
    parser.add_argument("--scratch", type=Path, default=ROOT / "build_mem/gate_scratch")
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument(
        "--spec", type=Path, help="Measurement command JSON (internal srun mode)"
    )
    args = parser.parse_args()
    if args.mode == "local":
        Gate(args).local()
    elif args.mode == "benchmark":
        benchmark(args)
    else:
        measure(args)


if __name__ == "__main__":
    main()
