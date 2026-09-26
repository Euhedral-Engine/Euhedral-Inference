"""Opt-in direct-driver Q3 cluster comparison on actual compact-model packed tensors."""
import ctypes as C
from contextlib import ExitStack
import hashlib
import json
import random
import subprocess
import statistics
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'native/tests'))
from test_q3_primitives import Gpu, P, I, CUDA, _bind, _check  # noqa: E402

ARTIFACT = Path('/mnt/shared/qwen38-quant/artifacts/qwen3_5_27b_compact_q3.edrl')
OUT = Path(__file__).with_suffix('.jsonl')
SHAPES = [(1, 1), (2, 1), (4, 1), (1, 2), (1, 4), (2, 2)]
NAMES = ['text/layers/0/mlp/gate_up', 'text/layers/0/mlp/down', 'text/layers/0/gdn/output']
ROWS = 256


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def tensor_index():
    with ARTIFACT.open('rb') as f:
        magic, version, metadata_at, metadata_size, table_at, count, reserved, data_at = struct.unpack('>IIQQQIIQ', f.read(48))
        require(magic == 0x5157454e and version == 2 and reserved == 0
                and table_at == metadata_at + metadata_size, 'invalid artifact header')
        f.seek(table_at)
        table = f.read(data_at - table_at)
    i = 0
    chosen = {}
    for _ in range(count):
        n, = struct.unpack_from('>I', table, i)
        i += 4
        name = table[i:i+n].decode()
        i += n
        rank, = struct.unpack_from('>I', table, i)
        i += 4
        shape = struct.unpack_from('>' + 'Q'*rank, table, i)
        i += 8*rank
        dtype, fmt, layout = struct.unpack_from('>III', table, i)
        i += 12
        offset, size = struct.unpack_from('>QQ', table, i)
        i += 16
        if name in NAMES:
            require(rank == 2 and dtype == 0 and fmt == 5 and layout == 1, f'invalid Q3 descriptor: {name}')
            require(size == ((shape[0] * ((shape[1] + 127)//128) * 48 + 255) & ~255)
                    + shape[0] * ((shape[1] + 127)//128) * 4, f'invalid Q3 size: {name}')
            chosen[name] = (shape, offset, size)
    require(i == len(table) and set(chosen) == set(NAMES), 'tensor table or target names do not match')
    return chosen


def event_functions():
    create = _bind(CUDA, 'cuEventCreate', [C.POINTER(P), C.c_uint])
    record = _bind(CUDA, 'cuEventRecord', [P, P])
    synchronize = _bind(CUDA, 'cuEventSynchronize', [P])
    elapsed = _bind(CUDA, 'cuEventElapsedTime', [C.POINTER(C.c_float), P, P])
    destroy = _bind(CUDA, 'cuEventDestroy_v2', [P])
    return create, record, synchronize, elapsed, destroy


def measure(gpu, function, grid, args, events, repeats=9):
    create, record, synchronize, elapsed, destroy = events
    start, stop = P(), P()
    samples = []
    try:
        _check(create(C.byref(start), 0), 'event create')
        _check(create(C.byref(stop), 0), 'event create')
        for _ in range(3):
            gpu.launch(function, grid, args)
        for _ in range(repeats):
            _check(record(start, None), 'event record')
            for _ in range(8):
                gpu.launch(function, grid, args, synchronize=False)
            _check(record(stop, None), 'event record')
            _check(synchronize(stop), 'event synchronize')
            millis = C.c_float()
            _check(elapsed(C.byref(millis), start, stop), 'event elapsed')
            samples.append(millis.value * 1000 / 8)
    finally:
        if start.value:
            destroy(start)
        if stop.value:
            destroy(stop)
    return {'median_us': statistics.median(samples), 'samples_us': samples}


def resources(gpu, name):
    get = _bind(CUDA, 'cuFuncGetAttribute', [C.POINTER(I), I, P])
    fn = P()
    _check(gpu.function(C.byref(fn), gpu.module, name.encode()), 'get function')
    result = {}
    for label, code in [('registers_per_thread', 4), ('static_shared_bytes', 1), ('local_bytes_per_thread', 3)]:
        value = I()
        _check(get(C.byref(value), code, fn), 'get attribute')
        result[label] = value.value
    return result


def run():
    index = tensor_index()
    events = event_functions()
    rng = random.Random(0x51A7)
    device = subprocess.check_output(['nvidia-smi', '--query-gpu=name,driver_version', '--format=csv,noheader'], text=True).strip()
    with ExitStack() as stack:
        baseline = Gpu(b'#include "q3/kernels.cu"\n')
        stack.callback(baseline.close)
        cluster = Gpu(b'#include "q3/cluster_kernels.cu"\n', cpp_std=17)
        stack.callback(cluster.close)
        with ARTIFACT.open('rb') as artifact, OUT.open('w') as result_file:
            for tensor in NAMES:
                with ExitStack() as buffers:
                    (outputs, width), offset, size = index[tensor]
                    artifact.seek(offset)
                    packed = artifact.read(size)
                    require(len(packed) == size, f'truncated tensor: {tensor}')
                    # Small finite BF16 inputs keep the bitwise comparison deterministic.
                    x = b''.join(struct.pack('<H', rng.choice((0x3e80, 0xbf00, 0x3f80, 0x3f00))) for _ in range(ROWS * width))
                    weights = baseline.upload(packed)
                    buffers.callback(baseline.free, weights)
                    inputs = baseline.upload(x)
                    buffers.callback(baseline.free, inputs)
                    result_size = ROWS * outputs * 2
                    reference_dst = baseline.zeros(result_size, fill=0xa5)
                    buffers.callback(baseline.free, reference_dst)
                    candidate_dst = baseline.zeros(result_size, fill=0xa5)
                    buffers.callback(baseline.free, candidate_dst)
                    common = [C.c_uint64(inputs), C.c_uint64(weights)]
                    common += [I(ROWS), I(width), I(outputs), C.c_uint64((outputs * ((width+127)//128) * 48 + 255) & ~255)]
                    reference_args = common[:2] + [C.c_uint64(reference_dst)] + common[2:]
                    candidate_args = common[:2] + [C.c_uint64(candidate_dst)] + common[2:]
                    for tile_rows in (32, 64):
                        base_name = 'euhedral_q3_prefill' + ('_64' if tile_rows == 64 else '')
                        base_grid = ((ROWS + tile_rows-1)//tile_rows) * ((outputs+31)//32)
                        baseline.memset(reference_dst, 0xa5, result_size)
                        baseline.launch(base_name, base_grid, reference_args)
                        expected = baseline.download(reference_dst, result_size)
                        require(all(expected[i:i+2] != b'\xa5\xa5' for i in range(0, result_size, 2)),
                                f'baseline missing writes: {tensor} {tile_rows}')
                        for ordinal, (cm, cn) in enumerate(SHAPES):
                            name = f'euhedral_q3_cluster_{tile_rows}_{cm}x{cn}'
                            grid = (((outputs + 32*cn-1)//(32*cn))*cn,
                                    ((ROWS + tile_rows*cm-1)//(tile_rows*cm))*cm)
                            baseline.memset(candidate_dst, 0xa5, result_size)
                            cluster.launch(name, grid, candidate_args)
                            actual = baseline.download(candidate_dst, result_size)
                            require(all(actual[i:i+2] != b'\xa5\xa5' for i in range(0, result_size, 2)),
                                    f'cluster missing writes: {tensor} {tile_rows} {cm}x{cn}')
                            if expected != actual:
                                offset_at = next((i for i, (a, b) in enumerate(zip(expected, actual)) if a != b), None)
                                raise RuntimeError(f'bitwise mismatch: {tensor} {tile_rows} {cm}x{cn} at byte {offset_at}')
                            # Pair each cluster with a nearby matched-tile CTA; alternate order.
                            if ordinal % 2 == 0:
                                base_time = measure(baseline, base_name, base_grid, reference_args, events)
                                timing = measure(cluster, name, grid, candidate_args, events)
                            else:
                                timing = measure(cluster, name, grid, candidate_args, events)
                                base_time = measure(baseline, base_name, base_grid, reference_args, events)
                            entry = dict(tensor=tensor, tensor_offset=offset, tensor_size=size,
                                         tensor_sha256=hashlib.sha256(packed).hexdigest(), device=device,
                                         rows=ROWS, width=width, outputs=outputs, tile_rows=tile_rows,
                                         production_auto_tile=(64 if tensor.endswith(('/gate_up', '/down')) else 32),
                                         cluster=[cm, cn], matched_tile_cta_us=base_time['median_us'],
                                         cluster_us=timing['median_us'],
                                         matched_tile_cta_samples_us=base_time['samples_us'],
                                         cluster_samples_us=timing['samples_us'],
                                         resources=resources(cluster, name), baseline_resources=resources(baseline, base_name),
                                         bitwise_match=True, complete_output=True)
                            result_file.write(json.dumps(entry) + '\n')
                            result_file.flush()
                            print(tensor, tile_rows, (cm, cn), round(timing['median_us'], 2),
                                  round(base_time['median_us'], 2), flush=True)

if __name__ == '__main__':
    run()
