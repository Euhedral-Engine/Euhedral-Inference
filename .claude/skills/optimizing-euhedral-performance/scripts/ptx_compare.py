#!/usr/bin/env python3
"""ptx_compare.py BEFORE_SRC AFTER_SRC MODULE[=AFTER_MODULE] ...: prove a CUDA refactor is code-neutral.

Compiles each NVRTC module root (path relative to its src dir, e.g. gdn/kernels.cu; give =AFTER_MODULE
when the root moved) from both source trees with the loader's options, then compares every kernel
entry's PTX after normalizing what moves without changing code: internal-linkage prefixes NVRTC derives
from the program name, basic-block and local-depot numbering, call-sequence numbers, mangled names of
helpers whose signatures changed, and entry order. Prints removed/added/changed entries per module.
Run with LD_LIBRARY_PATH=<repo>/build/cuda-dev/linux-x64/runtime (NVRTC builtins).
"""
import ctypes as C, pathlib, re, sys

ROOT = pathlib.Path(__file__).resolve().parents[4]
INCLUDE = ROOT / 'build/cuda-dev/linux-x64/include'
nvrtc = C.CDLL(str(next((ROOT / 'build/cuda-dev/linux-x64/runtime').glob('libnvrtc.so*'))))


def ptx(src_root, module):
    path = pathlib.Path(src_root) / module
    program = C.c_void_p()
    assert nvrtc.nvrtcCreateProgram(C.byref(program), path.read_bytes(), module.encode(), 0, None, None) == 0
    options = [b'--std=c++14', b'--gpu-architecture=compute_90', b'-I' + str(INCLUDE).encode(),
               b'-I' + str(path.parent).encode(), b'-I' + str(src_root).encode()]
    if nvrtc.nvrtcCompileProgram(program, len(options), (C.c_char_p * len(options))(*options)):
        size = C.c_size_t(); nvrtc.nvrtcGetProgramLogSize(program, C.byref(size))
        log = C.create_string_buffer(size.value); nvrtc.nvrtcGetProgramLog(program, log)
        sys.exit(f'{src_root}/{module} failed to compile:\n{log.value.decode()[:3000]}')
    size = C.c_size_t(); nvrtc.nvrtcGetPTXSize(program, C.byref(size))
    out = C.create_string_buffer(size.value); nvrtc.nvrtcGetPTX(program, out)
    return out.value.decode()


def entries(text):
    text = re.sub(r'_ZZ?N\d+_INTERNAL_\d+_\d+_\w+?_cu_[0-9a-f]{8}', 'INTERNAL', text)
    text = re.sub(r'(\$L__BB|__local_depot)\d+', r'\1N', text)
    text = re.sub(r'callseq \d+', 'callseq N', text)
    text = re.sub(r'\n\t// \.globl\t\w+', '', text)
    result = {}
    for block in re.split(r'\n(?=\.visible \.entry )', text):
        match = re.match(r'\.visible \.entry (\w+)\(', block)
        if match:
            result[match.group(1)] = re.sub(r'INTERNAL\w*', 'INTERNAL', block.strip())
    return result


before_root, after_root = sys.argv[1], sys.argv[2]
neutral = True
for spec in sys.argv[3:]:
    before_module, _, after_module = spec.partition('=')
    a, b = entries(ptx(before_root, before_module)), entries(ptx(after_root, after_module or before_module))
    changed = sorted(k for k in a.keys() & b.keys() if a[k] != b[k])
    removed, added = sorted(a.keys() - b.keys()), sorted(b.keys() - a.keys())
    neutral &= not changed
    print(f'{spec}: {len(a.keys() & b.keys())} common entries, changed {changed or "none"}, '
          f'removed {removed or "none"}, added {added or "none"}')
sys.exit(0 if neutral else 1)
