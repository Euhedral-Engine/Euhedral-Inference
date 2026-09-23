# Euhedral CUDA native ABI

This directory contains the plain C ABI used by the Java Foreign Function & Memory binding.

The required CUDA header/runtime ABI is 13.1.x, matching the NInfer 5070 Ti project's CUDA 13.1.2
toolkit. The host must also provide the CUDA driver library and NVRTC runtime; the driver must
support the target GPU. This native layer does not install or manage NVIDIA drivers.

`euhedral_cuda.c` is limited to memory allocation, memory queries, and synchronous copies.
`q3_embedding.c` owns the embedding C ABI and runtime compilation setup, while
`q3_embedding.cu` contains the actual Q3 row-split embedding kernel. The build installs the `.cu`
source at `share/euhedral_cuda/q3_embedding.cu` relative to the native library prefix. NVRTC compiles
it once per process for `compute_90`, and the CUDA driver JITs that PTX for the active device.
The operation reads model weights directly from their GPU allocation and writes BF16 hidden states;
it supports only `Q3_G64_FP16` with `ROW_SPLIT_K128_V1`. The pipeline synchronizes before returning.

Build it by supplying the CUDA 13.1.x header and runtime-library directories explicitly:

```text
./gradlew nativeBuild \\
  -Peuhedral.cuda.include-dir=/path/to/cuda/include \\
  -Peuhedral.cuda.library-dir=/path/to/cuda/lib
```

The same build structure accepts Zig targets such as `x86_64-linux-gnu` and `x86_64-windows-gnu`.
CUDA library paths for a Windows toolkit must be supplied explicitly; the Windows target links the
CUDA driver import library (`nvcuda`) and uses the same installed kernel-source location relative to
the native DLL.

Run the isolated CUDA integration suite with the same paths. The full compact-model test also
requires the compact Qwen EDRL artifact, its BF16 reference EDRL artifact, and sufficient free device
memory to keep the whole compact model resident:

```text
./gradlew cudaIntegrationTest \\
  -Peuhedral.cuda.include-dir=/path/to/cuda/include \\
  -Peuhedral.cuda.library-dir=/path/to/cuda/lib
```
