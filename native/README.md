# Euhedral CUDA native ABI

This directory contains the plain C ABI used by the Java Foreign Function & Memory binding.

The required CUDA header ABI is 13.1.x, matching the NInfer 5070 Ti project's CUDA 13.1.2
toolkit. The host driver must support the target GPU; this native layer does not install or manage
NVIDIA drivers. The integration proof was run with the exact NInfer CUDA 13.1.2 runtime.

Build it by supplying the CUDA 13.1.x header and runtime-library directories explicitly:

```text
./gradlew nativeBuild \\
  -Peuhedral.cuda.include-dir=/path/to/cuda/include \\
  -Peuhedral.cuda.library-dir=/path/to/cuda/lib
```

The same build structure accepts Zig targets such as `x86_64-linux-gnu` and `x86_64-windows-gnu`.
CUDA library paths for a Windows toolkit must be supplied explicitly.

Run the Java FFM round-trip proof with the same paths:

```text
./gradlew cudaIntegrationTest \\
  -Peuhedral.cuda.include-dir=/path/to/cuda/include \\
  -Peuhedral.cuda.library-dir=/path/to/cuda/lib
```
