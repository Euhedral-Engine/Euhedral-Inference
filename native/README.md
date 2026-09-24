# Euhedral CUDA native ABI

This directory contains the plain C ABI used by the Java Foreign Function & Memory binding.

The minimum CUDA header/runtime ABI is 13.1.x. An installed newer CUDA 13.x toolkit is accepted
when its headers and target libraries are complete. The host must also provide the CUDA driver library and NVRTC runtime; the driver must
support the target GPU. This native layer does not install or manage NVIDIA drivers.

`euhedral_cuda.c` is limited to memory allocation, memory queries, and synchronous copies.
`q3_embedding.c` owns the embedding C ABI and runtime compilation setup, while
`q3_embedding.cu` contains the actual Q3 row-split embedding kernel. The build installs the `.cu`
source at `share/euhedral_cuda/q3_embedding.cu` relative to the native library prefix. NVRTC compiles
it once per process for `compute_90`, and the CUDA driver JITs that PTX for the active device.
The operation reads model weights directly from their GPU allocation and writes BF16 hidden states;
it supports only `Q3_G64_FP16` with `ROW_SPLIT_K128_V1`. Its calling frame synchronizes before
publishing a successor.

`rms_norm_bf16.c` and `q3_linear_bf16.c` provide independent synchronous C ABI operations;
their `.cu` files contain the kernels. `cuda_kernel_loader.c` loads their separately installed
source assets and compiles them once per process. These operators accept opaque device addresses
and know nothing about Euhedral frames. The Q3 linear operation consumes the same row-split Q3
representation as embedding and writes BF16 output; RMSNorm consumes and writes BF16 rows.

`native-products.json` defines the two supported targets: `x86_64-linux-gnu` (`linux-x64`)
and `x86_64-windows-gnu` (`windows-x64`). There is no macOS CUDA product. The normal Gradle
`nativeBuild` and `:api:bootJar` paths build both products. `nativePackage` writes
`build/distributions/euhedral-cuda-native.zip` with the selected products. Zig installs each
product at `build/native/<product>/lib/<library>` and its seven runtime-compiled CUDA sources
at `build/native/<product>/share/euhedral_cuda/*.cu`. The Spring Boot JAR build produces
the native ZIP alongside the application JAR, but does not embed native files: Java FFM loads
the library from a filesystem path, with `.cu` files adjacent in the installed product layout.
The ZIP contains the native library and `.cu` sources, not NVIDIA CUDA runtime libraries.
Deploy target-matching CUDA runtime and NVRTC (including builtins) separately, or use the
container image, which installs the pinned Linux CUDA user-space runtime libraries.

For ordinary builds Gradle uses matching host toolkit metadata from `CUDA_HOME` or `CUDA_PATH`,
then the current platform's standard CUDA location (`/usr/local/cuda` on Linux, CUDA v13.1
under Program Files on Windows). The host toolkit must have headers at version 13.1 or higher,
the runtime/NVRTC link libraries, and the target driver stub/import library. When no suitable
host toolkit is present, Gradle downloads the official NVIDIA 13.1 cudart, NVRTC, CUDA CRT,
and CCCL redistributable archives for each target, checks the SHA-256 values in the manifest,
and caches them under the Gradle user home. The Windows driver import is generated with the
pinned Zig `dlltool` from an explicit driver ABI export list; it does not link the host's
Linux driver stub or NVIDIA's Windows static loader. Downloads include no NVIDIA kernel driver.

Use Java 25, Gradle 9.6.1, and Zig 0.16.0 (the versions recorded in `.mise.toml`).
Gradle invokes Zig directly with argument lists (and honors `ZIG` when set); no shell is needed
by the host build. Build and verify both products with:

```text
./gradlew nativeVerify nativePackage :api:bootJar
```

Explicit per-target overrides take precedence over discovery and download:

```text
-Peuhedral.cuda.linux-x64.include-dir=/path/to/linux/cuda/include
-Peuhedral.cuda.linux-x64.library-dir=/path/to/linux/cuda/lib
-Peuhedral.cuda.windows-x64.include-dir=C:/path/to/windows/cuda/include
-Peuhedral.cuda.windows-x64.library-dir=C:/path/to/windows/cuda/lib/x64
```

Legacy `euhedral.cuda.include-dir` and `euhedral.cuda.library-dir` apply only to the matching
host target. Overrides must be paired, originate from one toolkit root, and provide matching
CUDA driver and runtime header versions (13.1+). Windows cross
builds still require Windows CUDA import libraries, never Linux `.so` files. Set
`-Peuhedral.cuda.force-download=true` to choose the pinned 13.1 archives over an installed
toolkit; `-Peuhedral.native.products=linux-x64` restricts packaging (not `nativeBuild`) to
the Linux product, useful for the container image. `nativeBuildLinuxX64` and
`nativeBuildWindowsX64` are individual tasks; only the matching product is executed in CUDA
integration tests. The integration task adds the resolved CUDA runtime DLL/SO directory to
the test process search path; a compatible host driver is still required.

Run the isolated CUDA integration suite with the same automatic CUDA resolution. The full compact-model test also
requires the compact Qwen EDRL artifact, its BF16 reference EDRL artifact, and sufficient free device
memory to keep the whole compact model resident:

```text
./gradlew cudaIntegrationTest
```

The runtime needs a compatible NVIDIA driver installed on the host (or injected by NVIDIA
Container Toolkit), along with target-matching CUDA user-space runtime/NVRTC libraries and
the CUDA headers used by NVRTC to compile the installed `.cu` sources. Set
`EUHEDRAL_CUDA_INCLUDE_DIR` to their include directory. Driver stubs and development import
libraries are for linking only, not runtime deployment.
