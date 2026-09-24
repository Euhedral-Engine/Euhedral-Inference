# Euhedral-Inference

A Java-based ML/model inference engine.

## Project Structure

This is a multi-module Gradle project:

| Module | Purpose |
| ------ | ------- |
| `core` | Core inference engine: model loading, execution, and tensor handling. |
| `api`  | Spring Boot OpenAI-compatible HTTP API. Depends on `core`. |

## Requirements

- Java 25 (JDK)
- Gradle 9.6.1 (the committed wrapper handles the rest)
- Zig 0.16.0 for native builds; CUDA 13.1+ headers and libraries are resolved automatically

The optional `.mise.toml` records compatible tool versions. Install them or activate your
preferred toolchain manager before using the Gradle wrapper.

## Building

```bash
./gradlew build
```

The API boot JAR builds and packages Linux x86_64 and Windows x86_64 CUDA native products,
including NVRTC `.cu` sources. See [native/README.md](native/README.md) for the target
manifest, pinned NVIDIA development inputs, and target-specific override properties.

## Running

```bash
./gradlew :api:run
```

## Container

Build the Linux x86_64 image (JDK 25 and Zig 0.16.0 build stage; minimal Java 25 runtime):

```text
docker build -t euhedral-inference:local .
```

Mount your EDRL model and tokenizer checkpoint. They are not included in the image:

```text
docker run --rm --gpus all -p 1738:1738 \
  --mount type=bind,src=/absolute/path/model.edrl,dst=/models/model.edrl,readonly \
  --mount type=bind,src=/absolute/path/tokenizer,dst=/tokenizer,readonly \
  -e EUHEDRAL_INFERENCE_WORKER_CPUS=2-5 \
  -e EUHEDRAL_INFERENCE_MODEL_ID=qwen \
  euhedral-inference:local
```

The defaults are port 1738, `/models/model.edrl`, `/tokenizer`, and
`/opt/euhedral/lib/libeuhedral_cuda.so`. Override those paths with
`EUHEDRAL_INFERENCE_ARTIFACT_PATH`, `EUHEDRAL_INFERENCE_TOKENIZER_DIRECTORY`, and
`EUHEDRAL_INFERENCE_CUDA_LIBRARY_PATH`; set `PORT` at runtime for the server port and publish
that same container port with `-p` (for example, `-e PORT=18900 -p 18900:18900`). Docker
does not embed a static exposed-port declaration, since it would become misleading when
`PORT` changes. The health probe reads the same setting. The image carries
CUDA runtime/NVRTC userspace libraries, matching headers for NVRTC, the native library, and
installed kernel sources; it does not contain the NVIDIA kernel driver. NVIDIA Container
Toolkit and a compatible host driver are required. `/health` is checked by the image health
probe after the engine has loaded. Set worker CPUs to processor IDs available in your container.

## Testing

```bash
./gradlew test
```

## Formatting

Formatting is enforced via [Spotless](https://github.com/diffplug/spotless) with Palantir Java Format.

```bash
# Apply formatting to all sources
./gradlew spotlessApply

# Check formatting (used in CI)
./gradlew spotlessCheck
```

---

Licensed under the Apache License, Version 2.0 (see `LICENSE`).