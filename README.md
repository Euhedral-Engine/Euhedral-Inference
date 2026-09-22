# Euhedral-Inference

A Java-based ML/model inference engine.

## Project Structure

This is a multi-module Gradle project:

| Module | Purpose |
| ------ | ------- |
| `core` | Core inference engine: model loading, execution, and tensor handling. |
| `api`  | Application entry point / CLI. Depends on `core`. |

## Requirements

- Java 21 (JDK)
- Gradle 9.6.1 (the committed wrapper handles the rest)

The toolchain is pinned via `.mise.toml`; with [mise](https://mise.jdx.dev) installed, the
correct Gradle and Java versions are selected automatically.

## Building

```bash
./gradlew build
```

## Running

```bash
./gradlew :api:run
```

## Testing

```bash
./gradlew test
```

## Formatting

Formatting is applied via Spotless (see `core`/`app` build files once added).

---

Licensed under the Apache License, Version 2.0 (see `LICENSE`).