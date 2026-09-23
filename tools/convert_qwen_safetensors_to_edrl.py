#!/usr/bin/env python3
"""Convert the supported Qwen3.5 27B BF16 checkpoint to a streaming EDRL artifact.

The converter keeps the source tensor bytes opaque. It copies the exact BF16 payloads
referenced by the safetensors index and writes only the language-model and MTP tensors
accepted by QwenWeightLoader; the source vision tower is intentionally excluded because
that loader has no vision-weight model yet.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import struct
import sys
import tempfile
from typing import Any, BinaryIO, Iterable, NoReturn

MAGIC = 0x5157454E  # QWEN
VERSION = 1
HEADER_SIZE = 48
BF16_ITEM_BYTES = 2
MAX_TENSOR_COUNT = 1_000_000

# QwenLayerType ordinal values in the Java artifact codec.
FULL_ATTENTION = 0
GATED_DELTA_NET = 1
FULL_ATTENTION_LAYERS = frozenset(range(3, 64, 4))


def fail(message: str) -> "NoReturn":
    raise ValueError(message)


def checked_add(left: int, right: int, label: str) -> int:
    result = left + right
    if result < left:
        fail(f"{label} range overflows")
    return result


def checked_product(values: Iterable[int], label: str) -> int:
    result = 1
    for value in values:
        if value < 0:
            fail(f"{label} contains a negative dimension")
        result = result * value
    return result


def utf8(value: str, field: str) -> bytes:
    if not isinstance(value, str):
        fail(f"{field} must be a string")
    encoded = value.encode("utf-8", "strict")
    if not encoded:
        fail(f"{field} must not be empty")
    if len(encoded) > (1 << 20):
        fail(f"{field} is too long")
    return encoded


def read_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        fail(f"cannot read JSON {path}: {error}")
    if not isinstance(value, dict):
        fail(f"JSON root is not an object: {path}")
    return value


def read_safetensors_header(path: Path) -> dict[str, dict[str, Any]]:
    try:
        with path.open("rb") as handle:
            prefix = handle.read(8)
            if len(prefix) != 8:
                fail(f"safetensors header is truncated: {path}")
            header_size = struct.unpack("<Q", prefix)[0]
            file_size = path.stat().st_size
            if header_size > file_size - 8:
                fail(f"safetensors header extends beyond file: {path}")
            raw = handle.read(header_size)
    except OSError as error:
        fail(f"cannot read safetensors header {path}: {error}")
    try:
        header = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        fail(f"invalid safetensors header {path}: {error}")
    if not isinstance(header, dict):
        fail(f"safetensors header is not an object: {path}")
    return {name: value for name, value in header.items() if name != "__metadata__"}


def layer_names(prefix: str, full_attention: bool) -> list[str]:
    names = [
        f"{prefix}.input_layernorm.weight",
        f"{prefix}.post_attention_layernorm.weight",
    ]
    if full_attention:
        names.extend(
            f"{prefix}.self_attn.{suffix}.weight"
            for suffix in ("q_proj", "k_proj", "v_proj", "o_proj", "q_norm", "k_norm")
        )
    else:
        names.extend(
            f"{prefix}.linear_attn.{suffix}"
            for suffix in (
                "in_proj_qkv.weight",
                "in_proj_z.weight",
                "in_proj_b.weight",
                "in_proj_a.weight",
                "conv1d.weight",
                "norm.weight",
                "dt_bias",
                "A_log",
                "out_proj.weight",
            )
        )
    names.extend(
        f"{prefix}.mlp.{suffix}.weight" for suffix in ("gate_proj", "up_proj", "down_proj")
    )
    return names


def required_names() -> list[str]:
    names = [
        "model.language_model.embed_tokens.weight",
        "model.language_model.norm.weight",
        "lm_head.weight",
    ]
    names.extend(
        name
        for layer in range(64)
        for name in layer_names(
            f"model.language_model.layers.{layer}", layer in FULL_ATTENTION_LAYERS
        )
    )
    names.extend(
        [
            "mtp.pre_fc_norm_embedding.weight",
            "mtp.pre_fc_norm_hidden.weight",
            "mtp.fc.weight",
            "mtp.norm.weight",
        ]
    )
    names.extend(layer_names("mtp.layers.0", True))
    return names


def validate_source(
    model_dir: Path,
) -> tuple[dict[str, Any], dict[str, str], dict[str, dict[str, Any]]]:
    config = read_json(model_dir / "config.json")
    text = config.get("text_config")
    if config.get("architectures") != ["Qwen3_5ForConditionalGeneration"]:
        fail("source is not the supported Qwen3_5ForConditionalGeneration checkpoint")
    if not isinstance(text, dict):
        fail("config.json is missing text_config")
    expected = {
        "hidden_size": 5120,
        "num_hidden_layers": 64,
        "num_attention_heads": 24,
        "num_key_value_heads": 4,
        "head_dim": 256,
        "intermediate_size": 17408,
        "linear_num_key_heads": 16,
        "linear_num_value_heads": 48,
        "linear_key_head_dim": 128,
        "linear_value_head_dim": 128,
        "linear_conv_kernel_dim": 4,
        "max_position_embeddings": 262144,
        "vocab_size": 248320,
        "mtp_num_hidden_layers": 1,
    }
    for key, expected_value in expected.items():
        actual = text.get(key)
        if actual != expected_value:
            fail(f"config field {key} is {actual!r}, expected {expected_value!r}")
    if text.get("layer_types") != [
        "full_attention" if index in FULL_ATTENTION_LAYERS else "linear_attention"
        for index in range(64)
    ]:
        fail("config layer_types does not match the supported Qwen3.5 27B topology")
    index = read_json(model_dir / "model.safetensors.index.json")
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        fail("model.safetensors.index.json has no weight_map")
    headers: dict[str, dict[str, Any]] = {}
    shard_names = sorted(set(weight_map.values()))
    for shard_name in shard_names:
        if not isinstance(shard_name, str):
            fail("safetensors shard name is not a string")
        shard = model_dir / shard_name
        if not shard.is_file() or shard.stat().st_size == 0:
            fail(f"indexed shard is missing or empty: {shard}")
        headers[shard_name] = read_safetensors_header(shard)
        indexed_names = {name for name, value in weight_map.items() if value == shard_name}
        header_names = set(headers[shard_name])
        if indexed_names != header_names:
            fail(
                f"index/header mismatch for {shard_name}: "
                f"index-only={len(indexed_names - header_names)}, "
                f"header-only={len(header_names - indexed_names)}"
            )
    return config, {str(name): str(shard) for name, shard in weight_map.items()}, headers


def build_descriptors(
    model_dir: Path,
    weight_map: dict[str, str],
    headers: dict[str, dict],
) -> list[dict]:
    names = required_names()
    if len(names) > MAX_TENSOR_COUNT or len(set(names)) != len(names):
        fail("required tensor inventory is invalid")
    available = set(weight_map)
    missing = sorted(set(names) - available)
    if missing:
        fail(f"source is missing {len(missing)} required tensors; first: {missing[0]}")

    descriptors: list[dict] = []
    payload_offset = 0
    for name in names:
        shard_name = weight_map[name]
        entry = headers[shard_name][name]
        if entry.get("dtype") != "BF16":
            fail(f"tensor {name} has dtype {entry.get('dtype')!r}, expected BF16")
        shape = entry.get("shape")
        offsets = entry.get("data_offsets")
        if not isinstance(shape, list) or not isinstance(offsets, list) or len(offsets) != 2:
            fail(f"tensor {name} has malformed safetensors metadata")
        if any(not isinstance(value, int) or value < 0 for value in shape):
            fail(f"tensor {name} has an invalid shape")
        if any(not isinstance(value, int) or value < 0 for value in offsets):
            fail(f"tensor {name} has invalid data offsets")
        source_bytes = offsets[1] - offsets[0]
        expected_bytes = checked_product(shape, f"tensor {name} shape") * BF16_ITEM_BYTES
        if source_bytes != expected_bytes:
            fail(
                f"tensor {name} has {source_bytes} source bytes, "
                f"expected {expected_bytes} from shape"
            )
        shard = model_dir / shard_name
        with shard.open("rb") as handle:
            prefix = handle.read(8)
        if len(prefix) != 8:
            fail(f"safetensors header is truncated: {shard}")
        header_size = 8 + struct.unpack("<Q", prefix)[0]
        source_start = checked_add(header_size, offsets[0], f"tensor {name} source offset")
        source_end = checked_add(source_start, source_bytes, f"tensor {name} source range")
        if source_end > shard.stat().st_size:
            fail(f"tensor {name} extends beyond shard {shard_name}")
        descriptors.append(
            {
                "name": name,
                "shape": [int(value) for value in shape],
                "shard": shard_name,
                "source_offset": source_start,
                "byte_size": source_bytes,
                "data_offset": payload_offset,
            }
        )
        payload_offset = checked_add(payload_offset, source_bytes, "artifact payload size")
    return descriptors


def encode_metadata(config: dict) -> bytes:
    text = config["text_config"]
    layer_types = [
        FULL_ATTENTION if index in FULL_ATTENTION_LAYERS else GATED_DELTA_NET
        for index in range(64)
    ]
    activation = utf8("silu", "hidden activation")
    fields = [
        (">i", text["vocab_size"]),
        (">i", text["hidden_size"]),
        (">i", text["num_hidden_layers"]),
        (">i", text["num_attention_heads"]),
        (">i", text["num_key_value_heads"]),
        (">i", text["head_dim"]),
        (">i", text["intermediate_size"]),
        (">i", text["linear_num_key_heads"]),
        (">i", text["linear_num_value_heads"]),
        (">i", text["linear_key_head_dim"]),
        (">i", text["linear_value_head_dim"]),
        (">i", text["linear_conv_kernel_dim"]),
        (">d", text["rms_norm_eps"]),
        (">d", text["rope_parameters"]["rope_theta"]),
        (">d", text["partial_rotary_factor"]),
        (">i", text["max_position_embeddings"]),
    ]
    result = bytearray()
    for fmt, value in fields:
        result.extend(struct.pack(fmt, value))
    result.extend(struct.pack(">i", len(activation)))
    result.extend(activation)
    result.extend(struct.pack(">i", len(layer_types)))
    for layer_type in layer_types:
        result.extend(struct.pack(">i", layer_type))
    result.extend(struct.pack(">iiii", 0, 0, 0, 0))
    result.extend(struct.pack(">BBi", 0, 1, 1))
    return bytes(result)


def encode_table(descriptors: list[dict], payload_base: int) -> bytes:
    table = bytearray()
    for descriptor in descriptors:
        name = utf8(descriptor["name"], "tensor name")
        shape = descriptor["shape"]
        table.extend(struct.pack(">i", len(name)))
        table.extend(name)
        table.extend(struct.pack(">i", len(shape)))
        for dimension in shape:
            table.extend(struct.pack(">q", dimension))
        table.extend(struct.pack(">iiqq", 0, 0, payload_base + descriptor["data_offset"], descriptor["byte_size"]))
    return bytes(table)


def copy_payload(
    output, model_dir: Path, descriptors: list[dict], weight_map: dict[str, str]
) -> None:
    handles: dict[str, BinaryIO] = {}
    try:
        for index, descriptor in enumerate(descriptors, start=1):
            shard_name = descriptor["shard"]
            handle = handles.get(shard_name)
            if handle is None:
                handle = (model_dir / shard_name).open("rb")
                handles[shard_name] = handle
            handle.seek(descriptor["source_offset"])
            remaining = descriptor["byte_size"]
            while remaining:
                chunk = handle.read(min(8 * 1024 * 1024, remaining))
                if not chunk:
                    fail(f"source ended while copying {descriptor['name']}")
                output.write(chunk)
                remaining -= len(chunk)
            if index == 1 or index == len(descriptors) or index % 32 == 0:
                print(
                    f"copied {index}/{len(descriptors)} tensors "
                    f"({descriptor['name']})",
                    flush=True,
                )
    finally:
        for handle in handles.values():
            handle.close()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def convert(model_dir: Path, output_path: Path, force: bool) -> Path:
    if output_path.exists() and not force:
        fail(f"output already exists; choose a new versioned path or pass --force: {output_path}")
    config, weight_map, headers = validate_source(model_dir)
    descriptors = build_descriptors(model_dir, weight_map, headers)
    metadata = encode_metadata(config)
    table_offset = HEADER_SIZE + len(metadata)
    table = encode_table(descriptors, table_offset + 0)  # replaced after table size is known
    tensor_data_offset = table_offset + len(table)
    table = encode_table(descriptors, tensor_data_offset)
    tensor_data_offset = table_offset + len(table)
    # Re-encode because descriptor absolute offsets depend on the final table end.
    table = encode_table(descriptors, tensor_data_offset)
    tensor_data_offset = table_offset + len(table)
    payload_bytes = sum(descriptor["byte_size"] for descriptor in descriptors)
    file_size = tensor_data_offset + payload_bytes
    header = struct.pack(
        ">iiqqqiiq",
        MAGIC,
        VERSION,
        HEADER_SIZE,
        len(metadata),
        table_offset,
        len(descriptors),
        0,
        tensor_data_offset,
    )
    if len(header) != HEADER_SIZE:
        fail("internal header size mismatch")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.", suffix=".partial", dir=output_path.parent
    )
    os.close(fd)
    temporary_path = Path(temporary_name)
    try:
        print(
            f"validated {len(descriptors)} tensors, {payload_bytes} payload bytes "
            f"({payload_bytes / (1 << 30):.2f} GiB)",
            flush=True,
        )
        with temporary_path.open("w+b") as output:
            output.truncate(file_size)
            output.seek(0)
            output.write(header)
            output.write(metadata)
            output.write(table)
            output.seek(tensor_data_offset)
            copy_payload(output, model_dir, descriptors, weight_map)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_path, output_path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise

    manifest = {
        "format": "edrl-v1",
        "source_model": str(model_dir),
        "source_architecture": config["architectures"],
        "source_index": str(model_dir / "model.safetensors.index.json"),
        "tensor_count": len(descriptors),
        "payload_bytes": payload_bytes,
        "file_bytes": file_size,
        "sha256": sha256(output_path),
        "excluded_source_tensor_count": len(weight_map) - len(descriptors),
        "excluded_namespace": "model.visual.*",
    }
    manifest_path = Path(f"{output_path}.manifest.json")
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")
    print(f"created {output_path} ({file_size} bytes)", flush=True)
    print(f"sha256 {manifest['sha256']}", flush=True)
    print(f"manifest {manifest_path}", flush=True)
    return output_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    try:
        convert(args.model, args.out, args.force)
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
