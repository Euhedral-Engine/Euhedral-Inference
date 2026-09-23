#!/usr/bin/env python3
"""Create the deterministic compact Q3 EDRL inventory used by NInfer.

Quantization and fusion happen offline. The Java loader receives only final runtime
objects and never reconstructs Hugging Face tensors or performs quantization.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import mmap
import os
from pathlib import Path
import struct
import sys
import tempfile
from typing import Any, Callable, Iterable, NoReturn

import numpy as np

MAGIC = 0x5157454E
VERSION = 2
HEADER_SIZE = 48
FULL_ATTENTION_LAYERS = frozenset(range(3, 64, 4))
VOCAB_SIZE = 248320
TOKENIZER_VOCAB_SIZE = 248077
DRAFT_ROWS = 131072
HIDDEN = 5120
INTERMEDIATE = 17408
VISION_LAYERS = range(27)

DTYPE_ORDINAL = {"BF16": 0, "INT32": 6}
FORMAT_ORDINAL = {
    "BF16": 0,
    "FP32": 2,
    "I32": 3,
    "Q3G64_F16S": 5,
    "Q4G64_F16S": 6,
    "Q5G64_F16S": 7,
    "Q6G64_F16S": 8,
    "W8G32_F16S": 9,
}
LAYOUT_ORDINAL = {"contiguous-le-v1": 0, "row-split-k128-v1": 1}
QUANT = {
    "Q3G64_F16S": (3, 64, -4, 3),
    "Q4G64_F16S": (4, 64, -8, 7),
    "Q5G64_F16S": (5, 64, -16, 15),
    "Q6G64_F16S": (6, 64, -32, 31),
    "W8G32_F16S": (8, 32, -127, 127),
}


def fail(message: str) -> NoReturn:
    raise ValueError(message)


def checked_product(values: Iterable[int], label: str) -> int:
    result = 1
    for value in values:
        if type(value) is not int or value <= 0:
            fail(f"{label} contains a non-positive dimension")
        result *= value
    return result


def align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def utf8(value: str, field: str) -> bytes:
    if not isinstance(value, str) or not value:
        fail(f"{field} must be a non-empty string")
    encoded = value.encode("utf-8", "strict")
    if len(encoded) > 1 << 20:
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


def bf16_to_float32(words: np.ndarray) -> np.ndarray:
    bits = words.astype(np.uint32, copy=False) << 16
    return bits.view(np.float32)


def row_split_size(shape: tuple[int, ...], format_name: str) -> int:
    if len(shape) != 2:
        fail(f"{format_name} requires a rank-2 shape")
    n, k = shape
    bits, group_size, _, _ = QUANT[format_name]
    k_pad = align_up(k, 128)
    groups = k_pad // group_size
    base_per_group = 24 if bits == 3 else 32
    high_per_group = 0 if bits in (3, 4, 8) else 8 if bits == 5 else 16
    base = n * groups * base_per_group
    high = n * groups * high_per_group
    scale_offset = align_up(base, 256) + align_up(high, 256)
    return scale_offset + n * groups * 2


def direct_size(shape: tuple[int, ...], format_name: str) -> int:
    return checked_product(shape, "shape") * (2 if format_name == "BF16" else 4)


def payload_size(shape: tuple[int, ...], format_name: str) -> int:
    return row_split_size(shape, format_name) if format_name in QUANT else direct_size(shape, format_name)


def read_safetensors_header(path: Path) -> tuple[int, dict[str, dict[str, Any]]]:
    try:
        with path.open("rb") as handle:
            prefix = handle.read(8)
            if len(prefix) != 8:
                fail(f"safetensors header is truncated: {path}")
            header_size = struct.unpack("<Q", prefix)[0]
            raw = handle.read(header_size)
    except OSError as error:
        fail(f"cannot read safetensors header {path}: {error}")
    if len(raw) != header_size:
        fail(f"safetensors header is truncated: {path}")
    try:
        header = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        fail(f"invalid safetensors header {path}: {error}")
    if not isinstance(header, dict):
        fail(f"safetensors header is not an object: {path}")
    return 8 + header_size, {name: value for name, value in header.items() if name != "__metadata__"}


@dataclass(frozen=True)
class SourceRef:
    shard: str
    offset: int
    shape: tuple[int, ...]


class SourceStore:
    def __init__(self, model: Path):
        index = read_json(model / "model.safetensors.index.json")
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            fail("model.safetensors.index.json has no weight_map")
        self.model = model
        self.refs: dict[str, SourceRef] = {}
        self.files: dict[str, Any] = {}
        self.maps: dict[str, mmap.mmap] = {}
        self.headers: dict[str, dict[str, Any]] = {}
        for shard_name in sorted(set(weight_map.values())):
            if not isinstance(shard_name, str):
                fail("safetensors shard name is not a string")
            shard_path = model / shard_name
            header_base, header = read_safetensors_header(shard_path)
            self.headers[shard_name] = header
            file = shard_path.open("rb")
            self.files[shard_name] = file
            self.maps[shard_name] = mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_READ)
            for name, entry in header.items():
                if name not in weight_map or weight_map[name] != shard_name:
                    fail(f"source index/header mismatch for {name}")
                if not isinstance(entry, dict) or entry.get("dtype") != "BF16":
                    fail(f"source tensor {name} is not BF16")
                shape = entry.get("shape")
                offsets = entry.get("data_offsets")
                if not isinstance(shape, list) or not isinstance(offsets, list) or len(offsets) != 2:
                    fail(f"source tensor {name} has malformed metadata")
                shape_tuple = tuple(int(item) for item in shape)
                source_bytes = int(offsets[1]) - int(offsets[0])
                if source_bytes != checked_product(shape_tuple, name) * 2:
                    fail(f"source tensor {name} byte size conflicts with shape")
                self.refs[name] = SourceRef(
                    shard_name, header_base + int(offsets[0]), shape_tuple
                )
        if set(self.refs) != set(weight_map):
            fail("source index does not match safetensors headers")

    def close(self) -> None:
        for value in self.maps.values():
            value.close()
        for value in self.files.values():
            value.close()

    def __enter__(self) -> "SourceStore":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def ref(self, name: str, shape: tuple[int, ...] | None = None) -> SourceRef:
        try:
            value = self.refs[name]
        except KeyError:
            fail(f"source tensor is missing: {name}")
        if shape is not None and value.shape != shape:
            fail(f"source tensor {name} has shape {value.shape}, expected {shape}")
        return value

    def words(self, name: str, shape: tuple[int, ...] | None = None) -> np.ndarray:
        ref = self.ref(name, shape)
        count = checked_product(ref.shape, name)
        return np.frombuffer(self.maps[ref.shard], dtype="<u2", count=count, offset=ref.offset).reshape(ref.shape)

    def raw(self, name: str, shape: tuple[int, ...] | None = None) -> bytes:
        ref = self.ref(name, shape)
        return bytes(self.maps[ref.shard][ref.offset : ref.offset + checked_product(ref.shape, name) * 2])

    def float_rows(self, name: str, begin: int, end: int, shape: tuple[int, int]) -> np.ndarray:
        words = self.words(name, shape)[begin:end]
        return np.ascontiguousarray(bf16_to_float32(words), dtype=np.float32)

    def float_indices(self, name: str, indices: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
        words = self.words(name, shape)[indices]
        return np.ascontiguousarray(bf16_to_float32(words), dtype=np.float32)


@dataclass
class MatrixSource:
    shape: tuple[int, int]
    read_rows: Callable[[int, int], np.ndarray]


def source_matrix(store: SourceStore, name: str, shape: tuple[int, int]) -> MatrixSource:
    store.ref(name, shape)
    return MatrixSource(shape, lambda begin, end: store.float_rows(name, begin, end, shape))


def source_matrix_reshape(
    store: SourceStore,
    name: str,
    source_shape: tuple[int, ...],
    shape: tuple[int, int],
) -> MatrixSource:
    store.ref(name, source_shape)

    def read_rows(begin: int, end: int) -> np.ndarray:
        words = store.words(name, source_shape).reshape(shape)[begin:end]
        return np.ascontiguousarray(bf16_to_float32(words), dtype=np.float32)

    return MatrixSource(shape, read_rows)


def slice_matrix(source: MatrixSource, begin: int, end: int) -> MatrixSource:
    if begin < 0 or end <= begin or end > source.shape[0]:
        fail("invalid matrix row slice")
    return MatrixSource(
        (end - begin, source.shape[1]),
        lambda row_begin, row_end: source.read_rows(begin + row_begin, begin + row_end),
    )


def concat_matrix(*sources: MatrixSource) -> MatrixSource:
    if not sources or len({source.shape[1] for source in sources}) != 1:
        fail("matrix concatenation requires a common K dimension")
    shape = (sum(source.shape[0] for source in sources), sources[0].shape[1])

    def read_rows(begin: int, end: int) -> np.ndarray:
        parts: list[np.ndarray] = []
        cursor = 0
        for source in sources:
            source_end = cursor + source.shape[0]
            if begin < source_end and end > cursor:
                local_begin = max(begin, cursor) - cursor
                local_end = min(end, source_end) - cursor
                parts.append(source.read_rows(local_begin, local_end))
            cursor = source_end
        result = np.concatenate(parts, axis=0)
        if result.shape != (end - begin, shape[1]):
            fail("matrix concatenation produced an unexpected shape")
        return result

    return MatrixSource(shape, read_rows)


def head_part(store: SourceStore, name: str, gate: bool) -> MatrixSource:
    shape = (24 * 256, HIDDEN)
    source_shape = (24 * 512, HIDDEN)
    store.ref(name, source_shape)
    base = 256 if gate else 0

    def read_rows(begin: int, end: int) -> np.ndarray:
        logical = np.arange(begin, end, dtype=np.int64)
        heads = logical // 256
        within = logical % 256
        source_rows = heads * 512 + base + within
        return store.float_indices(name, source_rows, source_shape)

    return MatrixSource(shape, read_rows)


def gather_matrix(store: SourceStore, name: str, rows: np.ndarray) -> MatrixSource:
    source_shape = (VOCAB_SIZE, HIDDEN)
    store.ref(name, source_shape)
    rows = np.asarray(rows, dtype=np.int64)
    return MatrixSource(
        (int(rows.size), HIDDEN),
        lambda begin, end: store.float_indices(name, rows[begin:end], source_shape),
    )


def _canonical_scales(max_abs: np.ndarray, qmax: int) -> tuple[np.ndarray, np.ndarray]:
    if not np.isfinite(max_abs).all():
        fail("quantization source contains NaN or infinity")
    raw_scale = (max_abs.astype(np.float64) / float(qmax)).astype(np.float32)
    scales = raw_scale.astype(np.float16)
    underflow = (scales == 0) & (max_abs > 0)
    if underflow.any():
        scales = scales.copy()
        scales[underflow] = np.float16(2.0**-24)
    reciprocal = np.zeros(max_abs.shape, dtype=np.float32)
    positive = scales > 0
    reciprocal[positive] = (1.0 / scales[positive].astype(np.float64)).astype(np.float32)
    return scales, reciprocal


def pack_codes(codes: np.ndarray, bits: int) -> tuple[bytes, bytes]:
    rows, group_size = codes.shape
    unsigned = codes.astype(np.int16) & ((1 << bits) - 1)
    if bits == 8:
        return np.ascontiguousarray(codes.astype(np.int8).view(np.uint8)).tobytes(), b""
    if bits == 3:
        base = np.zeros((rows, group_size * 3 // 8), dtype=np.uint8)
        for index in range(group_size):
            bit = index * 3
            byte = bit // 8
            shift = bit % 8
            base[:, byte] |= ((unsigned[:, index] << shift) & 0xFF).astype(np.uint8)
            if shift > 5:
                base[:, byte + 1] |= (unsigned[:, index] >> (8 - shift)).astype(np.uint8)
        return base.tobytes(), b""
    low = unsigned & 0x0F
    base = (low[:, 0::2] | (low[:, 1::2] << 4)).astype(np.uint8)
    if bits == 4:
        return base.tobytes(), b""
    high_bits = bits - 4
    high = np.zeros((rows, group_size * high_bits // 8), dtype=np.uint8)
    upper = (unsigned >> 4) & ((1 << high_bits) - 1)
    values_per_byte = 8 // high_bits
    for index in range(group_size):
        bit = (index % values_per_byte) * high_bits
        byte = index // values_per_byte
        high[:, byte] |= ((upper[:, index] << bit) & 0xFF).astype(np.uint8)
    return base.tobytes(), high.tobytes()


def quantize_matrix(output, base_offset: int, matrix: MatrixSource, format_name: str) -> None:
    bits, group_size, qmin, qmax = QUANT[format_name]
    n, k = matrix.shape
    k_pad = align_up(k, 128)
    groups = k_pad // group_size
    base_per_group = 24 if bits == 3 else 32
    high_per_group = 0 if bits in (3, 4, 8) else 8 if bits == 5 else 16
    base_row_bytes = groups * base_per_group
    high_row_bytes = groups * high_per_group
    base_bytes = n * base_row_bytes
    high_offset = align_up(base_bytes, 256)
    scale_offset = high_offset + align_up(n * high_row_bytes, 256)
    rows_per_chunk = max(1, 32 * 1024 * 1024 // max(k, 1))
    for begin in range(0, n, rows_per_chunk):
        end = min(n, begin + rows_per_chunk)
        values = matrix.read_rows(begin, end)
        if values.shape != (end - begin, k):
            fail(f"matrix source returned {values.shape}, expected {(end - begin, k)}")
        if k_pad != k:
            padded = np.zeros((end - begin, k_pad), dtype=np.float32)
            padded[:, :k] = values
            values = padded
        grouped = values.reshape(end - begin, groups, group_size)
        max_abs = np.max(np.abs(grouped), axis=2)
        scales, reciprocal = _canonical_scales(max_abs, qmax)
        codes = np.rint(grouped * reciprocal[..., None])
        codes = np.clip(codes, qmin, qmax).astype(np.int8)
        flat_codes = codes.reshape(-1, group_size)
        base, high = pack_codes(flat_codes, bits)
        output.seek(base_offset + begin * base_row_bytes)
        output.write(base)
        if high_row_bytes:
            output.seek(base_offset + high_offset + begin * high_row_bytes)
            output.write(high)
        output.seek(base_offset + scale_offset + begin * groups * 2)
        output.write(np.ascontiguousarray(scales.astype("<f2")).tobytes())


def write_direct(output, base_offset: int, store: SourceStore, action: tuple[str, Any]) -> None:
    kind, value = action
    if kind == "raw":
        output.seek(base_offset)
        output.write(store.raw(value[0], value[1]))
        return
    if kind == "reshape":
        words = store.words(value[0], value[1]).reshape(value[2]).copy()
        output.seek(base_offset)
        output.write(words.astype("<u2", copy=False).tobytes())
        return
    if kind == "transpose":
        words = store.words(value[0], value[1]).reshape(value[2]).transpose(value[3]).copy()
        output.seek(base_offset)
        output.write(words.astype("<u2", copy=False).tobytes())
        return
    if kind == "fp32":
        values = bf16_to_float32(store.words(value[0], value[1])).astype("<f4", copy=False)
        output.seek(base_offset)
        output.write(np.ascontiguousarray(values).tobytes())
        return
    if kind == "i32":
        output.seek(base_offset)
        output.write(np.asarray(value, dtype="<i4").tobytes())
        return
    fail(f"unknown direct action {kind}")


@dataclass
class ObjectPlan:
    name: str
    shape: tuple[int, ...]
    source_dtype: str
    format_name: str
    layout: str
    byte_size: int
    writer: Callable[[Any, int], None]
    offset: int = 0


def add_direct(plans: list[ObjectPlan], name: str, shape: tuple[int, ...], store: SourceStore,
               source_name: str, source_shape: tuple[int, ...], kind: str = "raw", extra: Any = None,
               format_name: str = "BF16", source_dtype: str = "BF16") -> None:
    if format_name not in ("BF16", "FP32", "I32"):
        fail(f"direct object {name} has unsupported format {format_name}")
    if kind == "i32":
        value = extra
        writer = lambda output, offset: write_direct(output, offset, store, ("i32", value))
    else:
        action_value = (
            (source_name, source_shape, (source_shape[0], source_shape[2]), extra)
            if kind == "transpose"
            else (source_name, source_shape, extra)
        )
        writer = lambda output, offset, action=(kind, action_value): write_direct(output, offset, store, action)
    plans.append(ObjectPlan(name, shape, source_dtype, format_name, "contiguous-le-v1", direct_size(shape, format_name), writer))


def add_quant(plans: list[ObjectPlan], name: str, matrix: MatrixSource, format_name: str) -> None:
    if format_name not in QUANT:
        fail(f"unknown quantized format {format_name}")
    plans.append(ObjectPlan(
        name, matrix.shape, "BF16", format_name, "row-split-k128-v1", row_split_size(matrix.shape, format_name),
        lambda output, offset, source=matrix, fmt=format_name: quantize_matrix(output, offset, source, fmt),
    ))


def attention_parts(store: SourceStore, prefix: str) -> tuple[MatrixSource, MatrixSource]:
    q_name = prefix + "self_attn.q_proj.weight"
    return head_part(store, q_name, False), head_part(store, q_name, True)


def build_plans(store: SourceStore, selected: np.ndarray) -> list[ObjectPlan]:
    plans: list[ObjectPlan] = []
    add_quant(plans, "text/token_embedding", source_matrix(store, "model.language_model.embed_tokens.weight", (VOCAB_SIZE, HIDDEN)), "Q3G64_F16S")
    for layer in range(64):
        source_prefix = f"model.language_model.layers.{layer}."
        object_prefix = f"text/layers/{layer}/"
        add_direct(plans, object_prefix + "input_norm", (HIDDEN,), store, source_prefix + "input_layernorm.weight", (HIDDEN,))
        if layer in FULL_ATTENTION_LAYERS:
            query, gate = attention_parts(store, source_prefix)
            add_quant(plans, object_prefix + "attention/query_key", concat_matrix(query, source_matrix(store, source_prefix + "self_attn.k_proj.weight", (1024, HIDDEN))), "Q4G64_F16S")
            add_quant(plans, object_prefix + "attention/gate_value", concat_matrix(gate, source_matrix(store, source_prefix + "self_attn.v_proj.weight", (1024, HIDDEN))), "Q5G64_F16S")
            add_direct(plans, object_prefix + "attention/query_norm", (256,), store, source_prefix + "self_attn.q_norm.weight", (256,))
            add_direct(plans, object_prefix + "attention/key_norm", (256,), store, source_prefix + "self_attn.k_norm.weight", (256,))
            add_quant(plans, object_prefix + "attention/output", source_matrix(store, source_prefix + "self_attn.o_proj.weight", (HIDDEN, 6144)), "Q3G64_F16S")
        else:
            qkv = source_matrix(store, source_prefix + "linear_attn.in_proj_qkv.weight", (10240, HIDDEN))
            add_direct(plans, object_prefix + "gdn/a_log", (48,), store, source_prefix + "linear_attn.A_log", (48,), kind="fp32", format_name="FP32")
            add_direct(plans, object_prefix + "gdn/dt_bias", (48,), store, source_prefix + "linear_attn.dt_bias", (48,), kind="fp32", format_name="FP32")
            add_direct(plans, object_prefix + "gdn/convolution", (4, 10240), store, source_prefix + "linear_attn.conv1d.weight", (10240, 1, 4), kind="transpose", extra=(1, 0))
            add_direct(plans, object_prefix + "gdn/a_projection", (48, HIDDEN), store, source_prefix + "linear_attn.in_proj_a.weight", (48, HIDDEN))
            add_direct(plans, object_prefix + "gdn/b_projection", (48, HIDDEN), store, source_prefix + "linear_attn.in_proj_b.weight", (48, HIDDEN))
            add_quant(plans, object_prefix + "gdn/query_key", slice_matrix(qkv, 0, 4096), "Q4G64_F16S")
            add_quant(plans, object_prefix + "gdn/value_z", concat_matrix(slice_matrix(qkv, 4096, 10240), source_matrix(store, source_prefix + "linear_attn.in_proj_z.weight", (6144, HIDDEN))), "Q5G64_F16S")
            add_direct(plans, object_prefix + "gdn/norm", (128,), store, source_prefix + "linear_attn.norm.weight", (128,))
            add_quant(plans, object_prefix + "gdn/output", source_matrix(store, source_prefix + "linear_attn.out_proj.weight", (HIDDEN, 6144)), "Q3G64_F16S")
        add_direct(plans, object_prefix + "post_attention_norm", (HIDDEN,), store, source_prefix + "post_attention_layernorm.weight", (HIDDEN,))
        add_quant(plans, object_prefix + "mlp/gate_up", concat_matrix(
            source_matrix(store, source_prefix + "mlp.gate_proj.weight", (INTERMEDIATE, HIDDEN)),
            source_matrix(store, source_prefix + "mlp.up_proj.weight", (INTERMEDIATE, HIDDEN))), "Q3G64_F16S")
        add_quant(plans, object_prefix + "mlp/down", source_matrix(store, source_prefix + "mlp.down_proj.weight", (HIDDEN, INTERMEDIATE)), "Q3G64_F16S")
    add_direct(plans, "text/final_norm", (HIDDEN,), store, "model.language_model.norm.weight", (HIDDEN,))
    add_quant(plans, "text/output_head", source_matrix(store, "lm_head.weight", (VOCAB_SIZE, HIDDEN)), "Q3G64_F16S")
    add_quant(plans, "text/draft_head", gather_matrix(store, "lm_head.weight", selected), "Q3G64_F16S")
    add_direct(plans, "text/draft_head_token_ids", (DRAFT_ROWS,), store, "", (), kind="i32", extra=selected, format_name="I32", source_dtype="INT32")

    mtp = "mtp.layers.0."
    add_quant(plans, "mtp/input_projection", source_matrix(store, "mtp.fc.weight", (HIDDEN, 10240)), "Q3G64_F16S")
    add_direct(plans, "mtp/embedding_norm", (HIDDEN,), store, "mtp.pre_fc_norm_embedding.weight", (HIDDEN,))
    add_direct(plans, "mtp/hidden_norm", (HIDDEN,), store, "mtp.pre_fc_norm_hidden.weight", (HIDDEN,))
    query, gate = attention_parts(store, mtp)
    add_direct(plans, "mtp/layer/input_norm", (HIDDEN,), store, mtp + "input_layernorm.weight", (HIDDEN,))
    add_quant(plans, "mtp/layer/attention/query_key_gate_value", concat_matrix(
        query,
        source_matrix(store, mtp + "self_attn.k_proj.weight", (1024, HIDDEN)),
        gate,
        source_matrix(store, mtp + "self_attn.v_proj.weight", (1024, HIDDEN))), "W8G32_F16S")
    add_direct(plans, "mtp/layer/attention/query_norm", (256,), store, mtp + "self_attn.q_norm.weight", (256,))
    add_direct(plans, "mtp/layer/attention/key_norm", (256,), store, mtp + "self_attn.k_norm.weight", (256,))
    add_quant(plans, "mtp/layer/attention/output", source_matrix(store, mtp + "self_attn.o_proj.weight", (HIDDEN, 6144)), "Q3G64_F16S")
    add_direct(plans, "mtp/layer/post_attention_norm", (HIDDEN,), store, mtp + "post_attention_layernorm.weight", (HIDDEN,))
    add_quant(plans, "mtp/layer/mlp/gate_up", concat_matrix(
        source_matrix(store, mtp + "mlp.gate_proj.weight", (INTERMEDIATE, HIDDEN)),
        source_matrix(store, mtp + "mlp.up_proj.weight", (INTERMEDIATE, HIDDEN))), "Q3G64_F16S")
    add_quant(plans, "mtp/layer/mlp/down", source_matrix(store, mtp + "mlp.down_proj.weight", (HIDDEN, INTERMEDIATE)), "Q3G64_F16S")
    add_direct(plans, "mtp/final_norm", (HIDDEN,), store, "mtp.norm.weight", (HIDDEN,))

    add_quant(plans, "vision/patch_embedding", source_matrix_reshape(
        store, "model.visual.patch_embed.proj.weight", (1152, 3, 2, 16, 16), (1152, 1536)), "Q6G64_F16S")
    add_direct(plans, "vision/patch_embedding_bias", (1152,), store, "model.visual.patch_embed.proj.bias", (1152,))
    add_direct(plans, "vision/position_embedding", (2304, 1152), store, "model.visual.pos_embed.weight", (2304, 1152))
    for layer in VISION_LAYERS:
        source_prefix = f"model.visual.blocks.{layer}."
        object_prefix = f"vision/layers/{layer}/"
        add_quant(plans, object_prefix + "attention/qkv", source_matrix(store, source_prefix + "attn.qkv.weight", (3456, 1152)), "Q4G64_F16S")
        add_direct(plans, object_prefix + "attention/qkv_bias", (3456,), store, source_prefix + "attn.qkv.bias", (3456,))
        add_quant(plans, object_prefix + "attention/output", source_matrix(store, source_prefix + "attn.proj.weight", (1152, 1152)), "Q5G64_F16S")
        add_direct(plans, object_prefix + "attention/output_bias", (1152,), store, source_prefix + "attn.proj.bias", (1152,))
        add_quant(plans, object_prefix + "mlp/fc1", source_matrix(store, source_prefix + "mlp.linear_fc1.weight", (4304, 1152)), "Q4G64_F16S")
        add_direct(plans, object_prefix + "mlp/fc1_bias", (4304,), store, source_prefix + "mlp.linear_fc1.bias", (4304,))
        add_quant(plans, object_prefix + "mlp/fc2", source_matrix(store, source_prefix + "mlp.linear_fc2.weight", (1152, 4304)), "Q5G64_F16S")
        add_direct(plans, object_prefix + "mlp/fc2_bias", (1152,), store, source_prefix + "mlp.linear_fc2.bias", (1152,))
        add_direct(plans, object_prefix + "norm1/weight", (1152,), store, source_prefix + "norm1.weight", (1152,))
        add_direct(plans, object_prefix + "norm1/bias", (1152,), store, source_prefix + "norm1.bias", (1152,))
        add_direct(plans, object_prefix + "norm2/weight", (1152,), store, source_prefix + "norm2.weight", (1152,))
        add_direct(plans, object_prefix + "norm2/bias", (1152,), store, source_prefix + "norm2.bias", (1152,))
    add_quant(plans, "vision/merger/fc1", source_matrix(store, "model.visual.merger.linear_fc1.weight", (4608, 4608)), "W8G32_F16S")
    add_direct(plans, "vision/merger/fc1_bias", (4608,), store, "model.visual.merger.linear_fc1.bias", (4608,))
    add_quant(plans, "vision/merger/fc2", source_matrix(store, "model.visual.merger.linear_fc2.weight", (HIDDEN, 4608)), "W8G32_F16S")
    add_direct(plans, "vision/merger/fc2_bias", (HIDDEN,), store, "model.visual.merger.linear_fc2.bias", (HIDDEN,))
    add_direct(plans, "vision/merger/norm/weight", (1152,), store, "model.visual.merger.norm.weight", (1152,))
    add_direct(plans, "vision/merger/norm/bias", (1152,), store, "model.visual.merger.norm.bias", (1152,))
    if len(plans) != 1118:
        fail(f"compact inventory produced {len(plans)} objects, expected 1118")
    return plans


def shortlist(model: Path, ranking_path: Path) -> np.ndarray:
    counts = np.fromfile(ranking_path, dtype="<i8", count=VOCAB_SIZE)
    if counts.size != VOCAB_SIZE:
        fail("ranking file does not contain the complete vocabulary row")
    tokenizer = read_json(model / "tokenizer_config.json")
    forced = sorted(
        int(token_id)
        for token_id, value in tokenizer.get("added_tokens_decoder", {}).items()
        if isinstance(value, dict) and value.get("special", False) and 0 <= int(token_id) < TOKENIZER_VOCAB_SIZE
    )
    order = np.argsort(-counts[:TOKENIZER_VOCAB_SIZE], kind="stable")
    forced_set = set(forced)
    wanted = DRAFT_ROWS - len(forced)
    picked: list[int] = []
    for token_id in order.tolist():
        if token_id not in forced_set:
            picked.append(token_id)
            if len(picked) == wanted:
                break
    selected = np.asarray(picked + forced, dtype=np.int64)
    selected = selected[np.argsort(-counts[selected], kind="stable")]
    if selected.size != DRAFT_ROWS or np.unique(selected).size != DRAFT_ROWS:
        fail("draft shortlist is not exactly 131072 unique rows")
    return selected


def encode_metadata(config: dict[str, Any]) -> bytes:
    text = config["text_config"]
    layer_types = [0 if i in FULL_ATTENTION_LAYERS else 1 for i in range(64)]
    result = bytearray()
    values = [
        (">i", text["vocab_size"]), (">i", text["hidden_size"]), (">i", text["num_hidden_layers"]),
        (">i", text["num_attention_heads"]), (">i", text["num_key_value_heads"]), (">i", text["head_dim"]),
        (">i", text["intermediate_size"]), (">i", text["linear_num_key_heads"]), (">i", text["linear_num_value_heads"]),
        (">i", text["linear_key_head_dim"]), (">i", text["linear_value_head_dim"]), (">i", text["linear_conv_kernel_dim"]),
        (">d", text["rms_norm_eps"]), (">d", text["rope_parameters"]["rope_theta"]),
        (">d", text["partial_rotary_factor"]), (">i", text["max_position_embeddings"]),
    ]
    for fmt, value in values:
        result.extend(struct.pack(fmt, value))
    activation = utf8(text["hidden_act"], "hidden activation")
    result.extend(struct.pack(">i", len(activation)))
    result.extend(activation)
    result.extend(struct.pack(">i", len(layer_types)))
    for layer_type in layer_types:
        result.extend(struct.pack(">i", layer_type))
    result.extend(struct.pack(">iiii", 0, 0, 0, 0))
    result.extend(struct.pack(">BBi", 0, 1, 1))
    return bytes(result)


def encode_table(plans: list[ObjectPlan]) -> bytes:
    result = bytearray()
    for plan in plans:
        name = utf8(plan.name, "tensor name")
        result.extend(struct.pack(">i", len(name)))
        result.extend(name)
        result.extend(struct.pack(">i", len(plan.shape)))
        for dimension in plan.shape:
            result.extend(struct.pack(">q", dimension))
        result.extend(struct.pack(">iiiqq", DTYPE_ORDINAL[plan.source_dtype], FORMAT_ORDINAL[plan.format_name], LAYOUT_ORDINAL[plan.layout], plan.offset, plan.byte_size))
    return bytes(result)


def assign_offsets(plans: list[ObjectPlan], data_base: int) -> int:
    cursor = data_base
    for plan in plans:
        plan.offset = align_up(cursor, 256)
        cursor = plan.offset + plan.byte_size
    return cursor


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def compare_reference(path: Path, plans: list[ObjectPlan]) -> dict[str, Any]:
    with path.open("rb") as handle:
        if handle.read(8) != b"NINFER\x00\x02":
            fail("reference is not an NInfer v2 artifact")
        directory_size = struct.unpack("<Q", handle.read(8))[0]
        directory = json.loads(handle.read(directory_size))
    reference = [obj for obj in directory["objects"] if obj["kind"] == "tensor"]
    actual = {plan.name: plan for plan in plans}
    mismatches: list[str] = []
    for obj in reference:
        plan = actual.get(obj["name"])
        if plan is None:
            mismatches.append(f"missing object {obj['name']}")
            continue
        if tuple(obj["shape"]) != plan.shape or obj["format"] != plan.format_name or obj["layout"] != plan.layout or obj["bytes"] != plan.byte_size:
            mismatches.append(f"metadata mismatch {obj['name']}")
    extra = sorted(set(actual) - {obj["name"] for obj in reference})
    mismatches.extend(f"extra object {name}" for name in extra)
    return {
        "reference_tensor_count": len(reference),
        "edrl_tensor_count": len(plans),
        "reference_weight_bytes": sum(obj["bytes"] for obj in reference),
        "edrl_weight_bytes": sum(plan.byte_size for plan in plans),
        "semantic_inventory_match": not mismatches and len(reference) == len(plans),
        "mismatches": mismatches[:20],
    }


def convert(model: Path, output_path: Path, ranking_path: Path, reference: Path | None, force: bool) -> None:
    if output_path.exists() and not force:
        fail(f"output already exists; pass --force: {output_path}")
    config = read_json(model / "config.json")
    text = config.get("text_config")
    if config.get("architectures") != ["Qwen3_5ForConditionalGeneration"] or not isinstance(text, dict):
        fail("source is not the supported Qwen3_5ForConditionalGeneration checkpoint")
    if text.get("num_hidden_layers") != 64 or text.get("vocab_size") != VOCAB_SIZE:
        fail("source topology does not match the registered compact Q3 profile")
    selected = shortlist(model, ranking_path)
    with SourceStore(model) as store:
        plans = build_plans(store, selected)
        metadata = encode_metadata(config)
        table_size = len(encode_table(plans))
        data_base = HEADER_SIZE + len(metadata) + table_size
        file_size = assign_offsets(plans, data_base)
        table = encode_table(plans)
        header = struct.pack(">iiqqqiiq", MAGIC, VERSION, HEADER_SIZE, len(metadata), HEADER_SIZE + len(metadata), len(plans), 0, data_base)
        if len(header) != HEADER_SIZE:
            fail("internal EDRL header size mismatch")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary_name = tempfile.mkstemp(prefix=f".{output_path.name}.", suffix=".partial", dir=output_path.parent)
        os.close(fd)
        temporary = Path(temporary_name)
        try:
            with temporary.open("w+b") as output:
                output.truncate(file_size)
                output.seek(0)
                output.write(header)
                output.write(metadata)
                output.write(table)
                for index, plan in enumerate(plans, start=1):
                    plan.writer(output, plan.offset)
                    if index == 1 or index == len(plans) or index % 32 == 0:
                        print(f"converted {index}/{len(plans)} {plan.name}", flush=True)
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary, output_path)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    formats: dict[str, int] = {}
    layouts: dict[str, int] = {}
    for plan in plans:
        formats[plan.format_name] = formats.get(plan.format_name, 0) + 1
        layouts[plan.layout] = layouts.get(plan.layout, 0) + 1
    manifest: dict[str, Any] = {
        "format": "edrl-v2-compact-q3",
        "source_model": str(model),
        "object_count": len(plans),
        "payload_bytes": sum(plan.byte_size for plan in plans),
        "file_bytes": file_size,
        "sha256": sha256(output_path),
        "format_counts": formats,
        "layout_counts": layouts,
        "fused_object_count": 259,
    }
    if reference is not None:
        manifest["reference_comparison"] = compare_reference(reference, plans)
    with Path(f"{output_path}.manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--ranking", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    try:
        convert(args.model, args.out, args.ranking, args.reference, args.force)
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
