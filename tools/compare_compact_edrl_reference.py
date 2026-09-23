#!/usr/bin/env python3
"""Compare compact EDRL v2 objects and payloads with an NInfer v2 artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import struct


def read_edrl(path: Path):
    with path.open("rb") as handle:
        header = handle.read(48)
        magic, version, metadata_offset, metadata_size, table_offset, count, reserved, data_offset = struct.unpack(
            ">iiqqqiiq", header
        )
        if magic != 0x5157454E or version != 2 or reserved != 0:
            raise ValueError("not a compact EDRL v2 artifact")
        handle.seek(table_offset)
        objects = []
        for _ in range(count):
            name_size = struct.unpack(">i", handle.read(4))[0]
            name = handle.read(name_size).decode("utf-8")
            rank = struct.unpack(">i", handle.read(4))[0]
            shape = tuple(struct.unpack(">q", handle.read(8))[0] for _ in range(rank))
            dtype, fmt, layout, offset, size = struct.unpack(">iiiqq", handle.read(28))
            objects.append({"name": name, "shape": shape, "dtype": dtype, "format": fmt,
                            "layout": layout, "offset": offset, "bytes": size})
        return path, objects


def read_ninfer(path: Path):
    with path.open("rb") as handle:
        if handle.read(8) != b"NINFER\x00\x02":
            raise ValueError("not an NInfer v2 artifact")
        directory_size = struct.unpack("<Q", handle.read(8))[0]
        directory = json.loads(handle.read(directory_size))
        payload_base = ((16 + directory_size + 4095) // 4096) * 4096
    objects = [obj for obj in directory["objects"] if obj["kind"] == "tensor"]
    for obj in objects:
        obj = obj
        obj["payload_base"] = payload_base
    return path, objects


def digest(handle, offset: int, size: int) -> str:
    handle.seek(offset)
    result = hashlib.sha256()
    remaining = size
    while remaining:
        chunk = handle.read(min(16 * 1024 * 1024, remaining))
        if not chunk:
            raise ValueError("payload truncated while hashing")
        result.update(chunk)
        remaining -= len(chunk)
    return result.hexdigest()


def compare(edrl_path: Path, ninfer_path: Path) -> dict[str, object]:
    _, edrl = read_edrl(edrl_path)
    _, ninfer = read_ninfer(ninfer_path)
    edrl_by_name = {obj["name"]: obj for obj in edrl}
    ninfer_by_name = {obj["name"]: obj for obj in ninfer}
    metadata_mismatches: list[str] = []
    for name, reference in ninfer_by_name.items():
        candidate = edrl_by_name.get(name)
        if candidate is None:
            metadata_mismatches.append(f"missing {name}")
            continue
        if tuple(reference["shape"]) != candidate["shape"] or reference["format"] != format_name(candidate["format"]):
            metadata_mismatches.append(f"metadata {name}")
        if reference["layout"] != layout_name(candidate["layout"]):
            metadata_mismatches.append(f"layout {name}")
        if reference["bytes"] != candidate["bytes"]:
            metadata_mismatches.append(f"size {name}")
    metadata_mismatches.extend(f"extra {name}" for name in sorted(set(edrl_by_name) - set(ninfer_by_name)))

    payload_mismatches: list[str] = []
    with edrl_path.open("rb") as edrl_file, ninfer_path.open("rb") as ninfer_file:
        for name, reference in ninfer_by_name.items():
            candidate = edrl_by_name.get(name)
            if candidate is None or reference["bytes"] != candidate["bytes"]:
                continue
            edrl_hash = digest(edrl_file, candidate["offset"], candidate["bytes"])
            ninfer_hash = digest(ninfer_file, reference["payload_base"] + reference["offset"], reference["bytes"])
            if edrl_hash != ninfer_hash:
                payload_mismatches.append(name)
    return {
        "edrl_file_bytes": edrl_path.stat().st_size,
        "ninfer_file_bytes": ninfer_path.stat().st_size,
        "edrl_persistent_weight_bytes": sum(obj["bytes"] for obj in edrl),
        "ninfer_persistent_weight_bytes": sum(obj["bytes"] for obj in ninfer),
        "edrl_tensor_count": len(edrl),
        "ninfer_tensor_count": len(ninfer),
        "metadata_equivalent": not metadata_mismatches,
        "payload_byte_equivalent": not payload_mismatches,
        "metadata_mismatches": metadata_mismatches[:20],
        "payload_mismatches": payload_mismatches[:20],
    }


FORMAT_NAMES = {0: "BF16", 2: "FP32", 3: "I32", 5: "Q3G64_F16S", 6: "Q4G64_F16S",
                7: "Q5G64_F16S", 8: "Q6G64_F16S", 9: "W8G32_F16S"}
LAYOUT_NAMES = {0: "contiguous-le-v1", 1: "row-split-k128-v1"}


def format_name(ordinal: int) -> str:
    return FORMAT_NAMES[ordinal]


def layout_name(ordinal: int) -> str:
    return LAYOUT_NAMES[ordinal]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--edrl", type=Path, required=True)
    parser.add_argument("--ninfer", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(compare(args.edrl, args.ninfer), indent=2, sort_keys=True))
