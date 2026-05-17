#!/usr/bin/env python3
"""Patch Gemma 4 E2B LiteRT-LM audio sections with a PEFT LoRA adapter.

This is an engineering spike, not a polished converter.  It intentionally works
at the byte level because the public Gemma 4 audio exporter is not available.

Supported operations:

- inspect: summarize LoRA/TFLite mapping.
- patch-adapter: patch the float32 `tf_lite_audio_adapter` projection.
- roundtrip-encoder: verify int2/int4 unpack-pack logic without applying LoRA.
- patch-encoder: patch packed low-bit `tf_lite_audio_encoder_hw` tensors.
- patch-litertlm: replace same-size TFLite sections in an existing `.litertlm`.

The script auto-adds local downloaded schema wheels from `.tmp/` when present.
Only NumPy is required for the numeric path.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import struct
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parent.parent


def _add_local_schema_paths() -> None:
    """Make locally downloaded wheels importable without installing them."""
    for rel in (
        ".tmp/tflite_wheel",
        ".tmp/litert_lm_builder_wheel",
        ".tmp/deps",
    ):
        p = ROOT / rel
        if p.exists():
            sys.path.insert(0, str(p))


_add_local_schema_paths()

try:
    import numpy as np
except ModuleNotFoundError as exc:  # pragma: no cover - diagnostic path.
    raise SystemExit(
        "NumPy is required. Use the bundled Codex Python or the WSL conversion "
        "env, then rerun this script."
    ) from exc


try:
    import tflite
except ModuleNotFoundError as exc:  # pragma: no cover - diagnostic path.
    raise SystemExit(
        "The `tflite` schema package is required. Expected local path: "
        f"{ROOT / '.tmp' / 'tflite_wheel'}"
    ) from exc


try:
    from litert_lm_builder.schema.core import (
        litertlm_header_schema_py_generated as litertlm_schema,
    )
except ModuleNotFoundError:
    litertlm_schema = None


DEFAULT_LORA = ROOT / "outputs/gemma4_audio_v2/lora/adapter_model.safetensors"
DEFAULT_ADAPTER = ROOT / ".tmp/litertlm_sections/audio_adapter.tflite"
DEFAULT_ENCODER = ROOT / ".tmp/litertlm_sections/audio_encoder_hw.tflite"
DEFAULT_LITERTLM = ROOT / "models/gemma-4-E2B-it.gallery-7fa1d78.litertlm"


AUDIO_MODULE_MAP = {
    "ff1_l1": (
        "feed_forward1.ffw_layer_1.linear",
        "fflayer_start_ffn_layer1",
    ),
    "ff1_l2": (
        "feed_forward1.ffw_layer_2.linear",
        "fflayer_start_ffn_layer2",
    ),
    "q_proj": ("self_attn.q_proj.linear", "q_einsum"),
    "k_proj": ("self_attn.k_proj.linear", "k_einsum"),
    "v_proj": ("self_attn.v_proj.linear", "v_einsum"),
    "post": ("self_attn.post.linear", "attn_vec_einsum"),
    "lconv_start": ("lconv1d.linear_start.linear", "lconv_linear_start"),
    "lconv_end": ("lconv1d.linear_end.linear", "lconv_linear_end"),
    "ff2_l1": (
        "feed_forward2.ffw_layer_1.linear",
        "fflayer_end_ffn_layer1",
    ),
    "ff2_l2": (
        "feed_forward2.ffw_layer_2.linear",
        "fflayer_end_ffn_layer2",
    ),
}


TENSOR_TYPE_INT4 = 17
TENSOR_TYPE_INT2 = 19
TENSOR_TYPE_FLOAT32 = 0


@dataclass(frozen=True)
class SafeTensorInfo:
    dtype: str
    shape: tuple[int, ...]
    begin: int
    end: int


class SafeTensorReader:
    """Tiny safetensors reader for this project.

    The project LoRA is all F32. The merged HF checkpoint contains BF16, so BF16
    is supported for inspection/orientation checks.
    """

    def __init__(self, path: Path):
        self.path = path
        with path.open("rb") as f:
            header_len = struct.unpack("<Q", f.read(8))[0]
            self._data_begin = 8 + header_len
            raw_header = f.read(header_len)
        self.header = json.loads(raw_header)
        self.tensors: dict[str, SafeTensorInfo] = {}
        for key, value in self.header.items():
            if key == "__metadata__":
                continue
            begin, end = value["data_offsets"]
            self.tensors[key] = SafeTensorInfo(
                dtype=value["dtype"],
                shape=tuple(int(x) for x in value["shape"]),
                begin=int(begin),
                end=int(end),
            )

    def keys(self) -> list[str]:
        return sorted(self.tensors)

    def read(self, key: str) -> np.ndarray:
        info = self.tensors[key]
        with self.path.open("rb") as f:
            f.seek(self._data_begin + info.begin)
            data = f.read(info.end - info.begin)
        if info.dtype == "F32":
            arr = np.frombuffer(data, dtype="<f4")
        elif info.dtype == "F16":
            arr = np.frombuffer(data, dtype="<f2").astype(np.float32)
        elif info.dtype == "BF16":
            raw = np.frombuffer(data, dtype="<u2").astype(np.uint32)
            arr = (raw << 16).view("<f4")
        else:
            raise ValueError(f"Unsupported safetensors dtype {info.dtype}: {key}")
        return arr.copy().reshape(info.shape)


def lora_key(module_path: str, side: str) -> str:
    return f"base_model.model.model.{module_path}.lora_{side}.weight"


def lora_delta(reader: SafeTensorReader, module_path: str, scale: float) -> np.ndarray:
    a = reader.read(lora_key(module_path, "A"))
    b = reader.read(lora_key(module_path, "B"))
    # PEFT Linear LoRA convention: delta_weight = B @ A * alpha/rank.
    return (b @ a).astype(np.float32) * np.float32(scale)


def read_lora_scale(lora_dir_or_file: Path) -> float:
    if lora_dir_or_file.is_dir():
        cfg_path = lora_dir_or_file / "adapter_config.json"
    else:
        cfg_path = lora_dir_or_file.parent / "adapter_config.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    return float(cfg["lora_alpha"]) / float(cfg["r"])


def resolve_lora_file(path: Path) -> Path:
    if path.is_dir():
        return path / "adapter_model.safetensors"
    return path


@dataclass
class TensorRecord:
    index: int
    name: str
    type_code: int
    shape: tuple[int, ...]
    buffer_index: int
    data_pos: int
    data_len: int
    scale_pos: int | None
    scale_len: int
    zero_point_pos: int | None
    zero_point_len: int
    quantized_dimension: int | None

    @property
    def bits(self) -> int | None:
        if self.type_code == TENSOR_TYPE_INT2:
            return 2
        if self.type_code == TENSOR_TYPE_INT4:
            return 4
        return None


def _vector_pos(table, slot: int) -> int | None:
    off = table._tab.Offset(slot)
    if not off:
        return None
    return int(table._tab.Vector(off))


def tflite_tensor_record(model, subgraph, tensor_index: int) -> TensorRecord:
    tensor = subgraph.Tensors(tensor_index)
    buffer_index = int(tensor.Buffer())
    buf = model.Buffers(buffer_index)
    data_pos = _vector_pos(buf, 4)
    data_len = int(buf.DataLength())
    q = tensor.Quantization()
    scale_pos = None
    zero_pos = None
    scale_len = 0
    zero_len = 0
    qdim = None
    if q:
        scale_pos = _vector_pos(q, 8)
        zero_pos = _vector_pos(q, 10)
        scale_len = int(q.ScaleLength())
        zero_len = int(q.ZeroPointLength())
        qdim = int(q.QuantizedDimension())
    return TensorRecord(
        index=tensor_index,
        name=(tensor.Name() or b"").decode("utf-8", "ignore"),
        type_code=int(tensor.Type()),
        shape=tuple(int(tensor.Shape(i)) for i in range(tensor.ShapeLength())),
        buffer_index=buffer_index,
        data_pos=int(data_pos or 0),
        data_len=data_len,
        scale_pos=scale_pos,
        scale_len=scale_len,
        zero_point_pos=zero_pos,
        zero_point_len=zero_len,
        quantized_dimension=qdim,
    )


def load_tflite(path: Path):
    data = bytearray(path.read_bytes())
    model = tflite.Model.GetRootAsModel(data, 0)
    subgraph = model.Subgraphs(0)
    return data, model, subgraph


def iter_const_tensors(model, subgraph) -> Iterable[TensorRecord]:
    for idx in range(subgraph.TensorsLength()):
        rec = tflite_tensor_record(model, subgraph, idx)
        if rec.data_len > 0:
            yield rec


def find_audio_adapter_tensor(model, subgraph) -> TensorRecord:
    matches = [
        rec
        for rec in iter_const_tensors(model, subgraph)
        if "AudioAdapter/audio_input_projection" in rec.name
        and rec.type_code == TENSOR_TYPE_FLOAT32
        and rec.shape == (1536, 1536)
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one audio adapter tensor, found {len(matches)}")
    return matches[0]


def find_encoder_tensor(model, subgraph, layer: int, module: str) -> TensorRecord:
    _, pattern = AUDIO_MODULE_MAP[module]
    layer_pat = f"/layer_{layer}.block_with_rope/"
    matches = []
    for rec in iter_const_tensors(model, subgraph):
        if rec.type_code not in (TENSOR_TYPE_INT2, TENSOR_TYPE_INT4):
            continue
        if layer_pat not in rec.name:
            continue
        if pattern not in rec.name:
            continue
        if "FqEinsum_0/dot_general" not in rec.name:
            continue
        matches.append(rec)
    if len(matches) != 1:
        names = "\n  ".join(m.name for m in matches[:8])
        raise ValueError(
            f"Expected one encoder tensor for layer={layer} module={module}, "
            f"found {len(matches)}:\n  {names}"
        )
    return matches[0]


def encoder_targets(layers: list[int], modules: list[str]) -> Iterable[tuple[int, str]]:
    for layer in layers:
        for module in modules:
            yield layer, module


def parse_layers(value: str) -> list[int]:
    if value == "all":
        return list(range(12))
    out: list[int] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    if not out or min(out) < 0 or max(out) > 11:
        raise argparse.ArgumentTypeError("layers must be 0..11, e.g. all, 0, 0-2")
    return sorted(set(out))


def parse_modules(value: str) -> list[str]:
    if value == "all":
        return list(AUDIO_MODULE_MAP)
    out = [x.strip() for x in value.split(",") if x.strip()]
    bad = [x for x in out if x not in AUDIO_MODULE_MAP]
    if bad:
        raise argparse.ArgumentTypeError(
            f"unknown module(s) {bad}; choices: {', '.join(AUDIO_MODULE_MAP)}"
        )
    return out


def read_float32_matrix(data: bytearray, rec: TensorRecord) -> np.ndarray:
    if rec.type_code != TENSOR_TYPE_FLOAT32:
        raise ValueError(f"Tensor is not FLOAT32: {rec.name}")
    count = math.prod(rec.shape)
    return np.frombuffer(data, dtype="<f4", count=count, offset=rec.data_pos).copy().reshape(rec.shape)


def write_float32_matrix(data: bytearray, rec: TensorRecord, value: np.ndarray) -> None:
    if value.shape != rec.shape:
        raise ValueError(f"shape mismatch: tensor {rec.shape}, value {value.shape}")
    raw = np.asarray(value, dtype="<f4").tobytes(order="C")
    if len(raw) != rec.data_len:
        raise ValueError(f"size mismatch: {len(raw)} != {rec.data_len}")
    data[rec.data_pos : rec.data_pos + rec.data_len] = raw


def read_scales(data: bytearray, rec: TensorRecord) -> np.ndarray:
    if rec.scale_pos is None or rec.scale_len <= 0:
        raise ValueError(f"Tensor has no scale vector: {rec.name}")
    return np.frombuffer(data, dtype="<f4", count=rec.scale_len, offset=rec.scale_pos).copy()


def read_zero_points(data: bytearray, rec: TensorRecord) -> np.ndarray:
    if rec.zero_point_pos is None or rec.zero_point_len <= 0:
        return np.zeros((rec.scale_len,), dtype=np.int64)
    # TFLite QuantizationParameters.zero_point is a vector<long>.
    return np.frombuffer(
        data, dtype="<i8", count=rec.zero_point_len, offset=rec.zero_point_pos
    ).copy()


def write_scales(data: bytearray, rec: TensorRecord, scales: np.ndarray) -> None:
    if rec.scale_pos is None:
        raise ValueError(f"Tensor has no scale vector: {rec.name}")
    raw = np.asarray(scales, dtype="<f4").tobytes(order="C")
    expected = rec.scale_len * 4
    if len(raw) != expected:
        raise ValueError(f"scale size mismatch: {len(raw)} != {expected}")
    data[rec.scale_pos : rec.scale_pos + expected] = raw


def unpack_lowbit(raw: bytes | bytearray | memoryview, bits: int, shape: tuple[int, ...]) -> np.ndarray:
    packed = np.frombuffer(raw, dtype=np.uint8)
    per_byte = 8 // bits
    mask = (1 << bits) - 1
    values = np.empty((packed.size, per_byte), dtype=np.uint8)
    for i in range(per_byte):
        values[:, i] = (packed >> (i * bits)) & mask
    flat = values.reshape(-1)[: math.prod(shape)].astype(np.int16)
    sign_cut = 1 << (bits - 1)
    flat[flat >= sign_cut] -= 1 << bits
    return flat.astype(np.int8).reshape(shape)


def pack_lowbit(q: np.ndarray, bits: int) -> bytes:
    per_byte = 8 // bits
    mask = (1 << bits) - 1
    flat = np.asarray(q, dtype=np.int16).reshape(-1)
    if flat.size % per_byte:
        raise ValueError(f"flat tensor size {flat.size} is not divisible by {per_byte}")
    unsigned = (flat & mask).astype(np.uint8).reshape(-1, per_byte)
    packed = np.zeros((unsigned.shape[0],), dtype=np.uint8)
    for i in range(per_byte):
        packed |= unsigned[:, i] << (i * bits)
    return packed.tobytes(order="C")


def read_quantized_weight(data: bytearray, rec: TensorRecord) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    bits = rec.bits
    if bits is None:
        raise ValueError(f"Unsupported tensor type for low-bit read: {rec.type_code}")
    raw = data[rec.data_pos : rec.data_pos + rec.data_len]
    q = unpack_lowbit(raw, bits, rec.shape)
    scales = read_scales(data, rec)
    zps = read_zero_points(data, rec)
    if rec.quantized_dimension != 0:
        raise ValueError(
            f"Only quantized_dimension=0 is supported, got {rec.quantized_dimension}: {rec.name}"
        )
    if len(scales) != rec.shape[0]:
        raise ValueError(f"scale length {len(scales)} does not match rows {rec.shape[0]}")
    if len(zps) not in (0, rec.shape[0]):
        raise ValueError(f"zero-point length {len(zps)} does not match rows {rec.shape[0]}")
    return q, scales, zps


def dequantize(q: np.ndarray, scales: np.ndarray, zps: np.ndarray) -> np.ndarray:
    if zps.size == 0:
        zps = np.zeros((q.shape[0],), dtype=np.int64)
    return (q.astype(np.float32) - zps.astype(np.float32)[:, None]) * scales[:, None]


def quantize(
    weight: np.ndarray,
    bits: int,
    old_scales: np.ndarray,
    old_zps: np.ndarray,
    scale_mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    qmin = -(1 << (bits - 1))
    qmax = (1 << (bits - 1)) - 1
    if scale_mode == "keep":
        scales = old_scales.astype(np.float32).copy()
        scales = np.where(scales == 0.0, 1.0, scales)
        zps = old_zps.astype(np.float32) if old_zps.size else 0.0
        q = np.rint(weight / scales[:, None] + zps[:, None])
    elif scale_mode == "recompute":
        bound = np.max(np.abs(weight), axis=1)
        # Google LiteRT-LM uses the absolute negative bound for signed low-bit
        # weights: int2 -> 2, int4 -> 8. Using the positive qmax (1 or 7)
        # over-scales rows and destroys the model.
        denom = float(1 << (bits - 1))
        scales = np.where(bound == 0.0, 1.0, bound / denom).astype(np.float32)
        q = np.rint(weight / scales[:, None])
    else:
        raise ValueError(f"unknown scale mode: {scale_mode}")
    q = np.clip(q, qmin, qmax).astype(np.int8)
    return q, scales


def patch_adapter(args: argparse.Namespace) -> None:
    lora_path = resolve_lora_file(args.lora)
    scale = read_lora_scale(args.lora)
    reader = SafeTensorReader(lora_path)
    data, model, subgraph = load_tflite(args.input)
    rec = find_audio_adapter_tensor(model, subgraph)
    base = read_float32_matrix(data, rec)

    module_path = "embed_audio.embedding_projection"
    delta = lora_delta(reader, module_path, scale)
    if args.transpose_delta:
        delta = delta.T.copy()
    if delta.shape != base.shape:
        raise ValueError(f"delta shape {delta.shape} does not match adapter {base.shape}")
    delta = delta * np.float32(args.delta_gain)

    patched = base + delta
    write_float32_matrix(data, rec, patched)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(data)
    print(f"[adapter] tensor #{rec.index}: {rec.name}")
    print(f"[adapter] base shape: {base.shape}")
    print(f"[adapter] delta_gain={args.delta_gain:.9g}")
    print(f"[adapter] delta mean_abs={np.mean(np.abs(delta)):.6g} max_abs={np.max(np.abs(delta)):.6g}")
    print(f"[adapter] wrote: {args.output}")


def roundtrip_encoder(args: argparse.Namespace) -> None:
    data, model, subgraph = load_tflite(args.input)
    checked = 0
    failures = 0
    for layer, module in encoder_targets(args.layers, args.modules):
        rec = find_encoder_tensor(model, subgraph, layer, module)
        before = bytes(data[rec.data_pos : rec.data_pos + rec.data_len])
        q, _, _ = read_quantized_weight(data, rec)
        after = pack_lowbit(q, rec.bits or 0)
        ok = before == after
        checked += 1
        if not ok:
            failures += 1
        print(
            f"[roundtrip] layer={layer:02d} module={module:<12} "
            f"type={rec.type_code} bits={rec.bits} shape={rec.shape} "
            f"bytes={rec.data_len} ok={ok}"
        )
    if failures:
        raise SystemExit(f"[roundtrip] {failures}/{checked} tensor(s) failed")
    print(f"[roundtrip] all {checked} tensor(s) packed back byte-identically")


def patch_encoder(args: argparse.Namespace) -> None:
    lora_path = resolve_lora_file(args.lora)
    scale = read_lora_scale(args.lora)
    reader = SafeTensorReader(lora_path)
    data, model, subgraph = load_tflite(args.input)

    total_changed = 0
    total_tensors = 0
    for layer, module in encoder_targets(args.layers, args.modules):
        module_suffix, _ = AUDIO_MODULE_MAP[module]
        module_path = f"audio_tower.layers.{layer}.{module_suffix}"
        rec = find_encoder_tensor(model, subgraph, layer, module)
        q_old, scales_old, zps_old = read_quantized_weight(data, rec)
        weight = dequantize(q_old, scales_old, zps_old)
        delta = lora_delta(reader, module_path, scale) * np.float32(args.delta_scale)
        if args.transpose_delta:
            delta = delta.T.copy()
        if delta.shape != weight.shape:
            raise ValueError(
                f"delta shape {delta.shape} does not match tensor {weight.shape}: "
                f"layer={layer} module={module}"
            )
        q_new, scales_new = quantize(
            weight + delta,
            rec.bits or 0,
            scales_old,
            zps_old,
            args.scale_mode,
        )
        packed = pack_lowbit(q_new, rec.bits or 0)
        if len(packed) != rec.data_len:
            raise ValueError(f"packed size mismatch for {rec.name}")
        old_bytes = bytes(data[rec.data_pos : rec.data_pos + rec.data_len])
        changed = sum(a != b for a, b in zip(old_bytes, packed))
        data[rec.data_pos : rec.data_pos + rec.data_len] = packed
        if args.scale_mode == "recompute":
            write_scales(data, rec, scales_new)
        total_changed += changed
        total_tensors += 1
        print(
            f"[encoder] layer={layer:02d} module={module:<12} "
            f"bits={rec.bits} shape={rec.shape} "
            f"delta_mean={np.mean(np.abs(delta)):.6g} "
            f"delta_max={np.max(np.abs(delta)):.6g} "
            f"changed_bytes={changed}/{rec.data_len}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(data)
    print(f"[encoder] patched tensors: {total_tensors}")
    print(f"[encoder] changed packed bytes: {total_changed}")
    if total_changed == 0:
        print(
            "[encoder] WARNING: no packed bytes changed. The LoRA delta may be "
            "below the int2/int4 quantization step; try inspecting scale ratios "
            "before trusting this path."
        )
    print(f"[encoder] wrote: {args.output}")


def delta_stats(args: argparse.Namespace) -> None:
    lora_path = resolve_lora_file(args.lora)
    scale = read_lora_scale(args.lora)
    reader = SafeTensorReader(lora_path)
    data, model, subgraph = load_tflite(args.input)

    total_changed_values = 0
    total_values = 0
    for layer, module in encoder_targets(args.layers, args.modules):
        module_suffix, _ = AUDIO_MODULE_MAP[module]
        module_path = f"audio_tower.layers.{layer}.{module_suffix}"
        rec = find_encoder_tensor(model, subgraph, layer, module)
        q_old, scales_old, zps_old = read_quantized_weight(data, rec)
        weight = dequantize(q_old, scales_old, zps_old)
        delta = lora_delta(reader, module_path, scale)
        if args.transpose_delta:
            delta = delta.T.copy()
        if delta.shape != weight.shape:
            raise ValueError(
                f"delta shape {delta.shape} does not match tensor {weight.shape}: "
                f"layer={layer} module={module}"
            )
        q_keep, _ = quantize(
            weight + delta,
            rec.bits or 0,
            scales_old,
            zps_old,
            "keep",
        )
        changed_values = int(np.count_nonzero(q_keep != q_old))
        total_changed_values += changed_values
        total_values += int(q_old.size)
        median_scale = float(np.median(scales_old))
        max_scale = float(np.max(scales_old))
        max_delta = float(np.max(np.abs(delta)))
        mean_delta = float(np.mean(np.abs(delta)))
        print(
            f"[delta] layer={layer:02d} module={module:<12} bits={rec.bits} "
            f"shape={rec.shape} changed_values_keep={changed_values}/{q_old.size} "
            f"mean_delta={mean_delta:.6g} max_delta={max_delta:.6g} "
            f"median_scale={median_scale:.6g} max_delta/median_scale={max_delta / median_scale:.6g} "
            f"max_scale={max_scale:.6g}"
        )
    print(
        f"[delta] total changed quantized values with keep-scale: "
        f"{total_changed_values}/{total_values}"
    )


def inspect(args: argparse.Namespace) -> None:
    lora_path = resolve_lora_file(args.lora)
    scale = read_lora_scale(args.lora)
    reader = SafeTensorReader(lora_path)
    keys = reader.keys()
    area_counts: dict[str, int] = {}
    for key in keys:
        m = re.search(r"base_model\.model\.model\.([^.]+)", key)
        area = m.group(1) if m else "<other>"
        area_counts[area] = area_counts.get(area, 0) + 1
    print(f"[lora] file: {lora_path}")
    print(f"[lora] alpha/rank scale: {scale:g}")
    print(f"[lora] tensor counts: {area_counts}")

    if args.audio_adapter.exists():
        data, model, subgraph = load_tflite(args.audio_adapter)
        rec = find_audio_adapter_tensor(model, subgraph)
        delta = lora_delta(reader, "embed_audio.embedding_projection", scale)
        print()
        print(f"[adapter] {args.audio_adapter}")
        print(f"[adapter] tensor #{rec.index} shape={rec.shape} data_pos={rec.data_pos} bytes={rec.data_len}")
        print(f"[adapter] delta mean_abs={np.mean(np.abs(delta)):.6g} max_abs={np.max(np.abs(delta)):.6g}")

    if args.audio_encoder.exists():
        data, model, subgraph = load_tflite(args.audio_encoder)
        print()
        print(f"[encoder] {args.audio_encoder}")
        matched = 0
        type_counts: dict[int, int] = {}
        for layer, module in encoder_targets(list(range(12)), list(AUDIO_MODULE_MAP)):
            rec = find_encoder_tensor(model, subgraph, layer, module)
            matched += 1
            type_counts[rec.type_code] = type_counts.get(rec.type_code, 0) + 1
        print(f"[encoder] mapped target tensors: {matched} (expected 120)")
        print(f"[encoder] tensor type counts: {type_counts}")
        rec = find_encoder_tensor(model, subgraph, 0, "ff1_l1")
        print(
            "[encoder] sample layer0/ff1_l1: "
            f"tensor=#{rec.index} type={rec.type_code} bits={rec.bits} "
            f"shape={rec.shape} data_pos={rec.data_pos} "
            f"scale_pos={rec.scale_pos} zero_point_pos={rec.zero_point_pos}"
        )


def _section_model_type(section_object) -> str | None:
    if litertlm_schema is None:
        return None
    for j in range(section_object.ItemsLength()):
        item = section_object.Items(j)
        if item is None:
            continue
        key = item.Key()
        if not key or key.decode("utf-8") != "model_type":
            continue
        if item.ValueType() != litertlm_schema.VData.StringValue:
            continue
        value = item.Value()
        sv = litertlm_schema.StringValue()
        sv.Init(value.Bytes, value.Pos)
        raw = sv.Value()
        return raw.decode("utf-8") if raw else None
    return None


def litertlm_sections(path: Path) -> dict[str, tuple[int, int]]:
    if litertlm_schema is None:
        raise RuntimeError(
            "litert_lm_builder schema is unavailable. Expected local path: "
            f"{ROOT / '.tmp' / 'litert_lm_builder_wheel'}"
        )
    with path.open("rb") as f:
        if f.read(8) != b"LITERTLM":
            raise ValueError(f"not a LiteRT-LM file: {path}")
        f.seek(24)
        header_end = struct.unpack("<Q", f.read(8))[0]
        f.seek(32)
        header_data = f.read(header_end - 32)
    metadata = litertlm_schema.LiteRTLMMetaData.GetRootAs(header_data, 0)
    section_metadata = metadata.SectionMetadata()
    out: dict[str, tuple[int, int]] = {}
    for i in range(section_metadata.ObjectsLength()):
        section = section_metadata.Objects(i)
        model_type = _section_model_type(section)
        if model_type:
            out[model_type] = (int(section.BeginOffset()), int(section.EndOffset()))
    return out


def patch_litertlm(args: argparse.Namespace) -> None:
    replacements: dict[str, Path] = {}
    if args.audio_adapter:
        replacements["tf_lite_audio_adapter"] = args.audio_adapter
    if args.audio_encoder:
        replacements["tf_lite_audio_encoder_hw"] = args.audio_encoder
    if getattr(args, "prefill_decode", None):
        replacements["tf_lite_prefill_decode"] = args.prefill_decode
    if not replacements:
        raise SystemExit("No replacement sections were provided.")

    sections = litertlm_sections(args.input)
    missing = [name for name in replacements if name not in sections]
    if missing:
        raise ValueError(f"section(s) not found in .litertlm: {missing}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.resolve() != args.input.resolve():
        shutil.copyfile(args.input, args.output)

    with args.output.open("r+b") as f:
        for model_type, replacement in replacements.items():
            begin, end = sections[model_type]
            expected = end - begin
            actual = replacement.stat().st_size
            if actual != expected:
                raise ValueError(
                    f"replacement size mismatch for {model_type}: "
                    f"{actual} != {expected}"
                )
            f.seek(begin)
            with replacement.open("rb") as src:
                shutil.copyfileobj(src, f, length=1024 * 1024)
            print(f"[litertlm] replaced {model_type}: offset={begin} size={expected}")
    print(f"[litertlm] wrote: {args.output}")


def add_common_lora_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--lora",
        type=Path,
        default=DEFAULT_LORA,
        help="LoRA dir or adapter_model.safetensors path",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("inspect", help="summarize LoRA and TFLite mapping")
    add_common_lora_arg(p)
    p.add_argument("--audio-adapter", type=Path, default=DEFAULT_ADAPTER)
    p.add_argument("--audio-encoder", type=Path, default=DEFAULT_ENCODER)
    p.set_defaults(func=inspect)

    p = sub.add_parser("patch-adapter", help="patch tf_lite_audio_adapter")
    add_common_lora_arg(p)
    p.add_argument("--input", type=Path, default=DEFAULT_ADAPTER)
    p.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/litertlm_patch/audio_adapter.lora.tflite",
    )
    p.add_argument(
        "--transpose-delta",
        action="store_true",
        help="transpose B@A before adding; default is no transpose",
    )
    p.add_argument(
        "--delta-gain",
        type=float,
        default=1.0,
        help="multiplier for the adapter LoRA delta before adding to the official section",
    )
    p.set_defaults(func=patch_adapter)

    p = sub.add_parser("roundtrip-encoder", help="verify low-bit pack/unpack")
    p.add_argument("--input", type=Path, default=DEFAULT_ENCODER)
    p.add_argument("--layers", type=parse_layers, default=parse_layers("0"))
    p.add_argument("--modules", type=parse_modules, default=parse_modules("q_proj"))
    p.set_defaults(func=roundtrip_encoder)

    p = sub.add_parser("patch-encoder", help="patch tf_lite_audio_encoder_hw")
    add_common_lora_arg(p)
    p.add_argument("--input", type=Path, default=DEFAULT_ENCODER)
    p.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/litertlm_patch/audio_encoder_hw.lora.tflite",
    )
    p.add_argument("--layers", type=parse_layers, default=parse_layers("0"))
    p.add_argument("--modules", type=parse_modules, default=parse_modules("q_proj"))
    p.add_argument(
        "--scale-mode",
        choices=("keep", "recompute"),
        default="keep",
        help="keep current per-row scales or recompute them after applying LoRA",
    )
    p.add_argument(
        "--delta-scale",
        type=float,
        default=1.0,
        help="debug multiplier for LoRA delta; keep 1.0 for real patches",
    )
    p.add_argument(
        "--transpose-delta",
        action="store_true",
        help="transpose B@A before adding; default is no transpose",
    )
    p.set_defaults(func=patch_encoder)

    p = sub.add_parser(
        "delta-stats",
        help="estimate whether LoRA deltas survive current int2/int4 scales",
    )
    add_common_lora_arg(p)
    p.add_argument("--input", type=Path, default=DEFAULT_ENCODER)
    p.add_argument("--layers", type=parse_layers, default=parse_layers("0"))
    p.add_argument("--modules", type=parse_modules, default=parse_modules("q_proj"))
    p.add_argument(
        "--transpose-delta",
        action="store_true",
        help="transpose B@A before comparing; default is no transpose",
    )
    p.set_defaults(func=delta_stats)

    p = sub.add_parser(
        "patch-litertlm",
        help="replace same-size audio TFLite sections in a .litertlm",
    )
    p.add_argument("--input", type=Path, default=DEFAULT_LITERTLM)
    p.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/litertlm_patch/gemma-4-E2B-it.audio-lora.litertlm",
    )
    p.add_argument("--audio-adapter", type=Path)
    p.add_argument("--audio-encoder", type=Path)
    p.add_argument("--prefill-decode", type=Path)
    p.set_defaults(func=patch_litertlm)

    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
