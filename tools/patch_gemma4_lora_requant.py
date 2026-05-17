#!/usr/bin/env python3
"""Build full-LoRA LiteRT-LM requantization candidates.

This is the main-track patcher:

    official .litertlm base weight
      -> dequantize low-bit tensor
      -> add the full PEFT LoRA delta
      -> requantize with LiteRT-LM-compatible low-bit packing
      -> replace same-size TFLite sections inside the bundle

Unlike qproj, this does not select a flip fraction.  It tries to preserve the
trained LoRA itself.  `keep` is conservative but often below the quantization
floor.  `recompute` uses the reverse-engineered Google scale formula:

    scale = max(abs(weight), axis=1) / 2^(bits - 1)

for int2/int4 signed weights.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

import patch_gemma4_audio_lora as patcher  # noqa: E402
from patch_gemma4_quant_projected import LM_MODULE_MAP, parse_lm_layer_module  # noqa: E402


DEFAULT_LITERTLM = ROOT / "models/gemma-4-E2B-it.gallery-7fa1d78.litertlm"
DEFAULT_SECTIONS = ROOT / ".tmp/litertlm_sections"
DEFAULT_LORA = ROOT / "outputs/gemma4_audio_v2/lora"
DEFAULT_OUT = ROOT / "outputs/litertlm_patch"


@dataclass
class SectionStats:
    tensors: int = 0
    values: int = 0
    changed_values: int = 0
    byte_diff: int = 0
    scale_byte_diff: int = 0
    byte_total: int = 0


def quantize_google(weight: np.ndarray, bits: int) -> tuple[np.ndarray, np.ndarray]:
    qmin = -(1 << (bits - 1))
    qmax = (1 << (bits - 1)) - 1
    qmax_abs = float(1 << (bits - 1))
    bound = np.max(np.abs(weight), axis=1)
    scales = np.where(bound == 0.0, 1.0, bound / qmax_abs).astype(np.float32)
    q = np.rint(weight / scales[:, None])
    return np.clip(q, qmin, qmax).astype(np.int8), scales


def quantize_lsq(
    weight: np.ndarray,
    bits: int,
    init_scales: np.ndarray,
    iterations: int,
) -> tuple[np.ndarray, np.ndarray]:
    qmin = -(1 << (bits - 1))
    qmax = (1 << (bits - 1)) - 1
    scales = np.where(init_scales == 0.0, 1.0, init_scales).astype(np.float32).copy()
    q = np.zeros(weight.shape, dtype=np.int8)
    for _ in range(max(1, int(iterations))):
        q = np.clip(np.rint(weight / scales[:, None]), qmin, qmax).astype(np.int8)
        qf = q.astype(np.float32)
        denom = np.sum(qf * qf, axis=1)
        numer = np.sum(qf * weight, axis=1)
        fitted = np.where(denom > 0.0, numer / denom, scales)
        scales = np.where(fitted > 0.0, fitted, scales).astype(np.float32)
    q = np.clip(np.rint(weight / scales[:, None]), qmin, qmax).astype(np.int8)
    return q, scales


def orient_delta(delta: np.ndarray, shape: tuple[int, ...], label: str) -> np.ndarray:
    if delta.shape == shape:
        return delta
    if delta.T.shape == shape:
        return delta.T.copy()
    raise ValueError(f"{label}: delta shape {delta.shape} does not match tensor {shape}")


def requant_tensor(
    data: bytearray,
    rec: patcher.TensorRecord,
    delta: np.ndarray,
    mode: str,
    lsq_iters: int,
) -> tuple[int, int, int]:
    q_old, scales_old, zps_old = patcher.read_quantized_weight(data, rec)
    base = patcher.dequantize(q_old, scales_old, zps_old)
    target = base + delta
    if mode == "keep":
        q_new, scales_new = patcher.quantize(
            target,
            rec.bits or 0,
            scales_old,
            zps_old,
            "keep",
        )
        write_scale = False
    elif mode == "recompute":
        q_new, scales_new = quantize_google(target, rec.bits or 0)
        write_scale = True
    elif mode == "lsq":
        q_new, scales_new = quantize_lsq(target, rec.bits or 0, scales_old, lsq_iters)
        write_scale = True
    else:
        raise ValueError(f"bad requant mode: {mode}")

    packed = patcher.pack_lowbit(q_new, rec.bits or 0)
    if len(packed) != rec.data_len:
        raise ValueError(f"packed size mismatch for {rec.name}: {len(packed)} != {rec.data_len}")
    old_bytes = bytes(data[rec.data_pos : rec.data_pos + rec.data_len])
    byte_diff = sum(a != b for a, b in zip(old_bytes, packed))
    changed_values = int(np.count_nonzero(q_new != q_old))
    data[rec.data_pos : rec.data_pos + rec.data_len] = packed
    scale_byte_diff = 0
    if write_scale:
        old_scale_bytes = bytes(data[rec.scale_pos : rec.scale_pos + rec.scale_len * 4])
        patcher.write_scales(data, rec, scales_new)
        new_scale_bytes = bytes(data[rec.scale_pos : rec.scale_pos + rec.scale_len * 4])
        scale_byte_diff = sum(a != b for a, b in zip(old_scale_bytes, new_scale_bytes))
    return changed_values, byte_diff, scale_byte_diff


def patch_adapter(
    input_path: Path,
    output_path: Path,
    reader: patcher.SafeTensorReader,
    lora_scale: float,
) -> int:
    data, model, subgraph = patcher.load_tflite(input_path)
    rec = patcher.find_audio_adapter_tensor(model, subgraph)
    base = patcher.read_float32_matrix(data, rec)
    delta = orient_delta(
        patcher.lora_delta(reader, "embed_audio.embedding_projection", lora_scale),
        base.shape,
        "embed_audio.embedding_projection",
    )
    patched = base + delta
    before = bytes(data[rec.data_pos : rec.data_pos + rec.data_len])
    patcher.write_float32_matrix(data, rec, patched)
    after = bytes(data[rec.data_pos : rec.data_pos + rec.data_len])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(data)
    return sum(a != b for a, b in zip(before, after))


def patch_audio_encoder(
    input_path: Path,
    output_path: Path,
    reader: patcher.SafeTensorReader,
    lora_scale: float,
    mode: str,
    lsq_iters: int,
) -> SectionStats:
    data, model, subgraph = patcher.load_tflite(input_path)
    stats = SectionStats()
    for layer, module in patcher.encoder_targets(list(range(12)), list(patcher.AUDIO_MODULE_MAP)):
        suffix, _ = patcher.AUDIO_MODULE_MAP[module]
        module_path = f"audio_tower.layers.{layer}.{suffix}"
        rec = patcher.find_encoder_tensor(model, subgraph, layer, module)
        delta = orient_delta(patcher.lora_delta(reader, module_path, lora_scale), rec.shape, module_path)
        q_old, _, _ = patcher.read_quantized_weight(data, rec)
        changed_values, byte_diff, scale_byte_diff = requant_tensor(data, rec, delta, mode, lsq_iters)
        stats.tensors += 1
        stats.values += int(q_old.size)
        stats.changed_values += changed_values
        stats.byte_diff += byte_diff
        stats.scale_byte_diff += scale_byte_diff
        stats.byte_total += rec.data_len
        if stats.tensors <= 8 or stats.tensors % 30 == 0:
            print(
                f"[audio] layer={layer:02d} module={module:<12} bits={rec.bits} "
                f"changed={changed_values}/{q_old.size} byte_diff={byte_diff}/{rec.data_len} "
                f"scale_diff={scale_byte_diff}"
            )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(data)
    return stats


def patch_lm_prefill(
    input_path: Path,
    output_path: Path,
    reader: patcher.SafeTensorReader,
    lora_scale: float,
    mode: str,
    lsq_iters: int,
) -> SectionStats:
    data, model, subgraph = patcher.load_tflite(input_path)
    stats = SectionStats()
    for rec in patcher.iter_const_tensors(model, subgraph):
        if rec.type_code not in (patcher.TENSOR_TYPE_INT2, patcher.TENSOR_TYPE_INT4):
            continue
        parsed = parse_lm_layer_module(rec.name)
        if parsed is None:
            continue
        layer, tag = parsed
        suffix = LM_MODULE_MAP[tag].removesuffix(".weight")
        module_path = f"language_model.layers.{layer}.{suffix}"
        delta = orient_delta(patcher.lora_delta(reader, module_path, lora_scale), rec.shape, module_path)
        q_old, _, _ = patcher.read_quantized_weight(data, rec)
        changed_values, byte_diff, scale_byte_diff = requant_tensor(data, rec, delta, mode, lsq_iters)
        stats.tensors += 1
        stats.values += int(q_old.size)
        stats.changed_values += changed_values
        stats.byte_diff += byte_diff
        stats.scale_byte_diff += scale_byte_diff
        stats.byte_total += rec.data_len
        if stats.tensors <= 8 or stats.tensors % 30 == 0:
            print(
                f"[lm] layer={layer:02d} tag={tag:<18} bits={rec.bits} "
                f"changed={changed_values}/{q_old.size} byte_diff={byte_diff}/{rec.data_len} "
                f"scale_diff={scale_byte_diff}"
            )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(data)
    return stats


def patch_bundle(input_litertlm: Path, output_litertlm: Path, replacements: dict[str, Path]) -> None:
    sections = patcher.litertlm_sections(input_litertlm)
    output_litertlm.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(input_litertlm, output_litertlm)
    with output_litertlm.open("r+b") as f:
        for model_type, replacement in replacements.items():
            begin, end = sections[model_type]
            expected = end - begin
            actual = replacement.stat().st_size
            if actual != expected:
                raise ValueError(f"{model_type} size mismatch: {actual} != {expected}")
            f.seek(begin)
            with replacement.open("rb") as src:
                shutil.copyfileobj(src, f, length=16 * 1024 * 1024)
            print(f"[bundle] replaced {model_type}: {expected/1024/1024:.1f} MB")


def summarize(name: str, stats: SectionStats) -> None:
    print(
        f"[{name} summary] tensors={stats.tensors} "
        f"changed_values={stats.changed_values}/{stats.values} "
        f"byte_diff={stats.byte_diff}/{stats.byte_total} "
        f"scale_byte_diff={stats.scale_byte_diff}"
    )


def default_tag(adapter: bool, audio_mode: str, lm_mode: str) -> str:
    parts = ["full-lora-v2"]
    parts.append("adapter" if adapter else "noadapter")
    parts.append(f"audio-{audio_mode}")
    parts.append(f"lm-{lm_mode}")
    return "-".join(parts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-litertlm", type=Path, default=DEFAULT_LITERTLM)
    ap.add_argument("--sections-dir", type=Path, default=DEFAULT_SECTIONS)
    ap.add_argument("--lora-dir", type=Path, default=DEFAULT_LORA)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--tag", default="")
    ap.add_argument("--adapter", choices=("apply", "skip"), default="apply")
    ap.add_argument("--audio-mode", choices=("skip", "keep", "recompute", "lsq"), default="recompute")
    ap.add_argument("--lm-mode", choices=("skip", "keep", "recompute", "lsq"), default="recompute")
    ap.add_argument("--lsq-iters", type=int, default=4)
    ap.add_argument("--sections-only", action="store_true")
    args = ap.parse_args()

    lora_path = patcher.resolve_lora_file(args.lora_dir)
    lora_scale = patcher.read_lora_scale(args.lora_dir)
    reader = patcher.SafeTensorReader(lora_path)
    tag = args.tag or default_tag(args.adapter == "apply", args.audio_mode, args.lm_mode)
    print(f"[setup] lora={lora_path} alpha/rank={lora_scale:g}")
    print(f"[setup] tag={tag}")

    replacements: dict[str, Path] = {}
    adapter_out = args.out_dir / f"audio_adapter.{tag}.tflite"
    audio_out = args.out_dir / f"audio_encoder_hw.{tag}.tflite"
    lm_out = args.out_dir / f"prefill_decode.{tag}.tflite"

    if args.adapter == "apply":
        diff = patch_adapter(args.sections_dir / "audio_adapter.tflite", adapter_out, reader, lora_scale)
        replacements["tf_lite_audio_adapter"] = adapter_out
        print(f"[adapter] byte_diff={diff}/{adapter_out.stat().st_size}")

    if args.audio_mode != "skip":
        stats = patch_audio_encoder(
            args.sections_dir / "audio_encoder_hw.tflite",
            audio_out,
            reader,
            lora_scale,
            args.audio_mode,
            args.lsq_iters,
        )
        replacements["tf_lite_audio_encoder_hw"] = audio_out
        summarize("audio", stats)

    if args.lm_mode != "skip":
        stats = patch_lm_prefill(
            args.sections_dir / "prefill_decode.tflite",
            lm_out,
            reader,
            lora_scale,
            args.lm_mode,
            args.lsq_iters,
        )
        replacements["tf_lite_prefill_decode"] = lm_out
        summarize("lm", stats)

    if args.sections_only:
        print("[done] sections only")
        return

    bundle_out = args.out_dir / f"gemma-4-E2B-it.{tag}.litertlm"
    patch_bundle(args.input_litertlm, bundle_out, replacements)
    print(f"[wrote] {bundle_out} ({bundle_out.stat().st_size/1024/1024:.1f} MB)")


if __name__ == "__main__":
    main()
