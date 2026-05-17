"""Project Gemma 4 LoRA deltas onto the LiteRT-LM int2/int4 quant grid.

Plain post-hoc LoRA addition disappears because most deltas are smaller than
half a Google INT2/INT4 quantization step. This tool instead selects the
largest |delta| / scale entries and moves each selected quantized value by one
integer step in the LoRA direction. It is a controlled "quant-grid projection":
no scale recomputation, no fp16 re-quantization noise, and a tunable flip budget.
"""
from __future__ import annotations

import argparse
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

import patch_gemma4_audio_lora as patcher  # noqa: E402


DEFAULT_LITERTLM = ROOT / "models/gemma-4-E2B-it.gallery-7fa1d78.litertlm"
DEFAULT_SECTIONS = ROOT / ".tmp/litertlm_sections"
DEFAULT_LORA_DIR = ROOT / "outputs/gemma4_audio_qat/lora"
DEFAULT_FIT_MERGED_DIR = ROOT / "outputs/merged_gemma4_audio_qat"
DEFAULT_OUT_DIR = ROOT / "outputs/litertlm_patch"

LM_MODULE_MAP = {
    "q_einsum": "self_attn.q_proj.weight",
    "k_einsum": "self_attn.k_proj.weight",
    "v_einsum": "self_attn.v_proj.weight",
    "attn_vec_einsum": "self_attn.o_proj.weight",
    "gating_einsum1": "mlp.gate_proj.weight",
    "gating_einsum2": "mlp.up_proj.weight",
    "linear": "mlp.down_proj.weight",
}


def parse_csv_filter(value: str, valid: set[str], label: str) -> set[str] | None:
    if not value:
        return None
    items = {part.strip() for part in value.split(",") if part.strip()}
    unknown = items - valid
    if unknown:
        raise ValueError(f"Unknown {label}: {sorted(unknown)}. Valid: {sorted(valid)}")
    return items


def layer_allowed(layer: int, start: int | None, end: int | None) -> bool:
    if start is not None and layer < start:
        return False
    if end is not None and layer > end:
        return False
    return True


@dataclass
class FlipStats:
    tensors: int = 0
    values: int = 0
    flips: int = 0
    byte_diff: int = 0
    byte_total: int = 0


@dataclass(frozen=True)
class FoldOverrideRule:
    layer_start: int | None
    layer_end: int | None
    tags: set[str] | None
    mode: str


def parse_lm_layer_module(name: str) -> tuple[int, str] | None:
    m = re.search(r"layer_(\d+)", name)
    if not m:
        return None
    layer = int(m.group(1))
    for tag in LM_MODULE_MAP:
        if f"/{tag}/" in name + "/":
            return layer, tag
    for part in name.split("/"):
        if part in LM_MODULE_MAP:
            return layer, part
    return None


def parse_fold_override_specs(value: str) -> list[FoldOverrideRule]:
    """Parse semicolon rules: start:end:tag1,tag2:mode.

    Example:
        0:14:v_einsum:none;21:29:q_einsum,attn_vec_einsum:none
    Use '*' for all tags and an empty start/end for open ranges.
    """
    rules: list[FoldOverrideRule] = []
    if not value:
        return rules
    valid_modes = {"none", "global", "row", "rowcol"}
    valid_tags = set(LM_MODULE_MAP)
    for raw_rule in value.split(";"):
        raw_rule = raw_rule.strip()
        if not raw_rule:
            continue
        parts = raw_rule.split(":")
        if len(parts) != 4:
            raise ValueError(f"Bad fold override rule {raw_rule!r}; expected start:end:tags:mode")
        start_s, end_s, tags_s, mode = (part.strip() for part in parts)
        if mode not in valid_modes:
            raise ValueError(f"Bad fold override mode {mode!r}; valid={sorted(valid_modes)}")
        tags = None
        if tags_s and tags_s != "*":
            tags = {tag.strip() for tag in tags_s.split(",") if tag.strip()}
            unknown = tags - valid_tags
            if unknown:
                raise ValueError(f"Bad fold override tag(s) {sorted(unknown)}; valid={sorted(valid_tags)}")
        rules.append(
            FoldOverrideRule(
                layer_start=int(start_s) if start_s else None,
                layer_end=int(end_s) if end_s else None,
                tags=tags,
                mode=mode,
            )
        )
    return rules


def project_one_step(
    q_old: np.ndarray,
    scales: np.ndarray,
    delta: np.ndarray,
    bits: int,
    fraction: float,
    min_score: float,
) -> tuple[np.ndarray, int]:
    """Move top |delta|/scale values by one q step in LoRA direction."""
    qmin = -(1 << (bits - 1))
    qmax = (1 << (bits - 1)) - 1
    q_new = q_old.copy()
    if fraction <= 0.0:
        return q_new, 0

    direction = np.sign(delta).astype(np.int8)
    movable = ((direction > 0) & (q_old < qmax)) | ((direction < 0) & (q_old > qmin))
    if not np.any(movable):
        return q_new, 0

    safe_scales = np.where(scales == 0.0, 1.0, scales).astype(np.float32)
    score = np.abs(delta).astype(np.float32) / safe_scales[:, None]
    score = np.where(movable, score, -np.inf)
    if min_score > 0.0:
        score = np.where(score >= min_score, score, -np.inf)

    flat_score = score.reshape(-1)
    finite = int(np.count_nonzero(np.isfinite(flat_score)))
    if finite == 0:
        return q_new, 0
    k = max(1, int(round(q_old.size * fraction)))
    k = min(k, finite)
    selected = np.argpartition(flat_score, -k)[-k:]

    flat_q = q_new.reshape(-1)
    flat_dir = direction.reshape(-1)
    before = flat_q[selected].copy()
    flat_q[selected] = np.clip(
        flat_q[selected].astype(np.int16) + flat_dir[selected].astype(np.int16),
        qmin,
        qmax,
    ).astype(np.int8)
    flips = int(np.count_nonzero(flat_q[selected] != before))
    return q_new, flips


def safe_gain(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    gain = num / np.maximum(den, np.float32(1e-12))
    return np.where(np.isfinite(gain), gain, np.float32(1.0)).astype(np.float32)


def align_to_shape(weight: np.ndarray, shape: tuple[int, ...]) -> tuple[np.ndarray, str]:
    if weight.shape == shape:
        return weight.astype(np.float32, copy=False), "same"
    if weight.ndim == 2 and weight.T.shape == shape:
        return weight.T.astype(np.float32, copy=False), "transpose"
    raise ValueError(f"cannot align weight shape {weight.shape} to TFLite shape {shape}")


def estimated_base_weight(
    merged_reader,
    lora_reader,
    lora_scale: float,
    module_path: str,
) -> np.ndarray:
    merged = merged_reader.read(f"model.{module_path}.weight").astype(np.float32)
    a_key = patcher.lora_key(module_path, "A")
    b_key = patcher.lora_key(module_path, "B")
    if a_key not in lora_reader.tensors or b_key not in lora_reader.tensors:
        return merged
    delta = patcher.lora_delta(lora_reader, module_path, lora_scale)
    if delta.shape != merged.shape and delta.T.shape == merged.shape:
        delta = delta.T.copy()
    if delta.shape != merged.shape:
        raise ValueError(f"fit LoRA delta shape mismatch {module_path}: {delta.shape} vs {merged.shape}")
    return (merged - delta).astype(np.float32)


def fit_rowcol(
    target: np.ndarray,
    source: np.ndarray,
    iterations: int,
    clamp: float,
) -> tuple[np.ndarray, np.ndarray]:
    rows, cols = source.shape
    row = np.ones((rows,), dtype=np.float32)
    col = np.ones((cols,), dtype=np.float32)
    target = target.astype(np.float32, copy=False)
    source = source.astype(np.float32, copy=False)
    lo = np.float32(1.0 / clamp)
    hi = np.float32(clamp)
    for _ in range(iterations):
        source_col = source * col[None, :]
        row = safe_gain(np.sum(target * source_col, axis=1), np.sum(source_col * source_col, axis=1))
        row = np.clip(row, lo, hi).astype(np.float32)
        row_source = source * row[:, None]
        col = safe_gain(np.sum(target * row_source, axis=0), np.sum(row_source * row_source, axis=0))
        col = np.clip(col, lo, hi).astype(np.float32)
    return row, col


def folded_delta(
    delta: np.ndarray,
    official_deq: np.ndarray,
    hf_base: np.ndarray,
    mode: str,
    iterations: int,
    clamp: float,
) -> np.ndarray:
    if mode == "none":
        return delta
    if mode == "global":
        num = np.dot(official_deq.reshape(-1).astype(np.float32), hf_base.reshape(-1).astype(np.float32))
        den = np.dot(hf_base.reshape(-1).astype(np.float32), hf_base.reshape(-1).astype(np.float32))
        gain = np.float32(num / max(float(den), 1e-12))
        return (delta * gain).astype(np.float32)
    if mode == "row":
        row = safe_gain(
            np.sum(official_deq.astype(np.float32) * hf_base.astype(np.float32), axis=1),
            np.sum(np.square(hf_base.astype(np.float32)), axis=1),
        )
        row = np.clip(row, np.float32(1.0 / clamp), np.float32(clamp)).astype(np.float32)
        return (delta * row[:, None]).astype(np.float32)
    if mode == "rowcol":
        row, col = fit_rowcol(official_deq, hf_base, iterations, clamp)
        return (delta * row[:, None] * col[None, :]).astype(np.float32)
    raise ValueError(f"unknown LM fold mode: {mode}")


def patch_audio_adapter(data: bytearray, model, subgraph, reader, lora_scale: float) -> int:
    rec = patcher.find_audio_adapter_tensor(model, subgraph)
    base = patcher.read_float32_matrix(data, rec)
    delta = patcher.lora_delta(reader, "embed_audio.embedding_projection", lora_scale)
    if delta.shape != base.shape and delta.T.shape == base.shape:
        delta = delta.T.copy()
    if delta.shape != base.shape:
        raise ValueError(f"audio adapter delta shape mismatch: {delta.shape} vs {base.shape}")
    patched = base + delta
    before = data[rec.data_pos : rec.data_pos + rec.data_len]
    patcher.write_float32_matrix(data, rec, patched)
    after = data[rec.data_pos : rec.data_pos + rec.data_len]
    return sum(a != b for a, b in zip(before, after))


def patch_audio_encoder(
    input_path: Path,
    output_path: Path,
    reader,
    lora_scale: float,
    fraction: float,
    min_score: float,
    layer_start: int | None = None,
    layer_end: int | None = None,
    tags: set[str] | None = None,
) -> FlipStats:
    data, model, subgraph = patcher.load_tflite(input_path)
    stats = FlipStats()
    for layer, module in patcher.encoder_targets(list(range(12)), list(patcher.AUDIO_MODULE_MAP)):
        if not layer_allowed(layer, layer_start, layer_end):
            continue
        if tags is not None and module not in tags:
            continue
        suffix, _ = patcher.AUDIO_MODULE_MAP[module]
        module_path = f"audio_tower.layers.{layer}.{suffix}"
        rec = patcher.find_encoder_tensor(model, subgraph, layer, module)
        q_old, scales, _ = patcher.read_quantized_weight(data, rec)
        delta = patcher.lora_delta(reader, module_path, lora_scale)
        if delta.shape != q_old.shape and delta.T.shape == q_old.shape:
            delta = delta.T.copy()
        if delta.shape != q_old.shape:
            raise ValueError(f"audio shape mismatch {module_path}: {delta.shape} vs {q_old.shape}")
        q_new, flips = project_one_step(q_old, scales, delta, rec.bits or 0, fraction, min_score)
        packed = patcher.pack_lowbit(q_new, rec.bits or 0)
        old_bytes = bytes(data[rec.data_pos : rec.data_pos + rec.data_len])
        byte_diff = sum(a != b for a, b in zip(old_bytes, packed))
        data[rec.data_pos : rec.data_pos + rec.data_len] = packed
        stats.tensors += 1
        stats.values += int(q_old.size)
        stats.flips += flips
        stats.byte_diff += byte_diff
        stats.byte_total += rec.data_len
        if stats.tensors <= 8 or stats.tensors % 30 == 0:
            print(
                f"[audio] layer={layer:02d} module={module:<12} bits={rec.bits} "
                f"flips={flips}/{q_old.size} byte_diff={byte_diff}/{rec.data_len}"
            )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(data)
    return stats


def patch_lm_prefill(
    input_path: Path,
    output_path: Path,
    reader,
    lora_scale: float,
    fraction: float,
    extra_fraction: float | None,
    min_score: float,
    layer_start: int | None = None,
    layer_end: int | None = None,
    tags: set[str] | None = None,
    exclude_tags: set[str] | None = None,
    extra_layer_start: int | None = None,
    extra_layer_end: int | None = None,
    extra_tags: set[str] | None = None,
    fold_mode: str = "none",
    extra_fold_mode: str | None = None,
    override_layer_start: int | None = None,
    override_layer_end: int | None = None,
    override_tags: set[str] | None = None,
    override_fold_mode: str | None = None,
    override_rules: list[FoldOverrideRule] | None = None,
    fit_merged_reader=None,
    fit_lora_reader=None,
    fit_lora_scale: float = 1.0,
    fold_iterations: int = 8,
    fold_clamp: float = 8.0,
    delta_gain: float = 1.0,
) -> FlipStats:
    data, model, subgraph = patcher.load_tflite(input_path)
    stats = FlipStats()

    def selected_modes(layer: int, tag: str) -> tuple[str, float] | None:
        main_ok = layer_allowed(layer, layer_start, layer_end)
        if tags is not None and tag not in tags:
            main_ok = False
        extra_ok = False
        if extra_layer_start is not None or extra_layer_end is not None:
            extra_ok = layer_allowed(layer, extra_layer_start, extra_layer_end)
            if extra_tags is not None and tag not in extra_tags:
                extra_ok = False
        if exclude_tags is not None and tag in exclude_tags:
            main_ok = False
            extra_ok = False
        selected = main_ok or extra_ok
        selected_fraction = extra_fraction if extra_ok and extra_fraction is not None else fraction
        if selected and override_rules:
            for rule in override_rules:
                override_ok = layer_allowed(layer, rule.layer_start, rule.layer_end)
                if rule.tags is not None and tag not in rule.tags:
                    override_ok = False
                if override_ok:
                    return rule.mode, selected_fraction
        if selected and override_fold_mode is not None:
            override_ok = layer_allowed(layer, override_layer_start, override_layer_end)
            if override_tags is not None and tag not in override_tags:
                override_ok = False
            if override_ok:
                return override_fold_mode, selected_fraction
        if extra_ok and extra_fold_mode is not None:
            return extra_fold_mode, selected_fraction
        if main_ok:
            return fold_mode, selected_fraction
        if extra_ok:
            return fold_mode, selected_fraction
        return None

    for rec in patcher.iter_const_tensors(model, subgraph):
        if rec.type_code not in (patcher.TENSOR_TYPE_INT2, patcher.TENSOR_TYPE_INT4):
            continue
        parsed = parse_lm_layer_module(rec.name)
        if parsed is None:
            continue
        layer, tag = parsed
        selected = selected_modes(layer, tag)
        if selected is None:
            continue
        tensor_fold_mode, tensor_fraction = selected
        suffix = LM_MODULE_MAP[tag].removesuffix(".weight")
        module_path = f"language_model.layers.{layer}.{suffix}"
        q_old, scales, zps = patcher.read_quantized_weight(data, rec)
        delta = patcher.lora_delta(reader, module_path, lora_scale)
        if delta.shape != q_old.shape and delta.T.shape == q_old.shape:
            delta = delta.T.copy()
        if delta.shape != q_old.shape:
            raise ValueError(f"LM shape mismatch {module_path}: {delta.shape} vs {q_old.shape}")
        if tensor_fold_mode != "none":
            if fit_merged_reader is None or fit_lora_reader is None:
                raise ValueError("--lm-fold-mode requires fit readers")
            official_deq = patcher.dequantize(q_old, scales, zps)
            hf_base_raw = estimated_base_weight(fit_merged_reader, fit_lora_reader, fit_lora_scale, module_path)
            hf_base, _ = align_to_shape(hf_base_raw, q_old.shape)
            delta = folded_delta(delta, official_deq, hf_base, tensor_fold_mode, fold_iterations, fold_clamp)
        if delta_gain != 1.0:
            delta = (delta * np.float32(delta_gain)).astype(np.float32)
        q_new, flips = project_one_step(q_old, scales, delta, rec.bits or 0, tensor_fraction, min_score)
        packed = patcher.pack_lowbit(q_new, rec.bits or 0)
        old_bytes = bytes(data[rec.data_pos : rec.data_pos + rec.data_len])
        byte_diff = sum(a != b for a, b in zip(old_bytes, packed))
        data[rec.data_pos : rec.data_pos + rec.data_len] = packed
        stats.tensors += 1
        stats.values += int(q_old.size)
        stats.flips += flips
        stats.byte_diff += byte_diff
        stats.byte_total += rec.data_len
        if stats.tensors <= 8 or stats.tensors % 30 == 0:
            print(
                f"[lm] layer={layer:02d} tag={tag:<18} bits={rec.bits} "
                f"flips={flips}/{q_old.size} byte_diff={byte_diff}/{rec.data_len}"
            )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(data)
    return stats


def patch_bundle(
    input_litertlm: Path,
    output_litertlm: Path,
    audio_adapter: Path,
    audio_encoder: Path,
    prefill_decode: Path,
) -> None:
    replacements = {
        "tf_lite_audio_adapter": audio_adapter,
        "tf_lite_audio_encoder_hw": audio_encoder,
        "tf_lite_prefill_decode": prefill_decode,
    }
    sections = patcher.litertlm_sections(input_litertlm)
    output_litertlm.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(input_litertlm, output_litertlm)
    with output_litertlm.open("r+b") as f:
        for name, repl in replacements.items():
            begin, end = sections[name]
            expected = end - begin
            actual = repl.stat().st_size
            if actual != expected:
                raise ValueError(f"{name} size mismatch: {actual} != {expected}")
            f.seek(begin)
            with repl.open("rb") as src:
                shutil.copyfileobj(src, f, length=16 * 1024 * 1024)
            print(f"[bundle] replaced {name}: {expected/1024/1024:.1f} MB")


def fmt_fraction(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-litertlm", type=Path, default=DEFAULT_LITERTLM)
    ap.add_argument("--sections-dir", type=Path, default=DEFAULT_SECTIONS)
    ap.add_argument("--lora-dir", type=Path, default=DEFAULT_LORA_DIR)
    ap.add_argument("--audio-frac", type=float, default=0.001)
    ap.add_argument("--lm-frac", type=float, default=0.001)
    ap.add_argument(
        "--lm-extra-frac",
        type=float,
        help="Optional projection fraction for --lm-extra-* selected tensors.",
    )
    ap.add_argument("--min-score", type=float, default=0.0)
    ap.add_argument("--audio-layer-start", type=int)
    ap.add_argument("--audio-layer-end", type=int)
    ap.add_argument(
        "--audio-tags",
        default="",
        help="Comma-separated audio modules to include, e.g. q_proj,k_proj,v_proj,post.",
    )
    ap.add_argument("--lm-layer-start", type=int)
    ap.add_argument("--lm-layer-end", type=int)
    ap.add_argument(
        "--lm-tags",
        default="",
        help="Comma-separated LM tags to include, e.g. q_einsum,k_einsum,v_einsum,attn_vec_einsum.",
    )
    ap.add_argument(
        "--lm-exclude-tags",
        default="",
        help="Comma-separated LM tags to exclude after --lm-tags/layer filters.",
    )
    ap.add_argument("--lm-extra-layer-start", type=int)
    ap.add_argument("--lm-extra-layer-end", type=int)
    ap.add_argument(
        "--lm-extra-tags",
        default="",
        help="Optional second LM layer/tag range, e.g. add upper attention to L00-14.",
    )
    ap.add_argument(
        "--lm-fold-mode",
        choices=("none", "global", "row", "rowcol"),
        default="none",
        help="Transform LM LoRA delta into official prefill_decode coordinates before q projection.",
    )
    ap.add_argument(
        "--lm-extra-fold-mode",
        choices=("inherit", "none", "global", "row", "rowcol"),
        default="inherit",
        help="Optional fold mode override for --lm-extra-* selected tensors.",
    )
    ap.add_argument("--lm-fold-override-layer-start", type=int)
    ap.add_argument("--lm-fold-override-layer-end", type=int)
    ap.add_argument(
        "--lm-fold-override-tags",
        default="",
        help="Selected LM tags whose fold mode should be overridden inside the override layer range.",
    )
    ap.add_argument(
        "--lm-fold-override-mode",
        choices=("none", "global", "row", "rowcol"),
        help="Fold mode applied to already-selected tensors matching the override layer/tag filter.",
    )
    ap.add_argument(
        "--lm-fold-override-spec",
        default="",
        help=(
            "Semicolon-separated override rules start:end:tag1,tag2:mode. "
            "Example: 0:14:v_einsum:none;21:29:q_einsum:none"
        ),
    )
    ap.add_argument("--lm-fit-merged-dir", type=Path, default=DEFAULT_FIT_MERGED_DIR)
    ap.add_argument(
        "--lm-fit-lora-dir",
        type=Path,
        help="LoRA dir used to estimate HF base for fold fitting. Defaults to --lora-dir.",
    )
    ap.add_argument("--lm-fold-iterations", type=int, default=8)
    ap.add_argument("--lm-fold-clamp", type=float, default=8.0)
    ap.add_argument("--lm-delta-gain", type=float, default=1.0)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    audio_tags = parse_csv_filter(args.audio_tags, set(patcher.AUDIO_MODULE_MAP), "audio tags")
    lm_tags = parse_csv_filter(args.lm_tags, set(LM_MODULE_MAP), "LM tags")
    lm_exclude_tags = parse_csv_filter(args.lm_exclude_tags, set(LM_MODULE_MAP), "LM exclude tags")
    lm_extra_tags = parse_csv_filter(args.lm_extra_tags, set(LM_MODULE_MAP), "LM extra tags")
    lm_override_tags = parse_csv_filter(
        args.lm_fold_override_tags,
        set(LM_MODULE_MAP),
        "LM fold override tags",
    )
    lm_override_rules = parse_fold_override_specs(args.lm_fold_override_spec)

    lora_path = patcher.resolve_lora_file(args.lora_dir)
    lora_scale = patcher.read_lora_scale(args.lora_dir)
    reader = patcher.SafeTensorReader(lora_path)
    fit_merged_reader = None
    fit_lora_reader = None
    fit_lora_scale = 1.0
    fit_lora_dir = args.lm_fit_lora_dir or args.lora_dir
    extra_fold_mode = None if args.lm_extra_fold_mode == "inherit" else args.lm_extra_fold_mode
    needs_fold = (
        args.lm_fold_mode != "none"
        or (extra_fold_mode is not None and extra_fold_mode != "none")
        or (args.lm_fold_override_mode is not None and args.lm_fold_override_mode != "none")
        or any(rule.mode != "none" for rule in lm_override_rules)
    )
    if needs_fold:
        fit_model_path = args.lm_fit_merged_dir / "model.safetensors"
        if not fit_model_path.exists():
            raise FileNotFoundError(f"LM fold fit merged model not found: {fit_model_path}")
        fit_lora_path = patcher.resolve_lora_file(fit_lora_dir)
        fit_merged_reader = patcher.SafeTensorReader(fit_model_path)
        fit_lora_reader = patcher.SafeTensorReader(fit_lora_path)
        fit_lora_scale = patcher.read_lora_scale(fit_lora_dir)
    tag = args.tag or f"qproj-af{fmt_fraction(args.audio_frac)}-lf{fmt_fraction(args.lm_frac)}"

    print(f"[setup] lora={lora_path} scale={lora_scale:g}")
    print(f"[setup] audio_frac={args.audio_frac:g} lm_frac={args.lm_frac:g} min_score={args.min_score:g}")
    print(
        "[setup] filters "
        f"audio_layers={args.audio_layer_start}..{args.audio_layer_end} "
        f"audio_tags={sorted(audio_tags) if audio_tags else '*'} "
        f"lm_layers={args.lm_layer_start}..{args.lm_layer_end} "
        f"lm_frac={args.lm_frac:g} "
        f"lm_tags={sorted(lm_tags) if lm_tags else '*'} "
        f"lm_extra_layers={args.lm_extra_layer_start}..{args.lm_extra_layer_end} "
        f"lm_extra_frac={args.lm_extra_frac if args.lm_extra_frac is not None else 'inherit'} "
        f"lm_extra_tags={sorted(lm_extra_tags) if lm_extra_tags else '-'} "
        f"lm_exclude={sorted(lm_exclude_tags) if lm_exclude_tags else '-'} "
        f"lm_fold={args.lm_fold_mode} "
        f"lm_extra_fold={args.lm_extra_fold_mode} "
        f"lm_fold_override={args.lm_fold_override_layer_start}..{args.lm_fold_override_layer_end}:"
        f"{sorted(lm_override_tags) if lm_override_tags else '*'}:"
        f"{args.lm_fold_override_mode or '-'} "
        f"lm_fold_override_spec={args.lm_fold_override_spec or '-'}"
    )
    if needs_fold:
        print(
            "[setup] fold fit "
            f"merged={args.lm_fit_merged_dir} lora={fit_lora_dir} "
            f"iterations={args.lm_fold_iterations} clamp={args.lm_fold_clamp:g} "
            f"delta_gain={args.lm_delta_gain:g}"
        )

    adapter_in = args.sections_dir / "audio_adapter.tflite"
    encoder_in = args.sections_dir / "audio_encoder_hw.tflite"
    prefill_in = args.sections_dir / "prefill_decode.tflite"
    adapter_out = args.out_dir / f"audio_adapter.{tag}.tflite"
    encoder_out = args.out_dir / f"audio_encoder_hw.{tag}.tflite"
    prefill_out = args.out_dir / f"prefill_decode.{tag}.tflite"
    bundle_out = args.out_dir / f"gemma-4-E2B-it.{tag}.litertlm"

    data_a, model_a, subgraph_a = patcher.load_tflite(adapter_in)
    adapter_diff = patch_audio_adapter(data_a, model_a, subgraph_a, reader, lora_scale)
    adapter_out.write_bytes(data_a)
    print(f"[adapter] byte_diff={adapter_diff}/{adapter_in.stat().st_size}")

    audio_stats = patch_audio_encoder(
        encoder_in,
        encoder_out,
        reader,
        lora_scale,
        args.audio_frac,
        args.min_score,
        args.audio_layer_start,
        args.audio_layer_end,
        audio_tags,
    )
    print(
        f"[audio summary] tensors={audio_stats.tensors} flips={audio_stats.flips}/{audio_stats.values} "
        f"byte_diff={audio_stats.byte_diff}/{audio_stats.byte_total}"
    )

    lm_stats = patch_lm_prefill(
        prefill_in,
        prefill_out,
        reader,
        lora_scale,
        args.lm_frac,
        args.lm_extra_frac,
        args.min_score,
        args.lm_layer_start,
        args.lm_layer_end,
        lm_tags,
        lm_exclude_tags,
        args.lm_extra_layer_start,
        args.lm_extra_layer_end,
        lm_extra_tags,
        args.lm_fold_mode,
        extra_fold_mode,
        args.lm_fold_override_layer_start,
        args.lm_fold_override_layer_end,
        lm_override_tags,
        args.lm_fold_override_mode,
        lm_override_rules,
        fit_merged_reader,
        fit_lora_reader,
        fit_lora_scale,
        args.lm_fold_iterations,
        args.lm_fold_clamp,
        args.lm_delta_gain,
    )
    print(
        f"[lm summary] tensors={lm_stats.tensors} flips={lm_stats.flips}/{lm_stats.values} "
        f"byte_diff={lm_stats.byte_diff}/{lm_stats.byte_total}"
    )

    patch_bundle(args.input_litertlm, bundle_out, adapter_out, encoder_out, prefill_out)
    print(f"[wrote] {bundle_out} ({bundle_out.stat().st_size/1024/1024:.1f} MB)")


if __name__ == "__main__":
    main()
