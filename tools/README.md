# tools/ — `.litertlm` LoRA patcher

This is the engineering-spike patcher we used to carry a Korean
audio LoRA (PEFT, fine-tuned with Unsloth on Gemma 4 E2B) into a
deployable `.litertlm` bundle, **because at submission time (May
2026) no public Gemma 4 audio LoRA → `.litertlm` exporter exists**.

It is not a polished converter. It works at the byte level: it
parses the `.litertlm` FlatBuffers container, locates audio-encoder
and LM weight blocks, dequantizes them from Google's INT2/INT4
packing, applies the LoRA delta, re-quantizes **on the same grid**
(naive `quant(base + delta)` shifts the rounding and destroys the
adapter — that was hard to find), and rewrites same-size TFLite
sections in place.

## Files

| File | Role |
|---|---|
| `patch_gemma4_lora_requant.py` | Main entry point. Build full-LoRA requantization candidates. Imports the other two. |
| `patch_gemma4_audio_lora.py` | Low-level engine. `.litertlm` parse, TFLite int2/int4 unpack/pack, encoder + adapter patch ops. |
| `patch_gemma4_quant_projected.py` | Alternative path — flip K largest `|delta|/scale` quantized entries by ±1 LSB toward the LoRA direction. Avoids re-quantization noise; used as a control. |

## Dependencies

- Python ≥ 3.10
- `numpy` (only numeric requirement for the patch path)
- LiteRT-LM schema wheels in `.tmp/` (auto-detected; download from
  the LiteRT-LM release).

PEFT / Transformers / Unsloth are only needed for the LoRA
*training* side, not for these patchers.

## Typical usage

```bash
# Build a candidate from a trained Unsloth audio LoRA + base .litertlm.
python tools/patch_gemma4_lora_requant.py \
    --base   models/gemma-4-E2B-it.gallery-7fa1d78.litertlm \
    --lora   outputs/gemma4_audio_v2/lora \
    --out    outputs/litertlm_patch/full-lm-audio.litertlm \
    --target lm-and-audio        # or  lm-attn  /  full-lm

# Eval CER on a held-out Korean audio set.
python tools/eval_litertlm_audio_patch.py \
    --litertlm outputs/litertlm_patch/full-lm-audio.litertlm \
    --eval     data/ko_audio_eval
```

(Paths above assume the dev workspace layout — the public repo
ships the patcher source, not the model bundle. Download the base
`.litertlm` from the Gemma 4 release on Hugging Face.)

## Status

✅ **What works** — produces an on-device deployable bundle. Our
best candidate (graft + alpha-8 audio sidecar) gives **CER 5.00 %**
on a 134-utterance Korean held-out set, a ~2.6× reduction over the
13.14 % device base (−62 % relative).

⚠ **Known gap** — the trained adapter reaches CER 3.06 % at HF
checkpoint level; the on-device deployment loses ~2 pp to the
unsupported quantization path. We expect the gap to close once
Google publishes the official audio exporter; until then, this
patcher exists.

🔜 **Not done** — fp16 audio encoder path (we only re-quantize
int2/int4 blocks today), automated regression suite, and a
self-contained CLI (`requant.py` still imports `audio_lora.py` and
`quant_projected.py` as siblings — keep them in the same directory).

## License

MIT — same as the rest of the repo.
