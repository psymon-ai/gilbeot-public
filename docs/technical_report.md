<div align="right">

[EN](technical_report.md) · [KO](technical_report_ko.md)

</div>

# Gilbeot — Technical Report

This document is the long-form companion to **§5 Engineering Challenges** of the [Kaggle writeup](../KAGGLE_WRITEUP.md).

---

## Table of contents

1. [The product loop](#1-the-product-loop)
2. [Audio — Korean elderly speech LoRA](#2-audio--korean-elderly-speech-lora)
3. [Deployment — LoRA into `.litertlm`](#3-deployment--lora-into-litertlm)
4. [Cactus, as a parallel investigation](#4-cactus-as-a-parallel-investigation)
5. [Vision — left/right safety](#5-vision--leftright-safety)
6. [Latency — backend policy per device](#6-latency--backend-policy-per-device)
7. [Demo APK — substitutions and honesty](#7-demo-apk--substitutions-and-honesty)
8. [References](#8-references)

---

## 1. The product loop

The hard part of Gilbeot was not any single model score. **Korean audio → mobile deployment → safe visual guidance → real-device latency** had to work as one loop on an elderly user's phone. Each piece was hard enough on its own, and none of them could be solved in isolation:

- Better audio is useless if it cannot land on Android.
- A good `.litertlm` patcher is useless if the visual guidance flips left and right.
- Safe guidance is useless if a sentence takes several minutes on the user's phone.

So the rest of this document is organized as the four loop stages plus the demo-APK boundary that is forced by international judging.

---

## 2. Audio — Korean elderly speech LoRA

### 2.1 Why fine-tuning was necessary

Speech recognition is the first door into Gilbeot. If the destination is misheard, every step of the route is wrong. Korean place and facility names, plus the slow speed and irregular pauses common in elderly Korean speech, were not stable on base Gemma 4 audio alone.

The internal evaluation used a 134-utterance held-out set, drawn from destinations and short walking-intent utterances. The metric is CER (Character Error Rate).

| Path | CER | Verdict |
|---|---:|---|
| Base `gemma-4-E2B-it-litert-lm` (no LoRA) | 13.14 % | Not usable for destination input |
| HF merged reference | 3.619 % | Cloud reference |
| **Gilbeot deployed LiteRT-LM path** | **5.00 %** | **Production choice** |

5.00 % is the level at which destinations like "송파구 보건소" (Songpa Health Center), "강남역" (Gangnam Station), and "국립중앙박물관" (National Museum of Korea) become reliable in the product flow. The ~1.4 pp gap between the cloud reference (3.619 %) and the on-device bundle (5.00 %) is the price of an unsupported `.litertlm` audio-LoRA export path; see §3 for what that gap actually costs.

### 2.2 Dataset

The training set was about 45,823 utterances — roughly 50 hours of audio — drawn from AI Hub's [**Command Speech (Elderly Male/Female)**](https://www.aihub.or.kr/aihubdata/data/view.do?aihubDataSe=data&dataSetSn=94) dataset, plus light in-house augmentation. The split is speaker-stratified:

| Split | Utterances |
|---|---:|
| Train | 43,431 |
| Eval | 2,291 |
| Held-out (CER reporting) | 134 |

Because Gilbeot is not a general dictation app but a *destination input* app, the curated subset focuses on imperative / request-form utterances closest to "the first sentence of a walking instruction," and the transcripts are normalized identically for evaluation and training.

### 2.3 Training stack and the silent-skip trap

| Component | Choice |
|---|---|
| Base model | Gemma 4 E2B (multimodal) |
| LoRA rank | 8 |
| LoRA targets | audio encoder + LM attention/MLP |
| Quantization | bitsandbytes 4-bit (NF4) |
| Adapter framework | PEFT |
| Training driver | **Unsloth FastModel** |
| Epochs | 2 |

The Unsloth FastModel detail is load-bearing. Gemma 4's audio side contains custom layers such as `Gemma4ClippableLinear`, and the vanilla PEFT path silently skips some of them — on the surface, loss goes down, but the audio output does not change. We lost about a day to this before instrumenting the trainable-parameter list and confirming the skip.

The Unsloth path handles those layers correctly. Once it did, the unmerged HF checkpoint reached **3.06 % CER**, a >4× error reduction over the 13.14 % device base; merging the adapter into the base for cloud inference loses a little precision, giving the 3.619 % cloud reference shown above.

---

## 3. Deployment — LoRA into `.litertlm`

### 3.1 Public toolchain in May 2026

At submission time we audited the publicly available paths for landing a Gemma 4 audio LoRA inside an Android `.litertlm` bundle:

| Path | Status |
|---|---|
| Official LiteRT-LM converter | Text-side LoRA only; audio path raised unsupported-shape errors |
| `litert-torch-nightly` | LM export works; audio encoder path does not |
| MediaPipe LLM Inference | Public LoRA path is attention-only and GPU-runtime-centric |
| Simple HF adapter deployment | No public way to plug it into the Android `.litertlm` runtime |
| Cactus | Working `model_gemma4_audio.cpp`, but converter expects a *pre-merged* HF checkpoint, and at the time the prebuilt Android binary was x86-Linux-compiled only |

We verified each by reading the source. None of them was a viable straight-through path. So we built a byte-level `.litertlm` patcher.

### 3.2 Three patcher attempts

The patcher lives at `tools/patch_gemma4_lora_requant.py` (with two companions); the README is at [`tools/README.md`](../tools/README.md). Three patcher designs were tried before one worked.

| Attempt | File | Method | Outcome |
|---|---|---|---|
| 1. Fresh requant | `patch_gemma4_lm_fresh.py` (historical) | Add LoRA delta, then re-quantize to int4 with a freshly computed scale | No measurable effect on output — small deltas get rounded away by the new scale; surviving deltas land in random places |
| 2. Quant-projected flip | `patch_gemma4_quant_projected.py` | Keep the existing int4 grid; flip the K largest \|delta\|/scale entries by ±1 LSB toward the LoRA direction | Partial improvement; the flip fraction sweet spot differs per layer and is unstable |
| 3. **Same-grid requant** | `patch_gemma4_lora_requant.py` | Keep the per-row scale; dequant → add delta → requant using the **same** scale | **Production choice** |

### 3.3 Same-grid requant — why it works

The key idea of the final method is to **not** create a new scale. The runtime kernel was compiled against the original quantization grid; if we recompute the scale we shift the rounding boundaries, and the LoRA delta's signal is the first thing to die. So we keep the boundaries fixed and let the LoRA direction either bump a coefficient to the next integer step (because its contribution exceeded half an LSB) or not. Either outcome is in the LoRA direction by construction.

```text
For each weight block matched between the HF base and the .litertlm:
  scale, ints   ← unpack_int4(litertlm_block)
  base_fp32     ← dequantize(scale, ints)
  patched_fp32  ← base_fp32 + (lora_A @ lora_B) * (alpha / rank)
  patched_ints  ← round(patched_fp32 / scale)        # same scale!
  patched_ints  ← clip(patched_ints, -8, 7)          # int4 range
  litertlm_block.replace(pack_int4(scale, patched_ints))
```

The patcher parses the `.litertlm` FlatBuffers container, locates audio-encoder and LM weight blocks, dequantizes from Google's INT2/INT4 packing, applies the LoRA delta, re-quantizes on the same grid, and rewrites same-size TFLite sections in place.

### 3.4 Graft + sidecar split

| Part | Method | Why |
|---|---|---|
| LM side | Graft-patch the LM weights inside `.litertlm` | Stable across our 134-utt eval; survives Adreno 740 prefill |
| Audio side | Attach an alpha-8 LoRA sidecar via the runtime `loraPath` | Touching the audio encoder weights inside `.litertlm` destabilized the Adreno 740 prefill kernel (int4 codebook mismatch against the compiled-in xnnpack kernel) |

The sidecar is loaded at runtime via `flutter_gemma`'s `loraPath`, which required Patch 4 in the fork (see [`patches/README.md`](../patches/README.md)) — upstream `flutter_gemma` 0.14.5 throws on `loraPath` for the FFI `.litertlm` path because the public C API has no setter for the LiteRT-LM internal `SessionConfig::SetScopedLoraFile` hook. We added an app-side shim (`libgilbeot_litertlm_lora.so`) that `dlsym`s the internal C++ symbol and attaches the sidecar to the opaque `SessionConfig` before the `ConversationConfig` snapshots it.

### 3.5 Cache-invalidation gotcha

This was the most demoralizing lost day of the project.

LiteRT-LM caches compiled kernels under `/data/data/<pkg>/app_files/...` on first model load (`xnnpack/` and `mldrift/` subdirs). **The cache is keyed by the model file name, not by content hash.** Push a patched `.litertlm` with the same filename and the next launch silently re-uses the cached kernels compiled against the *old* weights. Every byte-level patch we tried looked like it had no effect — including patches that were definitely correct in isolation — and we very nearly concluded the whole patcher direction was unworkable.

The fix is one ADB line:

```bash
adb shell run-as com.psymon.gilbeot.real \
    rm -rf app_files/xnnpack app_files/mldrift
```

Every device CER eval since then starts with this command.

---

## 4. Cactus, as a parallel investigation

Cactus is a different path that does not go through `.litertlm`. Where our patcher keeps the LiteRT-LM runtime in place and patches the weights, Cactus brings its own mobile inference engine and re-converts Gemma 4 into its own format. Same problem ("run Gemma 4 audio on Android"); two different layers of attack.

### 4.1 Byte-fallback detokenization bug (PR #635)

While running the comparison we discovered that Cactus was leaking literal byte-token strings into the transcript. Gemma 4 emits SentencePiece byte-fallback tokens like `<0xEA><0xB9><0xB0>` whenever a codepoint is not a single piece — which in Korean is most of the time. Cactus was passing those through as literal strings instead of re-assembling them as UTF-8. So a sentence like "잠 깰 수 있는" was surfacing as "잠 `<0xEA><0xB9><0xB0>` 수 있는".

A one-line detokenization fix moved the CER on our 134-sample Korean eval set as follows:

| Backend | CER before fix | CER after fix |
|---|---:|---:|
| Cactus INT8 | 4.638 % | 3.949 % |
| Cactus INT4 | 5.545 % | 4.856 % |

The fix is upstreamed at [cactus-compute/cactus#635](https://github.com/cactus-compute/cactus/pull/635). The accuracy numbers used in §4.2 are post-fix.

### 4.2 Accuracy vs latency

Latency numbers below are **compiler-board anchors**, not S10e mic-flow timings. All four backends are run under the same `scripts/gemma4_edge_compiler_board.py` harness on the 134-sample Korean held-out set, so the per-sample seconds include board overhead (per-sample reload, fixture I/O, eval driver) on top of pure inference. The point of this table is the *relative gap* between backends — the absolute seconds are not what a user perceives in the live mic flow.

| Backend | CER | Board median/sample | Verdict |
|---|---:|---:|---|
| HF merged reference | 3.619 % | 0.691 s | Cloud reference |
| LiteRT graft + alpha-8 sidecar | 5.003 % | 12.48 s | Production choice |
| Cactus INT8 | 3.949 % | 41.94 s | Better accuracy, ~3.4× slower under the same harness |
| Cactus INT4 | 4.856 % | 45.32 s | Smaller, even slower |

Cactus's accuracy was attractive. The blocker was the *relative* latency. On the same board harness, Cactus took ~3.4× as long per sample as the LiteRT path. Translating that ratio to a 2019 S10e mic flow on Gemma 4 E2B audio (102 MB encoder + 781 MB LM prefill/decode), whatever the LiteRT path was costing in seconds, Cactus would multiply it well past the window an elderly user can patiently wait between speaking a destination and hearing the first instruction. So the production app stays on the LiteRT-LM `.litertlm` + LoRA sidecar path.

This conclusion is bounded by the two devices we verified — S10e (2019) and S23 (2023). On hardware with more compute, for example a recent flagship or a device with strong NPU acceleration, Cactus's accuracy advantage would carry over while the latency concern would shrink; in that case Cactus is still a good choice.

---

## 5. Vision — left/right safety

### 5.1 The documented VLM weakness

Gilbeot's photo guidance is not "describe the picture" but "tell me where to go right now." The model has to safely read the actual cues in front of the user, not produce a pretty caption.

The problem is that VLMs are weak at basic spatial reasoning, especially left/right. This is a general weakness, not just a small-model issue. Kamath, Hessel, and Chang's EMNLP 2023 paper [*What's "up" with vision-language models? Investigating their struggle with spatial reasoning*](https://aclanthology.org/2023.emnlp-main.568/) evaluated 18 VLMs and reports that all of them were near 50 % chance on left/right pairs vs ~99 % human. A more direct piece of evidence is Hoehing, Rushe, and Ventresque's [*What's left can't be right — The remaining positional incompetence of contrastive vision-language models*](https://arxiv.org/abs/2311.11477), which focuses on left-right in CLIP-family models and shows the failure pattern is predictable even at large data scale.

Gemma 4 read exit numbers and signs well, but occasionally flipped "left" and "right" for horizontal arrows. In a walking-guidance product for elderly users, that is not a small error.

Chain-of-thought prompting did **not** fix this. Small VLMs cannot carry CoT through internal hidden state when the JSON schema only has room for one short instruction string.

### 5.2 bbox-style output

The fix exploits Gemma 4's native bbox output. We added two optional fields to the response schema:

```jsonc
{
  ...,
  "arrow_tip_x": null,   // 0=left edge, 1=right edge of image
  "arrow_tail_x": null,  // null if no horizontal arrow visible
  "instruction": "...",
  ...
}
```

The model only fills them when a horizontal arrow is visible. The Dart code then compares the two x coordinates and **overwrites left/right wording in the instruction** based on pixel arithmetic:

```dart
final pixelDir = tipX < tailX ? 'left' : 'right';
if (RegExp(r'\b' + opposite(pixelDir) + r'\b').hasMatch(instruction)) {
  instruction = instruction.replaceAll(opposite(pixelDir), pixelDir);
}
```

This is **not** the same as writing the answer into the prompt. The model is asked to *localize* the arrow — a strength of vision encoders — and a deterministic block of code makes the left/right decision. The model is unaware of which side our code considers "left"; the coordinates are just numbers.

Per-photo cost: about +0.3 s prefill (two extra field declarations); about 0 s decode (only filled when an arrow is present).

### 5.3 The word-frequency prior bug

While polishing the bbox path we hit a *consistent* "to your right" output that was not random confusion — it was a one-directional bias. Empirical probe: removing the single word `right` from the schema example's instruction (`"I can see the exit sign right ahead..."` → `"...directly ahead..."`) flipped the output from consistent "right" to consistent *hedging* ("follow the indicated direction") with no direction word at all.

Small VLMs treat their own example outputs in the prompt as a strong vocabulary prior. The model was lifting `right` from "right ahead" (adverb) into directional output. Fix: remove every gratuitous "right"/"left" from the prompt, then add *balanced* LEFT/RIGHT example phrasings:

```text
Example phrasings for arrow photos (pick ONE direction word):
- "The arrow points to the LEFT — please walk to the left..."
- "The arrow points to the RIGHT — please walk to the right..."
```

Plus an explicit rule: *"If a horizontal arrow is visible, your instruction include either 'left' or 'right' — the app overrides this from arrow_tip_x/arrow_tail_x pixel coords if your word choice disagrees with the pixels."*

### 5.4 The mirror experiment

As a final check, we horizontally mirrored test photo (EXIF-aware so the rendered scene genuinely flips) so the arrow now pointed right in the displayed image, then re-ran the demo flow:

```text
Photo (mirrored):
  arrow_tip_x: 0.75, arrow_tail_x: 0.25,
  instruction: "I can see a yellow sign with an arrow pointing
                to the right. Please walk to the right..."
```

Coordinates moved from (0.15, 0.45) on the original to (0.75, 0.25) on the mirror. The direction word changed too. The model is actually looking at the image, not echoing a constant.

### 5.5 Observed regimes on S23

| Coords returned? | Model wording | Final spoken instruction |
|---|---|---|
| no | "...the arrow points to the left..." | unchanged (correct) |
| yes, model said "right" | "...follow the arrow to your right..." | **overridden → "...to your left..."** |
| yes, model said "left" | "...follow the arrow to your left..." | unchanged |

The override (`home_screen._applyArrowBboxOverride`) is wired into both the demo and the Korean production builds. Total addition: about 25 lines of Dart plus three field declarations and the balanced example phrasings across two prompt templates.

---

## 6. Latency — backend policy per device

Speed is not a convenience feature here. It is accessibility. If a user has to wait too long after taking a photo on the road, an accurate model still cannot ship as a real product.

We deliberately did not set the latest flagship phone as the reference. Gilbeot's core users — elderly Koreans — are not the segment that upgrades every year; many keep one phone for several years after buying it. The hardware range had to match that reality, so the upper bound is the Galaxy S23 (2023) and the lower bound is the Galaxy S10e (2019).

### 6.1 MTP wiring (fork patches 1–3)

Gemma 4's MTP (Multi-Token Prediction) is a speculative-decoding scheme: a small drafter proposes several tokens up front, and the main model verifies them in parallel.

The shipped `libLiteRtLm.so` (native-v0.10.2-b) already exports `litert_lm_engine_settings_set_enable_speculative_decoding` and bundles the MTP runtime. `flutter_gemma` 0.14.5 simply never wired it. Three small fork edits (purely additive — see [`patches/README.md`](../patches/README.md)) declare the symbol in the header, add the FFI binding, and call it from `initialize()` with a backend-aware default:

```dart
bool? enableSpeculativeDecoding,   // null → auto by backend
...
final enableSpec = enableSpeculativeDecoding ?? (backend != 'cpu');
b.litert_lm_engine_settings_set_enable_speculative_decoding(
    settings, enableSpec);
```

No native bump was needed — the 0.10.2-b prebuilt is C-API-symbol identical to 0.11.0-b for this surface.

### 6.2 S23 GPU — MTP wins (~1.5×), and why not more

A/B on the S23, demo-normal flow, same 13-rule prompt + 1280 px input:

| | baseline | MTP on |
|---|---:|---:|
| prefill | ~4.9 s | ~5.06–6.53 s |
| decode | ~17.8 s | ~11.34–11.80 s |
| chunks | ~128 | ~55–65 |
| total | ~22 s | ~16.4–18.3 s |

Decode 17.8 s → ~11.5 s is **1.5–1.6×**. Chunk count halves — MTP is batching roughly two tokens per chunk. Quality is preserved (speculative decoding is lossless by construction).

Google's published Gemma 4 MTP target on the base model is around **2.87×**, and vLLM's QLoRA + MTP measurement reports about **92 % of the stock speedup retained** when a LoRA is applied. So a fine-tuned model on the right setup should not drop this far. We landed at roughly **half** of the base-only acceptance rate, and the gap was worth investigating rather than waving away.

A 4-way controlled measurement on S23 GPU + MTP (each row averaged over the same demo flow's vision photos; acceptance length back-calculated from `chunks/sec ÷ sequential baseline ~5.6 tok/sec`):

| Setup | Model file | Sidecar | Vision chunks/sec | Avg total | Implied acceptance |
|---|---|---|---:|---:|---|
| Base only | `gemma-4-E2B-it.litertlm` | — | 14.4 | 8.9 s | ~2.6 tok/step |
| Graft only (sidecar removed) | graft `.litertlm` | — | 7.0 | 14.2 s | ~1.25 tok/step |
| Graft + sidecar (production) | graft | alpha 8 | 7.4 | 13.2 s | ~1.3 tok/step |
| Base + sidecar | base | alpha 8 | 7.4 | 13.8 s | ~1.3 tok/step |

The flatness across the last three rows is the finding: **the moment any LoRA path is active — baked-in graft, runtime sidecar, or both — chunks/sec drops to roughly half of base-only, and the specific path barely matters**. The LoRA's *presence* is what moves the needle, not how it is attached.

Our first framing of this in earlier notes was "the `tf_lite_mtp_drafter` has no LoRA, so its predictions diverge from the LoRA'd main LM." That is an oversimplification we want to correct. Gemma 4's drafter is not a fully separate model: it shares the input embedding table and KV cache with the main LM, and crucially it *conditions* on the main LM's final-layer activations (drafter input ≈ token embedding + main LM final hidden state → down-projection → drafter's own transformer layers). The drafter's own ~42 MB of weights are untuned, but the LoRA's effect on activations does reach the drafter through that conditioning channel — which is exactly why vLLM observes 92 % retention rather than a collapse.

So the real question is not "the drafter has no LoRA" but "why does our specific LoRA configuration push acceptance from ~2.6 to ~1.3 per step when vLLM's typical configurations don't." We have four candidate hypotheses, none directly verified:

1. **LoRA magnitude and coverage.** Our sidecar is a full-LM `atten_mlp_alpha8` adapter — every attention layer, every MLP layer, alpha 8. Typical PEFT QLoRA cases are narrower and lower-alpha. A larger activation drift pushes the drafter's conditioning input further outside the range it saw during pretraining.
2. **Cross-task generalization.** The LoRA was trained for Korean audio transcription, but this measurement is on a vision photo benchmark generating English route instructions. The drafter sees a hidden state that carries a Korean-audio LoRA's distributional signal applied to an English vision prompt — a combination neither the drafter's pretraining nor the LoRA training covered.
3. **Patcher × runtime conditioning interaction.** Our `patch_gemma4_lora_requant.py` patches `prefill_decode` and `audio_encoder_hw`. We have not verified that the LiteRT-LM `SetScopedLoraFile` runtime path interacts with the drafter conditioning the same way Google's (unpublished) audio-LoRA exporter would. The patcher's INT4 re-quantization on the same grid is itself an approximation; cumulative numerical drift in the conditioning channel could cost extra acceptance.
4. **Graft baseline gap.** chunks/sec was not measured during the graft selection sweep — we optimized CER. Other graft candidates may behave differently, and there is no guarantee that the current `...-qat-g11875-v2sidecar` graft is the only point worth exploring on the LoRA × MTP trade-off surface.

A few adjacent measurements that would help narrow the cause but which we did not run: per-step acceptance directly (LiteRT-LM's INFO logs don't expose accept/reject events), lower-alpha sidecars (alpha 4, alpha 2), drafter-side LoRA attachment, audio-only vs LM-only LoRA isolation on chunks/sec, and an apples-to-apples comparison against vLLM on the same model.

So the 1.5–1.6× on S23 is what we shipped — not because it is the structural ceiling, but because it is what our current LoRA configuration on this specific deployment path yields. There is a real gap to vLLM's 92 % retention figure, and we'd like to keep working on it rather than declare it solved.

### 6.3 S10e CPU — MTP loses

The first time MTP ran on S10e CPU+XNNPACK, per-photo total was 50–70 s. A two-step A/B then halved the photo time:

```text
Baseline (MTP on, English DEMO prompt 3.74 KB):
  prefill avg 40.8 s + decode avg 21.9 s = 62.7 s / photo
  decode rate 2.7 chunks/sec

Step A — auto-disable MTP on CPU backend
  prefill avg 30.5 s + decode avg 10.0 s = 40.5 s / photo
  decode rate 9.5 chunks/sec
  Δ = -22.2 s / photo (-35 % total)

Step B — drop [Good example] / [Bad example] block from prompt
  prefill avg 21.4 s + decode avg 8.9 s = 30.2 s / photo
  decode rate 9.7 chunks/sec
  Cumulative Δ = -32.5 s (-52 %, 62.7 → 30.2)
```

Diagnosis: the MTP drafter's forward-pass per cycle on CPU is not offset by acceptance gain. On GPU the drafter is fast and accepted drafts skip target compute → net 1.5×. On CPU both costs land on the same arithmetic unit, and the LoRA-drafter mismatch keeps acceptance modest → net negative.

### 6.4 Final on-device numbers

| Device | Backend | MTP | Per-photo |
|---|---|---|---:|
| Galaxy S23 | GPU (Adreno 740) | on | ~12 s |
| Galaxy S10e | CPU (XNNPACK) | off (auto) | ~50 s |

Prefill scales with prompt size (English DEMO prompt is ~3.3 KB / ~900 tokens text + ~256 visual tokens at 768 px); decode scales with output length (typical 50–105 chunks per photo).

Policy: use GPU + MTP when possible; keep older phones slow but functional through the CPU-safe path.

---

## 7. Demo APK — substitutions and honesty

International judges may not have a Korean speaker, a Korean map API key, a Korean GPS location, or a Korean subway environment available at once. So a separate Demo APK is the only viable submission path.

The principle was: **the on-device Gemma 4 model still runs for real on every photo and audio input**. Only the Korean-territory inputs (map APIs, live GPS) are pre-baked.

| Build | Package id | Role |
|---|---|---|
| Judge Demo | `com.psymon.gilbeot.demo` | International judging |
| Korea Production | `com.psymon.gilbeot.real` | Real Korean users |

### 7.1 What is fixed vs what runs live

| Component | Judge mode | Korea production |
|---|---|---|
| Gemma 4 audio (STT) | **Real** (bundled WAV → live model) | Real (mic → live model) |
| Gemma 4 vision | **Real** (bundled photo → live model) | Real (camera → live model) |
| Gemma 4 text generation | **Real** | Real |
| Destination → place lookup | hardcoded (Songpa Health Center) | live T-Map POI API |
| Routing | cached T-Map polyline (asset) | live ODsay + T-Map API |
| Origin (current location) | per-photo EXIF GPS, stepwise | live Geolocator GPS |
| Off-route guard | EXIF distance vs polyline | live GPS Haversine |
| Arrival detection | model `is_arrival` + last-photo fallback | model + GPS Haversine ≤ 30 m |

The Korean production build (`com.psymon.gilbeot.real`) does not substitute either of these — it runs live end-to-end.

### 7.2 BYO photo flow with EXIF gate

A reasonable reviewer could ask whether the on-device Gemma 4 vision model is *actually* generalizing or whether the demo cherry-picks photos the model has seen during prompt engineering.

Long-pressing the camera button (DEMO_MODE only) opens the system gallery. The picked photo goes through the *same* on-device vision path as the canned demo — no special tuning, no asset overlap. The judge can prove the model isn't a four-photo lookup table by feeding it anything they have on the phone.

A randomly chosen photo would otherwise let the model invent directions toward the hardcoded Songpa destination ("walk left to exit 10") even if the photo was taken in another city. We extract EXIF GPS, compute minimum perpendicular distance to the planned T-Map polyline, and branch the per-photo step context:

| EXIF distance from route | Step context branch |
|---|---|
| ≤ 200 m | "on or near the route — short scene description + brief Exit 10 guidance" |
| > 200 m | "NOT on the route — required prefix sentence + ONE describing sentence + STOP. FORBIDDEN: `follow`, `walk toward`, `go to`, …" |
| missing | "generic — describe only, no destination directions" |

The off-route branch ships a neutral example JSON (`["bench", "tree", "grass"]` + "wooden bench under a leafy tree") chosen specifically so its vocabulary cannot leak into a subway/landmark photo's response — the model must do real vision, not copy the example.

---

## 8. References

- AI Hub — [Command Speech (Elderly Male/Female)](https://www.aihub.or.kr/aihubdata/data/view.do?aihubDataSe=data&dataSetSn=94)
- Kamath, Hessel, Chang. [*What's "up" with vision-language models? Investigating their struggle with spatial reasoning*](https://aclanthology.org/2023.emnlp-main.568/). EMNLP 2023.
- Hoehing, Rushe, Ventresque. [*What's left can't be right — The remaining positional incompetence of contrastive vision-language models*](https://arxiv.org/abs/2311.11477).
- Cactus byte-fallback detokenization fix: [cactus-compute/cactus#635](https://github.com/cactus-compute/cactus/pull/635)
- LiteRT-LM v0.11.0 release notes (Gemma 4 MTP support): https://github.com/google-ai-edge/LiteRT-LM/releases/tag/v0.11.0
- LiteRT-LM C++ API (speculative decoding on mobile): https://ai.google.dev/edge/litert-lm/cpp
- Korean version of this report: [`technical_report_ko.md`](technical_report_ko.md)
- In-repo documents referenced from this report: [`tools/README.md`](../tools/README.md), [`patches/README.md`](../patches/README.md), [`docs/architecture.md`](architecture.md).
