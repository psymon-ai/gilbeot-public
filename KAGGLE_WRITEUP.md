# Gilbeot — On-device multimodal AI walking assistant for elderly users

## 1. Motivation

For many elderly users, getting lost is not about lacking a map app. It happens when the app's words do not connect to the scene in front of them. Navigation says "turn right in 250 m," but an older user often understands the road as **visible objects and scenes**: "under the yellow sign," "past the pharmacy," or "the side with the stair handrail."

> **Gilbeot translates machine coordinates into human coordinates.**

Modern multimodal AI can understand speech, images, and text together, but it still rarely reaches the moment when an elderly person is outside and uncertain. Gilbeot puts Gemma 4 onto scenes older users already know.

## 2. Solution Approach

```text
voice destination → Gemma 4 audio → Korean transcript
                  → T-Map / ODsay route + Naver Map rendering
                  → user takes a photo at a confusing point
                  → Gemma 4 vision reads signs, exits, arrows, entrances
                  → natural instruction: "walk left below the yellow Exit 10 sign"
                  → GPS-route polyline Haversine off-route check
                  → if off-route, suppress guidance and ask for reroute/retake
                  → TTS speaks the instruction
```

Off-route handling is a safety mechanism: when GPS drifts from the planned polyline, the production build blocks guidance instead of letting the VLM hallucinate from an unrelated scene.

### Why On-device

This product is needed in subway stations, underground malls, and hospital areas, where networks can be unreliable and surroundings are complex. Voice and photos also reveal location and movement. Korea Gallup 2025 reports Galaxy use at 92% among people in their 60s and 82% among users 70+, making Android Galaxy the right first target. [Korea Gallup 2025](https://www.gallup.co.kr/dir/GallupReport/GallupReport%2820250707%29_%EC%8A%A4%EB%A7%88%ED%8A%B8%ED%8F%B0.pdf)

### Two APKs

Korean production requires Korean map APIs and live location. International judges may not have those. The Demo APK fixes the scenario, but Gemma 4 still runs for every audio and photo.

| Build | Package ID | Purpose |
|---|---|---|
| Judge Demo | `com.psymon.gilbeot.demo` | International judging |
| Korea Production | `com.psymon.gilbeot.real` | Real Korean users |

## 3. Development Process

We tested conservatively. Elderly users are not a segment that upgrades every year.

| Item | Galaxy S23 | Galaxy S10e |
|---|---:|---:|
| Release | 2023 | 2019 |
| Role | realistic upper bound | old-phone lower bound |
| SoC | Snapdragon 8 Gen 2 for Galaxy | Exynos 9820 |
| GPU | Adreno 740 | Mali-G76 MP12 |
| Main backend | GPU | CPU (XNNPACK) |
| MTP policy | on | off |

Architecture: [`docs/architecture.md`](https://github.com/psymon-ai/gilbeot-public/blob/main/docs/architecture.md).

## 4. Google AI Edge

Gemma 4 is not just an API here. It is the **sensory organ of the product**: hearing a destination, seeing the scene, and producing words the user can follow.

| Gemma 4 capability | Role in Gilbeot |
|---|---|
| Audio understanding | Korean destination speech → transcript |
| Vision understanding | signs, exits, arrows, entrances |
| Text generation | simple Korean walking instruction |
| Native bbox-style output | arrow tip/tail coordinates → left/right correction |
| LiteRT-LM deployment | Android inference without a server |
| MTP / speculative decoding | faster S23 GPU decode |

The bundle is ~2.4GB: vision encoder ~208MB, audio encoder ~102MB, LM prefill/decode ~781MB, per-layer embedder ~1.2GB, MTP drafter ~42MB. One model file handles speech, image, and text, so memory/backend policy mattered.

## 5. Engineering Challenges

The hard part was not one model score. **Korean audio → mobile deployment → safe visual guidance → real-device latency** had to work as one product loop. Better audio is useless if it cannot land on Android; good guidance is dangerous if left/right flips.

### 5.1 Audio: Elderly Korean Speech

If the destination is misheard, every route step is wrong. Base Gemma 4 audio scored **13.14% CER** on our 134-utterance Korean held-out set, not enough for destination input.

| Problem | Response | Result |
|---|---|---|
| slow speech and pauses | AI Hub *Command Speech (Elderly Male/Female)*, 45,823 utterances, ~50h | elderly command style |
| Gemma 4 audio custom layers | Unsloth FastModel + PEFT + 4-bit | `Gemma4ClippableLinear` trained |
| mobile deployment accuracy | rank-8 LoRA on audio encoder + LM attention/MLP | deployed CER **5.00%** |

[AI Hub](https://www.aihub.or.kr/aihubdata/data/view.do?aihubDataSe=data&dataSetSn=94)

| Path | CER |
|---|---:|
| Base `gemma-4-E2B-it-litert-lm` (no LoRA) | 13.14% |
| Gilbeot deployed LiteRT-LM path | **5.00%** |
| Improvement vs device base | ~2.6x reduction (-62%) |

That 5.00% is where phrases like "Songpa Health Center" or "Gangnam Station" become usable in the product flow.

### 5.2 Deployment: LoRA into `.litertlm`

Training was easier than deployment. There was no official path for putting Gemma 4 audio LoRA into an Android `.litertlm` bundle.

| Path | Blocker |
|---|---|
| LiteRT-LM converter | audio path shape unsupported |
| `litert-torch-nightly` | LM export only |
| MediaPipe LLM | attention-only / GPU-runtime centered |

We built a patcher. The key was **same-grid requant**: keep the original quant scale and move weights only within the runtime's expected grid.

```text
patched_int4 = clip(round((base_fp32 + LoRA_delta) / scale), -8, 7)

HF LoRA delta → same-grid requant → LM graft in .litertlm
audio adapter → alpha-8 sidecar  → runtime loraPath
```

The compromise was deliberate: graft stable parts into the bundle, and attach the risky audio adapter as a runtime sidecar. Patcher source: [`tools/README.md`](https://github.com/psymon-ai/gilbeot-public/blob/main/tools/README.md).

We also tested Cactus. It was accurate, but S10e latency exceeded the product limit.

| Backend | CER | S10e median/sample |
|---|---:|---:|
| LiteRT (graft+sidecar) | 5.003% | 12.48s |
| Cactus INT8 | 3.949% | 41.94s |
| Cactus INT4 | 4.856% | 45.32s |

The Cactus values are after our byte-fallback detokenization fix, submitted upstream: [cactus-compute/cactus#635](https://github.com/cactus-compute/cactus/pull/635). Cactus may fit stronger devices; for S10e/S23, LiteRT was deployable.

### 5.3 Vision: Do Not Trust Left/Right Words Alone

VLMs are weak at basic spatial relations, especially left/right ([EMNLP 2023](https://aclanthology.org/2023.emnlp-main.568/), [arXiv:2311.11477](https://arxiv.org/abs/2311.11477)). Gemma 4 read signs and exit numbers well, but sometimes flipped words for horizontal arrows.

The fix was bbox-style output. The model emits arrow tip/tail x-coordinates; deterministic code decides direction.

```text
arrow_tip_x < arrow_tail_x → left
arrow_tip_x > arrow_tail_x → right
```

The model localizes; code decides left/right. If the sentence conflicts with pixels, the app corrects the word. This is a safety layer, not a lack of trust in the model.

### 5.4 Latency: Backend Policy per Device

Speed is accessibility. On S23 GPU, MTP reduced decode from 17.8s to 11.5s, a 1.5-1.6x gain. On S10e CPU, MTP hurt because drafter and main model share compute, and the LoRA-tuned LM differs from the base drafter distribution.

| Device | Backend | MTP | Per photo |
|---|---|---|---:|
| Galaxy S23 | GPU | on | ~12s |
| Galaxy S10e | CPU | off | ~50s |

Policy: use GPU+MTP when possible; keep older phones slow but functional through the CPU-safe path.

### 5.5 Demo: Fixed Flow, Real Inference

Judges may not have Korean API keys, GPS, subway photos, or Korean speech. The Demo APK fixes some inputs, but the model still decides. The bundled Korean WAV is not just played back; Gemma 4 audio extracts the transcript and destination every run.

| Fixed | Actually executed |
|---|---|
| bundled Korean WAV | Gemma 4 audio STT + destination recognition |
| cached T-Map route | route starts only after transcript validation |
| demo photo sequence | Gemma 4 vision instruction generation |
| generated JSON | parsing, left/right correction, arrival detection |
| gallery long-press | arbitrary photo + EXIF off-route handling |

Long-pressing the camera opens the gallery; any photo goes through the same vision path, and an off-route EXIF GPS triggers description instead of guidance. The Korean production build has no such scaffolding.

## 6. Future Work

Navigation is the starting point. The real gap is between what AI can do and what elderly users can do on smartphones. Phone banking, public documents, hospital booking, medication, and welfare services involve private health or financial data, so they must be on-device. Gilbeot is one first interface for that future.

## 7. Thanks

To my father-in-law: thank you for joining product testing more eagerly than anyone and for stepping in front of the camera for the demo video. Through every stretch of building Gilbeot, your honest feedback and steady encouragement carried this project to the finish line. As you begin this new chapter after retirement, may every road ahead be filled with happiness.

## 8. License

Released under the Attribution 4.0 International (CC BY 4.0) license.

_Gemma is a trademark of Google LLC._

## 9. Citation

Psymon. Gilbeot — On-device multimodal AI walking assistant for elderly users. https://www.kaggle.com/competitions/gemma-4-good-hackathon/writeups/gilbeot. 2026. Kaggle.
