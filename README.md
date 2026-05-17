# 길벗 (Gilbeot) — On-device walking assistant for elderly users

> An on-device multimodal walking assistant for elderly Android users. The user speaks a Korean destination; Gemma 4 audio transcribes it and the app builds a route. On the road, photos go through Gemma 4 vision and the app speaks back a short, scene-specific instruction. Every core inference runs on the phone.

Submission for the [Kaggle Gemma 4 Good Hackathon][hackathon] (Digital Equity & Inclusivity track). Full write-up: [`KAGGLE_WRITEUP.md`](KAGGLE_WRITEUP.md) · 한국어판 [`KAGGLE_WRITEUP_KO.md`](KAGGLE_WRITEUP_KO.md) · README: 한국어 [`README_KO.md`](README_KO.md).

| | |
|---|---|
| **Models** | Gemma 4 E2B (audio + vision + text), on-device via LiteRT-LM |
| **Stack** | Flutter • dart:ffi • LiteRT-LM • flutter_gemma 0.14.5 (+ small local fork for MTP + LoRA sidecar) |
| **Korean STT** | LoRA sidecar bundled in APK — **13.14 % → 5.00 % CER** on a 134-utterance Korean held-out set (−62 %) |
| **Devices** | Galaxy S23 (GPU + MTP, ~12 s/photo) and S10e (CPU, MTP off, ~50 s/photo) verified |
| **Privacy** | Fully offline after first launch (model downloads once; LoRA + canned route bundled in APK) |

[hackathon]: https://www.kaggle.com/competitions/gemma-4-good-hackathon

---

## Try it — Judge Demo APK

The Korean production build needs Korean Map APIs (ODsay / T-Map / Naver) and a Korean speaker — neither is available to most international judges. The Judge Demo APK runs a pinned scenario but **still executes the on-device Gemma 4 model for real on every photo and every audio clip**.

1. **Install** [`gilbeot-judge-demo.apk`](https://github.com/psymon-ai/gilbeot-public/releases/download/v1.0.0/gilbeot-judge-demo.apk) (201 MB) from the [GitHub Releases page](https://github.com/psymon-ai/gilbeot-public/releases/tag/v1.0.0). Android 10+ required; sideload via `adb install` or enable "Install unknown apps" for your browser.
2. Open **Gilbeot Demo**. First launch downloads the ~2.4 GB Gemma 4 bundle from Hugging Face ([`psymon/gilbeot-korean-audio-litertlm`](https://huggingface.co/psymon/gilbeot-korean-audio-litertlm); ~3–5 minutes on Wi-Fi).
3. Tap the **microphone** once. A pre-recorded Korean utterance (*"송파구보건소 가야 해" — I need to go to Songpa Health Center*) plays audibly. The on-device Gemma 4 audio model transcribes it in real time and shows both Korean and English text.
4. Tap the **camera** button. A bundled subway-station photo opens full-screen with a round shutter at the bottom. Tap the shutter — the on-device Gemma 4 vision model emits a JSON instruction (`{ instruction, landmarks, is_arrival, arrow_tip_x, arrow_tail_x }`) and TTS reads the instruction aloud.
5. Repeat for photos 2 → 3 → 4. Photo 4 is the destination building; the model marks `is_arrival=true` and Gilbeot says goodbye.

The map button shows the cached T-Map walking polyline with origin markers that step through the journey.

What runs live in the Demo APK: Gemma 4 audio, vision, text generation, JSON parsing, arrow-based left/right correction, arrival recognition. What is canned: routing data (cached T-Map polyline), the destination utterance (recorded WAV), GPS (per-photo EXIF). The split is laid out in [Honest scaffolding](#honest-scaffolding-judge-mode) below.

## Demo video

📺 **[3-minute walkthrough on YouTube](https://www.youtube.com/watch?v=jg-FSf5QdNI)**

## Architecture (one breath)

```
voice (Korean WAV) ──► Gemma 4 audio ──► transcript
                                           │
                                           ▼
                            destination + canned route
                                           │
                                           ▼
photo (camera or bundled) ─► Gemma 4 vision (English systemPrompt)
                                           │
                                           ▼
JSON { instruction, landmarks, is_arrival,
       arrow_tip_x, arrow_tail_x } ─────► English TTS
        │
        └─► tip_x vs tail_x → deterministic left/right correction
            (bbox grounding; VLMs frequently flip the word "left/right")
```

Gemma 4 runs through `dart:ffi` → LiteRT-LM C API → `libLiteRtLm.so` (arm64). The release APK uses a forked `flutter_gemma` 0.14.5 — four inline additions: three to wire Gemma 4 MTP (the C symbol is already exported by the shipped `.so` but upstream never declares it), one to thread the LoRA sidecar through the FFI client (upstream rejects `loraPath` on the `.litertlm` path). See [`patches/README.md`](patches/README.md).

Speculative decoding (Gemma 4 MTP) is gated by backend:

| Backend | MTP | Reason |
|---|---|---|
| GPU (S23 Adreno 740) | on  | Measured 1.5–1.6× decode speedup, lossless. |
| CPU (S10e XNNPACK)   | off | Drafter overhead exceeds acceptance gain; the LoRA-shifted target distribution lowers acceptance further. Net ~10–15 s slowdown vs MTP-off per photo. |

Full diagram + per-component sizing: [`docs/architecture.md`](docs/architecture.md).

## Korean STT — what makes it work

The base Gemma 4 audio path returns **13.14 % CER** on our 134-utterance Korean held-out set (destinations + everyday phrases, elderly speakers). That is not usable for destination input. We trained a rank-8 LoRA on the audio encoder and the LM attention / MLP layers (Unsloth FastModel + PEFT + bitsandbytes 4-bit; ~45,823 training utterances / ~50 hours from AI Hub's *Command Speech (Elderly Male/Female)* dataset), then deployed it on Android through a same-grid requant patcher we built ourselves — no official Gemma 4 audio LoRA → `.litertlm` exporter exists at submission time.

| Path | CER | Note |
|---|---:|---|
| Base `gemma-4-E2B-it-litert-lm` (no LoRA) | 13.14 % | Device baseline |
| Gilbeot deployed (graft + alpha-8 sidecar) | **5.00 %** | Android deployment path |
| Improvement vs device base                | ~2.6× reduction | (−62 % relative) |
| HF reference (merged, fp16)               |  3.06 % | HF checkpoint, pre-deployment merge; quantization gap to deployed path ~2 pp |

The deployed bundle has two parts: the LM weights are graft-patched in-place on the `.litertlm` (uploaded as a single `.litertlm` to Hugging Face at [`psymon/gilbeot-korean-audio-litertlm`](https://huggingface.co/psymon/gilbeot-korean-audio-litertlm)), and the audio adapter ships as an alpha-8 LoRA sidecar bundled in the APK at `assets/lora/`. Touching the audio encoder too aggressively inside `.litertlm` destabilized the Adreno 740 prefill kernel — the split keeps the encoder bytes untouched.

Patcher source + the engineering saga: [`tools/README.md`](tools/README.md).

## Build from source

See [`docs/build.md`](docs/build.md). Two side-by-side build targets:

| Build | Package id | Label | Use |
|---|---|---|---|
| Judge demo       | `com.psymon.gilbeot.demo` | Gilbeot Demo | Hackathon judges anywhere |
| Korea production | `com.psymon.gilbeot.real` | Gilbeot      | Real users in Korea |

Both install side-by-side because the `applicationId`s are distinct.

## Repo layout

```
gilbeot-public/
├── app/                 Flutter project (Dart + Android)
│   ├── lib/             ~25 .dart files
│   ├── assets/
│   │   ├── env_config.example   ← copy to env_config and fill in
│   │   ├── demo/                ← bundled photos / WAV / cached route polyline
│   │   ├── demo_photos/         ← 4 hand-picked station photos
│   │   └── lora/                ← Korean audio LoRA sidecar (50.7 MB, bundled)
│   └── pubspec.yaml
├── third_party/         patched flutter_gemma 0.14.5 fork used by the app
├── patches/             notes for the flutter_gemma fork edits (MTP + LoRA sidecar)
├── scripts/             build_install_demo_apk.py + build_install_realuse_apk.py
├── tools/               .litertlm LoRA patcher (Korean audio adapter, byte-level)
└── docs/
    ├── architecture.md
    └── build.md
```

## Honest scaffolding (judge mode)

The Demo APK includes three pieces of scaffolding that a code reviewer will see — they are documented because they are necessary, not hidden.

1. **Canned route + per-photo EXIF GPS as origin** — Korean map APIs (ODsay / T-Map / NaverMap-Korea) are IP-blocked outside Korea. The cached T-Map polyline was fetched once from Korea and bundled as `assets/demo/route_polyline.json`. The 4 demo photos use their real EXIF GPS to advance the origin marker on the map.
2. **Last-photo arrival fallback** — `if (isLastDemoPhoto || textArrival) llmArrival = true;`. Empirically the model gets `is_arrival` right on photo 4 ~90 % of the time; the fallback keeps the demo from stranding on a single missed detection.
3. **Scene-context hint per step** (in English) — gives the model the *role* of the current photo ("you're at the fare gates"), but **does not** prescribe direction (left / right / up). The model reads the arrows visible in the photo itself.

The Korean production build (`com.psymon.gilbeot.real`) has none of these — it does live everything end-to-end.

## Comment language convention

Domain-logic source files (`app/lib/services/`, `app/lib/screens/`) use Korean comments — they encode reasoning specific to Korean elderly STT quality, Korean Map API quirks, Korean utterance patterns, and Korean accessibility UX decisions, where Korean is the natural language of the domain itself. Generic UI components (`app/lib/widgets/`) use English — they are intended to be reusable outside the Korean domain context. The split is deliberate.

## License

MIT — see [LICENSE](LICENSE).

_Gemma is a trademark of Google LLC._
