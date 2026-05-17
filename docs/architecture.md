# Architecture

## Stack

```
                ┌──────────────────────────────────────┐
                │       Flutter UI (Dart, Android)     │
                │  HomeScreen / DemoPhotoPreviewScreen │
                │  GilbeotMicButton / CameraButton     │
                └──────────────┬───────────────────────┘
                               │
        ┌──────────────────────┼──────────────────────────┐
        ▼                      ▼                          ▼
 GemmaService            TtsService                ODsayService /
 (lib/services/)         (flutter_tts)             TMapPedestrianService
        │                      │                  (routes + POI search) /
        │                      │                  NaverStaticMapService
        │                      │                          │
        │                      ▼                          ▼
        │             system Korean / English      Korea-only HTTP APIs
        │             TTS engine                   (production build only)
        │
        ▼   dart:ffi
 ┌─────────────────────────────────────────────────────┐
 │  flutter_gemma 0.14.5 (forked — see patches/)       │
 │  lib/core/ffi/litert_lm_client.dart                 │
 │  lib/core/ffi/litert_lm_bindings.dart               │
 └──────────────────────────┬──────────────────────────┘
                            │
                            ▼  C API
 ┌─────────────────────────────────────────────────────┐
 │  libLiteRtLm.so  (native-v0.10.2-b prebuilt)        │
 │  litert_lm_engine_settings_set_*                    │
 │  + Gemma 4 MTP runtime (drafter + acceptance)       │
 └──────────────────────────┬──────────────────────────┘
                            │
                            ▼
 ┌─────────────────────────────────────────────────────┐
 │  gemma-4-E2B-it.litertlm  (~2.4 GB, on-device)      │
 │  ─ tf_lite_vision_encoder    (208 MB)               │
 │  ─ tf_lite_vision_adapter    (4.5 MB)               │
 │  ─ tf_lite_audio_encoder_hw  (102 MB)               │
 │  ─ tf_lite_audio_adapter     (9 MB)                 │
 │  ─ tf_lite_per_layer_embedder (1.2 GB)              │
 │  ─ tf_lite_prefill_decode    (781 MB)               │
 │  ─ tf_lite_mtp_drafter       (42 MB) ← speculative  │
 │  + optional LoRA sidecar (Korean audio QAT)         │
 └─────────────────────────────────────────────────────┘
```

## Demo flow (Judge mode)

```
1. App launches
     ↓
   English splash:  "Loading on-device Gemma 4 model... (~10s)"
   First launch:    ~2.4 GB download from Hugging Face

2. User taps microphone (single tap)
     ↓
   Audio: bundled Korean WAV plays out loud (judge hears Korean)
   Internal: WAV → Gemma 4 audio → real Korean transcript
   Display: "Korean: 송파구보건소 가야 해 / I need to go to Songpa Health Center"
   State:   destination=Songpa Health Center, origin=Jamsil Station,
            route=cached T-Map walking polyline
     ↓
   English intro + TTS:
     "Heading to Songpa Health Center. Take Jamsil Station Exit 10 and
      walk about 800 meters."

3. User taps camera (per photo, repeat 4 times)
     ↓
   First tap:    push DemoPhotoPreviewScreen
                 (full-screen photo + bottom round shutter button)
   Second tap:   shutter → real Gemma 4 vision call
                 (English systemPrompt, photo + scene-context hint)
     ↓
   JSON { is_arrival, landmarks_in_photo, instruction, confidence,
          fallback_action }
     ↓
   English instruction + TTS:
     "I can see a yellow sign with numbers 1 through 11 and an arrow
      pointing to the left. If you're looking for exit 10, follow that
      arrow. You're doing great."

4. Photo 4 (destination building) → arrival
     ↓
   Status card turns primaryContainer (visual arrival indicator)
   Farewell TTS: "Thank you for using 길벗.
                  Have a good visit at Songpa Health Center."
   5 second pause → reset to home
```

## What runs on-device vs what is canned (Judge mode)

| Component | Judge mode | Korea production |
|---|---|---|
| **Gemma 4 audio (STT)** | **Real** (bundled WAV → live model) | Real (mic → live model) |
| **Gemma 4 vision** | **Real** (bundled photo → live model) | Real (camera → live model) |
| **Gemma 4 text generation** | **Real** | Real |
| **English systemPrompt** | yes (gemma_service DEMO branch) | no (Korean systemPrompt) |
| **Google TTS / system TTS** | en-US | ko-KR |
| **Destination → place lookup** | hardcoded (Songpa Health Center) | live T-Map POI API |
| **Routing** | cached T-Map polyline (asset) | live ODsay (transit) + T-Map (pedestrian) API |
| **Map preview** | NaverMap drawing canned polyline | NaverMap drawing live polyline |
| **Origin (current location)** | per-photo EXIF GPS, stepwise | live Geolocator GPS |
| **Off-route guard** | skipped (judge GPS isn't in Jamsil) | live GPS Haversine |
| **Arrival detection** | model `is_arrival` + last-photo fallback + "arrived" text | model + GPS Haversine ≤ 30 m |

The canned bits are exactly the ones that depend on Korean territory or
a Korean speaker. Everything that exercises Gemma 4 is live.

## Speculative decoding (Gemma 4 MTP)

LiteRT-LM v0.11+ supports Multi-Token Prediction for Gemma 4 — a
small drafter model proposes N tokens, the main LM verifies them in
parallel; accepted drafts skip per-token compute. The shipped
`libLiteRtLm.so` (`native-v0.10.2-b` from the DenisovAV fork's
releases) **already exports** the
`litert_lm_engine_settings_set_enable_speculative_decoding` C symbol —
upstream `flutter_gemma` 0.14.5 simply never calls it.

Our patch wires the call (see [`patches/README.md`](../patches/README.md))
and gates it by backend:

| Backend | MTP default | Why |
|---|---|---|
| GPU (S23 Adreno 740) | **on** | Measured 1.5× decode speedup, lossless. |
| NPU | on | Same family of acceleration. |
| CPU (S10e) | **off** | A/B measured: drafter forward-pass overhead exceeds acceptance gain on CPU. The LoRA sidecar shifts the target distribution away from the un-LoRA'd drafter, lowering acceptance further. Net ~10–15 s slowdown vs MTP-off per photo. |

Override via the `enableSpeculativeDecoding` named param if needed.

## On-device timing (verified)

| Device | Backend | MTP | Model | Per-photo total |
|---|---|---|---|---|
| Galaxy S23 | GPU (Adreno 740) | on | Gemma 4 E2B | **~12 s** (prefill ~5 s + decode ~7 s, ~7.5 chunks/s) |
| Galaxy S10e | CPU (XNNPACK) | off (auto) | Gemma 4 E2B | **~50 s** (post-LoRA, MTP-off) |

Prefill scales with prompt size (English DEMO prompt is ~3.3 KB / ~900
tokens text + ~256 visual tokens at 768 px); decode scales with output
length (typical 50–105 chunks per photo).

## Demo build substitutions

The judge demo substitutes two things that cannot run end-to-end outside
Korea. Both are necessary for the demo to function as a self-contained
experience for non-Korean reviewers — they are not concealments of
anything.

1. **Per-photo EXIF GPS in place of live GPS** — required because live
   GPS is meaningless when the reviewer is not at Jamsil Station. The
   4 demo photos carry their real EXIF GPS, which advances the origin
   marker on the map in step with the walk. The production build uses
   live Geolocator.

2. **Last-photo arrival fallback** — `if (isLastDemoPhoto || textArrival)
   llmArrival = true;` in `home_screen.dart`. The production build
   combines two independent signals for arrival: GPS-Haversine to the
   destination + the vision model's `is_arrival` flag. The demo has
   only the model signal, so this fallback ensures the demo does not
   strand on a single missed detection. In our S23 reviewer simulation
   the model recognised photo 4 on its own — the fallback never fired.

The Korean production build (`com.psymon.gilbeot.real`) does not
substitute either of these — it runs live end-to-end.
