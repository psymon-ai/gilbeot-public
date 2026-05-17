# Gilbeot — On-device AI that translates machine coordinates into human walking directions

> **TL;DR.** Gilbeot is an on-device multimodal AI walking assistant for elderly users on Android. The user says a destination in Korean; Gemma 4 audio analyzes the utterance and the app builds a route. When the user takes a photo on the road, Gemma 4 vision reads the scene and produces a sentence like "walk left, just below the yellow Exit 10 sign," which the app then speaks via TTS. Speech recognition, photo understanding, and instruction generation all run inside a single Gemma 4 E2B model on-device — no network needed, and voice / photo / location never leave the phone. Verified end-to-end on Galaxy S23 (GPU + MTP, ~12 s/photo) and S10e (CPU, ~50 s/photo). Technical detail in §5; international-judge flow in §2 "Two APKs."

## 1. Motivation

For older users, the moment they lose their way is not the moment they lack a map app — it is the moment the map app's words don't connect to the scene in front of them.

Maps and turn-by-turn navigation usually speak in terms of coordinates and roads:

- "In 250 m, turn right"
- "Two o'clock direction at the three-way fork"
- "Head northeast"

But for many elderly users, the actual road does not look that way. The road is understood as **the objects and scenes right in front of them** — "under that yellow sign," "past the pharmacy, on the right," "the side with the stair handrail," "the sign above the building entrance."

Gilbeot was built to close that gap.

> **Gilbeot is an on-device AI walking assistant that translates machine coordinates into human coordinates.**

The gap we wanted to close is not just a gap in map-app literacy. Modern multimodal AI can now understand speech, images, and text in one pass, but that technology still does not naturally reach the moment when an elderly person actually goes out. Gilbeot is an attempt to put Gemma 4 on top of the language and scenes that older users already know, and shrink the distance between state-of-the-art AI and everyday mobility.

The intended user flow looks like this. The user speaks a destination, and the app builds a route. While walking, when something is confusing, the user takes a photo; Gemma 4 reads the scene and translates it into a sentence the user can immediately act on. Every core inference call finishes inside the phone.

## 2. Solution Approach

### Product flow

```text
voice destination
  → Gemma 4 audio transcribes the Korean utterance
  → T-Map / ODsay / Naver Map build the route
  → user takes a photo at any confusing point
  → Gemma 4 vision interprets signs, exit numbers, arrows, and entrances
  → app speaks an instruction like "walk left, just below the yellow Exit 10 sign"
  → GPS is compared against the route polyline to detect off-route
  → if off-route, refuse to imagine guidance — ask for a re-route or new photo
  → TTS reads the instruction aloud
```

Off-route handling is a critical safety mechanism in the product flow. If the user has drifted off the planned polyline and then takes a photo, the model could look at an unrelated scene and still hallucinate "keep going." In the production build, Gilbeot computes the Haversine distance between the current GPS and the nearest walking-route segment; when it exceeds the threshold, the app suppresses guidance generation and asks the user to re-route or retake a photo.

### Why on-device

The moments when this service is actually needed are precisely the moments where the network is unreliable or the surroundings are complex — subway stations, underground shopping centers, hospital approaches, bus stops. And voice and photos are sensitive data: they reveal the user's current location and movement.

That is why Gilbeot runs the core model inference on the Android device, not on a server.

| Component | Gilbeot's choice |
|---|---|
| Speech recognition | Gemma 4 audio on-device |
| Photo understanding | Gemma 4 vision on-device |
| Instruction generation | Gemma 4 text generation on-device |
| Maps and routing | Korean production uses T-Map, ODsay, and Naver Map APIs |
| Privacy | Voice / photo / scene context never leaves the device |

There is a clear reason for picking Android first. According to the Korea Gallup 2025 smartphone survey, 92 % of Koreans in their 60s use Samsung Galaxy; among users 70 and over, Galaxy share drops slightly to 82 % but still dominates. In other words, if you have to verify on real hardware for older Korean users, the first reference point is not iOS but Samsung Galaxy on Android. [Korea Gallup 2025 smartphone survey (PDF)](https://www.gallup.co.kr/dir/GallupReport/GallupReport%2820250707%29_%EC%8A%A4%EB%A7%88%ED%8A%B8%ED%8F%B0.pdf)

### Two APKs

The Korean production app uses Korean map APIs and live device location. But a Kaggle judge may not have a Korean speaker, a Korean map API key, a Korean GPS location, or a Korean subway environment available all at once. So we built a separate Demo APK.

| Build | Package id | Role |
|---|---|---|
| Judge Demo | `com.psymon.gilbeot.demo` | Reproduces the same scenario anywhere in the world |
| Korea Production | `com.psymon.gilbeot.real` | Uses Korean map APIs and the live microphone/camera |

The Demo APK pins the route and a few inputs, but the core AI runs for real: the demo WAV is transcribed by Gemma 4 audio every time, and each of the four station photos is read by Gemma 4 vision every time. Even with judging-side conveniences in place, the model inference is not decoration — it is the actual operation that changes app state.

## 3. Development Process

The development process was organized less around adding features one by one, and more around completing **a mobility loop an elderly user could actually use**.

### 3.1 Service architecture

```text
Flutter UI
  ├─ microphone input
  │    └─ Gemma 4 audio → destination transcript
  ├─ maps / routing
  │    ├─ T-Map POI + pedestrian
  │    ├─ ODsay transit
  │    └─ Naver Map rendering
  └─ camera input
       └─ route / off-route context
            └─ Gemma 4 vision/text
                 └─ JSON {instruction, landmarks, is_arrival, arrow_tip_x, ...}
                      └─ bbox left/right correction
                           └─ TTS guidance
```

### 3.2 Target hardware range

Elderly users are not the customer segment that upgrades to the latest phone every year, so our target-device choice had to be conservative.

| Item | Galaxy S23 | Galaxy S10e |
|---|---:|---:|
| Release | 2023 | 2019 |
| Role in our test matrix | realistic upper bound | older-phone lower bound |
| SoC | Snapdragon 8 Gen 2 Mobile Platform for Galaxy | Exynos 9820 (Korean SM-G970N) |
| CPU | 64-bit octa-core, up to 3.36 GHz | octa-core, 2.73 GHz Mongoose M4 + 2.31 / 1.95 GHz Cortex |
| GPU | Adreno 740 | Mali-G76 MP12 |
| RAM | 8 GB | 6 GB |
| Primary backend | GPU (Adreno 740) | CPU (XNNPACK) |
| MTP policy | on | off |

## 4. Google AI Edge

The core of this project is using Gemma 4 not as a generic API but as the **sensory organ of the product**. We listen for a destination through audio, see the scene through vision, and speak to the user through text. All three are connected inside a single on-device multimodal model.

Gilbeot is an Edge AI project not just because the inference happens on the device. A cloud-based STT or VLM looks easier on the surface, but a system that subscribes to a specific service and ships an older user's location and voice to a server every time does not match Gilbeot's intent.

Thanks to Gemma 4 and LiteRT-LM, Gilbeot can keep these principles:

- Destination utterances, photos, and scene context never go to a model server.
- After the initial model download, the core guidance loop runs offline.
- On a good phone it goes fast via GPU/MTP; on an older phone it stays slow but functional via the CPU-safe path.
- All multimodal processing stays inside one model family, keeping the product structure simple.

### 4.1 Why Gemma 4 was the right choice

| Gemma 4 capability | Role in Gilbeot |
|---|---|
| Audio understanding | Convert a Korean destination utterance into a transcript |
| Vision understanding | Read signs, exit numbers, arrows, and building entrances in the photo |
| Text generation | Produce short, kind walking instructions an elderly user can follow |
| Native bbox-style output | Provide arrow tip / tail coordinates for left-right correction |
| LiteRT-LM deployment | Run on Android without a server |
| MTP / speculative decoding | Improve decode speed on the S23 GPU |

The thing that fit Gemma 4 especially well to this product is that "the inputs needed for walking guidance are inherently multimodal." The destination is audio, the uncertainty on the road is a photo, and the final output is a short natural-language instruction. Stacking several specialized models would accumulate errors and latency at each step; Gemma 4 E2B handles all three within a single model family.

### 4.2 Model bundle spec

The Gemma 4 E2B `.litertlm` bundle Gilbeot uses is about 2.4 GB.

| Internal component | Role | Size |
|---|---|---:|
| `tf_lite_vision_encoder` | image encoding | ~208 MB |
| `tf_lite_vision_adapter` | vision embedding adapter | ~4.5 MB |
| `tf_lite_audio_encoder_hw` | audio encoding | ~102 MB |
| `tf_lite_audio_adapter` | audio embedding adapter | ~9 MB |
| `tf_lite_prefill_decode` | LM prefill / decode | ~781 MB |
| `tf_lite_per_layer_embedder` | MTP-conditioning component | ~1.2 GB |
| `tf_lite_mtp_drafter` | speculative-decoding drafter | ~42 MB |

This structure lets a single model file handle speech, image, and text. The flip side is that actually running this model in a mobile app demands very careful memory and backend choices.

## 5. Challenges I faced

Below is a short summary of the problems we ran into building Gilbeot, and how each one was resolved.

### 5.1 Korean audio training was necessary

Speech recognition is the first door into Gilbeot. If the destination is misheard, every later step of the route is wrong. Korean place names, facility names, and the imprecise pronunciation, slow speed, and irregular pauses common in elderly speech were not stable on the base model alone.

The internal evaluation used 134 Korean utterance samples covering destinations and everyday phrases. The metric is CER (Character Error Rate).

| Model / path | CER | Meaning |
|---|---:|---|
| Base gemma-4-E2B-it-litert-lm (no LoRA) | 13.14 % | Not usable for destination input |
| Gilbeot deployed LiteRT-LM path | **5.00 %** | Usable in the Android deployment path |
| Improvement vs device base | ~2.6× reduction | (−62 % relative) |

5.00 % is not just a benchmark number. That is the level you need before you can actually handle destination utterances like "Songpa Health Center," "Gangnam Station," or "National Museum of Korea" inside a product flow.

#### 5.1.1 How much data, and how we gathered it

The training data is about 45,823 utterances (about 50 hours of audio, a speaker-stratified subset; after quality filtering, 43,431 for training and 2,291 for evaluation). The main source was AI Hub's **Command Speech (Elderly Male/Female)** dataset — a Korean public dataset built specifically to transcribe spoken commands from elderly Koreans into text and develop the Korean speech-language tech around them.

| Source | Purpose |
|---|---|
| AI Hub Command Speech (Elderly Male/Female) | Captures elderly speakers' imperative speech, slow speaking rate, intonation, and pause distribution |
| Gilbeot curated/normalized subset | Focused on destination requests, facility names, and short walking-intent utterances |

Because Gilbeot is not a general dictation app but a destination-input app, we picked from the public data the imperative / request-form utterances closest to "the first sentence of a walking instruction," and normalized the transcripts for both evaluation and training. [AI Hub — Command Speech (Elderly Male/Female)](https://www.aihub.or.kr/aihubdata/data/view.do?aihubDataSe=data&dataSetSn=94)

#### 5.1.2 How we trained it

The model is Gemma 4 E2B. We applied rank-8 LoRA to the audio encoder and to the attention / MLP layers on the LM side. Training went through the Unsloth FastModel + bitsandbytes 4-bit + PEFT path.

The detail that mattered was Unsloth FastModel. The Gemma 4 audio side has custom layers such as `Gemma4ClippableLinear`, and the regular PEFT path silently skipped some of them. On the surface, loss went down, but the actual audio output did not change. The Unsloth path handled those layers correctly, and at the HF checkpoint level we reached 3.06 % CER.

### 5.2 Bringing it to mobile

Deployment turned out to be harder than training. At the time of submission, no public toolchain provided a clean path to land a Gemma 4 audio LoRA inside an Android `.litertlm` bundle.

#### 5.2.1 No official `.litertlm` support, and what we did instead

At submission time, the public paths looked like this:

| Path | Problem we hit |
|---|---|
| Official LiteRT-LM converter | Text-side LoRA focused; audio path raised unsupported-shape errors |
| `litert-torch-nightly` | LM export works, but the audio-encoder deployment path doesn't fit |
| MediaPipe LLM Inference | Public LoRA path is attention-only and GPU-runtime-centric |
| Simple HF adapter deployment | No official way to plug it into the Android `.litertlm` runtime |

So we built a `.litertlm` patcher ourselves. We tried three approaches:

| Attempt | Method | Result |
|---|---|---|
| Fresh quantization | Add the LoRA delta, then re-quantize to int4 with a new scale | Small deltas vanish; scale mismatch causes instability |
| Quant-projected flip | Keep the existing int4 grid; nudge selected entries one step toward the largest `\|delta\|/scale` direction | Some improvement, but per-layer tuning unstable |
| Same-grid requant | Keep per-row scale; dequant → add delta → requant with the same scale | Final choice |

The key idea of the final method is to **not** create a new scale.

```text
existing .litertlm block:
  scale, int4_weights

patch:
  base_fp32     = dequantize(scale, int4_weights)
  patched_fp32  = base_fp32 + LoRA_delta
  patched_int4  = round(patched_fp32 / scale)   # reuse the existing scale
  patched_int4  = clip(patched_int4, -8, 7)
  write back
```

This way, what survives is only the LoRA direction of movement, and we preserve as much of the quantization grid that the runtime kernel expects as possible.

The final deployed path has two parts:

| Part | Method |
|---|---|
| LM side | Graft-patch the LM weights inside `.litertlm` |
| Audio side | Attach an alpha-8 LoRA sidecar via the runtime `loraPath` |

Touching the audio encoder too aggressively inside `.litertlm` destabilized the Adreno 740 prefill kernel. So we separated the audio-side delta into a sidecar and added a path to attach it through the LiteRT-LM session config.

#### 5.2.2 Cactus's workaround and the INT4/INT8 limits

Cactus is a different path that does not go through `.litertlm`. Instead of touching the quantized weights inside the LiteRT-LM binary the way our patcher does, Cactus brings its own mobile inference engine and re-converts the Gemma 4 model into its own format. So we and Cactus solve the same problem — "run Gemma 4 audio on Android" — at two different layers: we keep the existing runtime in place and patch the weights, while Cactus replaces the runtime itself.

On accuracy alone, Cactus was quite attractive.

| Backend | CER | S10e median/sample | Verdict |
|---|---:|---:|---|
| HF merged reference | 3.619 % | 0.691 s | Cloud reference |
| LiteRT best, graft + alpha-8 sidecar | 5.003 % | 12.48 s | Production choice |
| Cactus INT8 | 3.949 % | 41.94 s | Better accuracy, 3.4× slower |
| Cactus INT4 | 4.856 % | 45.32 s | Smaller but even worse latency |

The blocker was latency. On the S10e, the LiteRT path measures prefill 11,992 ms + decode 3,389 ms = 15,381 ms total for a single mic input — already not what you'd call fast. In the same environment, Cactus measures 42–45 seconds per sample. That goes past the window of time an elderly user can patiently wait between speaking a destination and hearing the first instruction.

So the final submission app keeps the LiteRT-LM `.litertlm` + LoRA sidecar path. But this conclusion is bounded by the two devices we verified — S10e (2019) and S23 (2023). On hardware with more compute, for example a recent flagship or a device with strong NPU acceleration, Cactus's accuracy advantage would carry over while the latency concern would shrink; in that case Cactus is still a good choice.

One important caveat: the accuracy numbers above are after fixing Cactus's byte-fallback detokenization bug. When Gemma 4 emits codepoints that are not a single SentencePiece piece — which happens often in Korean — it falls back to byte tokens like `<0xEA><0xB9><0xB0>`. Cactus was passing those through as literal strings, so a sentence like "잠 깰 수 있는" surfaced as "잠 `<0xEA><0xB9><0xB0>` 수 있는." That one-line detokenization bug, once fixed, moved INT8 CER from 4.638 % to 3.949 % and INT4 from 5.545 % to 4.856 % on the 134-sample Korean evaluation. The fix is upstreamed in [cactus-compute/cactus#635](https://github.com/cactus-compute/cactus/pull/635).

### 5.3 VLM left/right limits, and the bbox-based fix

Gilbeot's photo guidance is not "describe the picture" but "tell me where to go right now." So instead of a pretty caption, the model has to safely read the actual cues in front of the user.

The problem is that VLMs are weak at basic spatial reasoning such as left and right. This is not just a small-model issue but a general VLM weakness. The EMNLP 2023 paper **"What's 'up' with vision-language models? Investigating their struggle with spatial reasoning"** by Kamath, Hessel, and Chang evaluated 18 vision-language models and reports that all of them underperform on basic spatial relations. [ACL Anthology](https://aclanthology.org/2023.emnlp-main.568/)

A more direct piece of evidence on left/right is **"What's left can't be right — The remaining positional incompetence of contrastive vision-language models"** by Hoehing, Rushe, and Ventresque, which focuses on the simple left-right positional relation in CLIP-family models and shows the failure pattern is predictable even with large-scale data. [arXiv:2311.11477](https://arxiv.org/abs/2311.11477)

Gemma 4 read exit numbers and signs well, but it occasionally flipped "left" and "right" when shown a horizontal arrow. In a walking-guidance product for older users, this is not a small error.

The fix was to use Gemma 4's native bbox-style output. We added two fields to the response JSON:

```json
{
  "arrow_tip_x": null,
  "arrow_tail_x": null
}
```

When a horizontal arrow is visible, the model returns the x coordinates of its tip and tail, normalized to 0–1. The app compares the two numbers:

```text
tip_x < tail_x  →  arrow points left
tip_x > tail_x  →  arrow points right
```

If the model's instruction sentence says the opposite direction from what the pixel coordinates indicate, the app corrects the word. This is not the same as putting the answer in the prompt — the model still does the localization, and a deterministic block of code makes the left/right decision.

As a final check, we flipped the same photo horizontally. The original came back as a left arrow and the flipped version as a right arrow, with both the coordinates and the sentence changing together. That confirmed the model was actually looking at the image, not memorizing a constant answer.

### 5.4 Pushing inference speed up

Speed is not a convenience feature — it is accessibility. If the user has to wait too long after taking a photo on the road, an accurate model still cannot ship as a real product.

We did not set the latest flagship phone as the reference point.

| Tier | Device | Goal |
|---|---|---|
| Upper bound | Galaxy S23, 2023 model | Comfortable even on a non-flagship phone |
| Lower bound | Galaxy S10e, 2019 model | Still completes on an older phone |

Gemma 4's MTP (Multi-Token Prediction) is a speculative-decoding scheme: a small drafter proposes several tokens up front, and the main model verifies them in parallel. On a GPU this can cut decode time with no quality loss.

On the S23, MTP was a clear win.

| Item | baseline | MTP on |
|---|---:|---:|
| decode time | ~17.8 s | ~11.5 s |
| speedup | – | ~1.5–1.6× |
| quality | preserved | preserved |

On the S10e CPU it was the opposite. The drafter and main model share the same compute, and the distribution shift between the fine-tuned main LM and the base drafter lowers the acceptance rate as well. So MTP overhead exceeded its gain.

Final policy:

| Backend | MTP default | Reason |
|---|---|---|
| GPU / NPU | on | Quality-neutral speedup on S23 |
| CPU | off | MTP-off is faster on S10e |

Final on-device performance:

| Device | Backend | MTP | Per-photo guidance time | Usability |
|---|---|---|---:|---|
| Galaxy S23 | GPU | on | ~12 s | Comfortable for both demo and real use |
| Galaxy S10e | CPU | off | ~50 s | Slow, but still completes on an older phone |

### 5.5 The judge-mode pinned flow, and honesty about it

ODsay, T-Map, and Naver Map are strongly bound to the Korean environment. It is unrealistic for an international judge to come with a Korean API key, a Korean GPS location, photos of Korean subway signs, and a Korean spoken utterance all at once.

The answer was the Demo APK. But not a simple scripted demo — we kept these principles:

| Item | Demo APK handling |
|---|---|
| Route | Use a route that was cached in Korea ahead of time |
| Voice input | Bundled Korean WAV |
| STT | Gemma 4 audio runs for real |
| Photo guidance | Gemma 4 vision runs for real |
| Destination check | Route only starts when the transcript points to "Songpa Health Center" |
| Off-route photos | In the long-press gallery flow, EXIF metadata stands in for live GPS |

Long-pressing the camera button opens the phone's gallery. The user can pick anything; whatever they pick goes through the same on-device Gemma 4 vision path as the canned demo. If the EXIF GPS is far from the planned route, the app does not invent guidance — it only describes the photo and notes that the photo does not look like it is on the planned route. This feature is what demonstrates that the demo is not a four-photo lookup table — it really is a live vision-model flow.

## 6. Future Work

Wayfinding is a starting point. The bigger destination Gilbeot is heading toward is to dismantle the barrier between elderly users and AI technology.

The difficulties elderly users face in front of a smartphone are not limited to wayfinding. Phone banking, public-document processing, hospital appointments, medication reminders, welfare-service applications, organizing photos and documents to send to family — important everyday problems are increasingly moving into the phone. Most of them touch personal data, health data, or financial data, so they cannot be shipped to the cloud casually. There is a real reason to handle them on-device.

The next steps of Gilbeot could expand in these directions.

| Area | Future task |
|---|---|
| Wayfinding | Wider coverage, live off-route re-routing, on-the-ground user testing |
| Phone-banking assistance | On-device guidance that doesn't just say "press this button" but proactively flags risky transfers and phishing |
| Public-document processing | Read photos of community-center / health-insurance / welfare documents and summarize them in plain language |
| Hospital appointments | Read hospital apps, SMS messages, and appointment slips, and organize the schedule and prep items inside the device |
| Family connection | Generate "this is where I am and what's going on"-style updates for adult children, released only on user approval |

Wayfinding got picked first because the problem is most visible on the road. But the real problem is the gap between "what AI can do" and "what an older user can actually do on a smartphone." Gilbeot is a first instance of an on-device AI interface aimed at closing that gap.

## 7. Thanks

The last words of this writeup go to one person.

To my father-in-law — who jumped into product testing more eagerly than anyone and gladly stepped in front of the camera for the demo recording: without you, this project would not have crossed the finish line. Through every stretch of building Gilbeot you were there with honest feedback and unwavering encouragement. Thank you, truly.

As you begin this new chapter after retirement, may every road ahead be filled with happiness.

## 8. License

This writeup is released under the Creative Commons Attribution 4.0 International (CC BY 4.0) license.

## 9. Citation

Psymon. Gilbeot — On-device multimodal AI walking assistant for elderly users. https://www.kaggle.com/competitions/gemma-4-good-hackathon/writeups/gilbeot. 2026. Kaggle.
