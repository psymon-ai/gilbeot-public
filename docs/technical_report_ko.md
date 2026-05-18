<div align="right">

[EN](technical_report.md) · [KO](technical_report_ko.md)

</div>

# 길벗(Gilbeot) — 기술 리포트

이 문서는 [Kaggle writeup](../KAGGLE_WRITEUP_KO.md)의 **§5 Engineering Challenges**에 대한 풀버전 부록입니다.

---

## 목차

1. [제품 루프 전체 구조](#1-제품-루프-전체-구조)
2. [Audio — 고령 한국어 발화 LoRA](#2-audio--고령-한국어-발화-lora)
3. [Deployment — `.litertlm`에 LoRA 넣기](#3-deployment--litertlm에-lora-넣기)
4. [Cactus — 병행 조사](#4-cactus--병행-조사)
5. [Vision — 좌/우 안전성](#5-vision--좌우-안전성)
6. [Latency — 단말별 backend 정책](#6-latency--단말별-backend-정책)
7. [Demo APK — 무엇을 고정했고 무엇을 실제로 돌렸나](#7-demo-apk--무엇을-고정했고-무엇을-실제로-돌렸나)
8. [References](#8-references)

---

## 1. 제품 루프 전체 구조

길벗에서 어려운 부분은 어느 한 모델의 점수가 아니었습니다. **한국어 오디오 → 모바일 배포 → 안전한 시각 안내 → 실기기 지연시간**이 어르신의 폰 위에서 한 제품 루프로 동작해야 했고, 각 단계는 따로 떨어뜨려 풀 수 없었습니다.

- 오디오가 좋아도 Android `.litertlm`에 얹히지 않으면 제품이 아닙니다.
- patcher가 잘 동작해도 시각 안내가 좌/우를 뒤집으면 어르신에게 위험합니다.
- 안전한 안내라도 한 문장이 몇 분이나 걸리면 어르신은 쓰지 않습니다.

이 문서는 그래서 네 단계의 루프와, 해외 심사를 위해 부득이하게 만든 demo APK 경계, 이렇게 다섯 챕터로 나뉩니다.

---

## 2. Audio — 고령 한국어 발화 LoRA

### 2.1 학습이 필요했다

음성 인식은 길벗의 첫 문입니다. 목적지를 잘못 들으면 이후 모든 경로가 틀어집니다. 한국어 지명·시설명과, 고령 발화 특유의 느린 속도와 불규칙한 휴지는 base Gemma 4 audio만으로는 안정적이지 않았습니다.

내부 평가는 134개의 한국어 held-out 발화로 진행했고, 지표는 CER(Character Error Rate)입니다.

| 경로 | CER | 의미 |
|---|---:|---|
| Base `gemma-4-E2B-it-litert-lm` (no LoRA) | 13.14% | 목적지 입력용으로 부적합 |
| HF merged reference | 3.619% | 클라우드 참고선 |
| **길벗 deployed LiteRT-LM path** | **5.00%** | **production 채택** |

5.00%는 "송파구 보건소", "강남역", "국립중앙박물관" 같은 목적지를 제품 흐름 안에서 안정적으로 다룰 수 있는 기준선입니다. 클라우드 참고선(3.619%)과 on-device bundle(5.00%) 사이 ~1.4pp 갭은 공식 `.litertlm` audio-LoRA export 경로가 없는 상태에서 byte-level patcher로 우회한 비용이고, 그 비용의 출처는 §3에서 설명합니다.

### 2.2 데이터를 얼마나, 어떻게 모았나

학습 데이터는 약 45,823개 utterance — 약 50시간 — 으로, 핵심 원천은 AI Hub의 [**명령어 음성(노인남녀)**](https://www.aihub.or.kr/aihubdata/data/view.do?aihubDataSe=data&dataSetSn=94) 데이터셋입니다. 이 데이터셋은 한국인 노인남녀의 음성 명령어를 문자로 바꾸고 문맥을 이해하는 한국어 음성언어처리 기술 개발을 목표로 구축된 공개 데이터입니다. 화자 단위 stratified split:

| split | utterance 수 |
|---|---:|
| Train | 43,431 |
| Eval | 2,291 |
| Held-out (CER 보고용) | 134 |

길벗은 일반 받아쓰기 앱이 아니라 *목적지 입력* 앱이므로, 공개 데이터 안에서 "길안내 첫 문장"에 가까운 명령형·요청형 발화를 선별했고, 전사 텍스트는 평가와 학습에서 같은 규칙으로 정규화했습니다.

### 2.3 학습 스택과 silent-skip 함정

| 구성요소 | 선택 |
|---|---|
| Base model | Gemma 4 E2B (multimodal) |
| LoRA rank | 8 |
| LoRA targets | audio encoder + LM attention/MLP |
| Quantization | bitsandbytes 4-bit (NF4) |
| Adapter framework | PEFT |
| Training driver | **Unsloth FastModel** |
| Epochs | 2 |

여기서 중요한 디테일은 Unsloth FastModel입니다. Gemma 4 audio 쪽에는 `Gemma4ClippableLinear` 같은 custom layer가 있고, 일반 PEFT 경로는 이 일부를 조용히 건너뜁니다. 겉으로는 loss가 잘 줄지만 실제 audio 출력은 바뀌지 않는 상황이 생기는데, trainable parameter 리스트를 instrumentation 하기 전까지 이 silent skip에 약 하루를 잃었습니다.

Unsloth 경로는 이 layer들을 정확히 처리했습니다. 그 결과 unmerged HF checkpoint 수준에서 **CER 3.06%**까지 도달했고, 13.14% device base 대비 >4× 오류 감소입니다. 이 adapter를 base에 merge해서 클라우드에서 돌리면 정밀도가 살짝 빠져 위 표의 3.619% 클라우드 참고선이 됩니다.

---

## 3. Deployment — `.litertlm`에 LoRA 넣기

### 3.1 2026년 5월 시점의 공개 toolchain

학습보다 어려웠던 것은 배포였습니다. 제출 시점에 Gemma 4 audio LoRA를 Android `.litertlm` bundle에 넣는 공개 경로를 모두 확인해 봤지만, 어느 것도 그대로 쓸 수 있는 길이 아니었습니다.

| 경로 | 막힌 지점 |
|---|---|
| 공식 LiteRT-LM 변환기 | text-side LoRA 중심, audio path는 unsupported-shape 에러 |
| `litert-torch-nightly` | LM export는 가능하지만 audio encoder 경로는 부적합 |
| MediaPipe LLM Inference | 공개 LoRA path가 attention-only / GPU runtime 중심 |
| 단순 HF adapter 배포 | Android `.litertlm` runtime에 직접 연결할 공식 방법 없음 |
| Cactus | `model_gemma4_audio.cpp`는 동작하지만 converter가 *pre-merged* HF checkpoint를 가정, 당시 prebuilt Android binary는 x86-Linux 빌드 전용 |

각 경로는 소스를 직접 읽어 확인했고, 어느 것도 straight-through path가 아니었습니다. 그래서 byte-level `.litertlm` patcher를 직접 만들었습니다.

### 3.2 세 가지 patcher 시도

Patcher는 `tools/patch_gemma4_lora_requant.py`(companion 둘과 함께)에 있고, README는 [`tools/README.md`](../tools/README.md)에 있습니다. 마지막 세 번째 design만 실제로 동작했습니다.

| 시도 | 파일 | 방식 | 결과 |
|---|---|---|---|
| 1. Fresh requant | `patch_gemma4_lm_fresh.py` (historical) | LoRA delta를 더한 뒤 새 scale로 int4 재양자화 | 출력에 측정 가능한 변화 없음. 작은 delta는 새 scale로 반올림 소실, 살아남은 delta도 위치가 임의로 흩어짐 |
| 2. Quant-projected flip | `patch_gemma4_quant_projected.py` | 기존 int4 grid 유지, \|delta\|/scale이 큰 상위 K개를 LoRA 방향으로 ±1 LSB flip | 일부 개선되지만 flip fraction sweet spot이 layer별로 달라 불안정 |
| 3. **Same-grid requant** | `patch_gemma4_lora_requant.py` | per-row scale 유지, dequant → delta add → 같은 scale로 requant | **production 채택** |

### 3.3 Same-grid requant — 왜 동작하나

마지막 방식의 핵심은 **scale을 새로 만들지 않는 것**입니다. Runtime kernel은 원래의 양자화 grid에 컴파일되어 있는데, scale을 다시 계산하면 rounding 경계가 흔들리고 가장 먼저 죽는 것이 LoRA delta의 신호입니다. 그래서 경계를 그대로 두고, LoRA 방향이 어떤 계수를 다음 정수 step으로 밀어 올리거나(반 LSB 이상 기여한 경우) 아니면 그대로 두도록 합니다. 두 경우 모두 정의상 LoRA 방향입니다.

```text
HF base와 .litertlm 사이 매칭되는 weight block 각각에 대해:
  scale, ints   ← unpack_int4(litertlm_block)
  base_fp32     ← dequantize(scale, ints)
  patched_fp32  ← base_fp32 + (lora_A @ lora_B) * (alpha / rank)
  patched_ints  ← round(patched_fp32 / scale)        # 같은 scale!
  patched_ints  ← clip(patched_ints, -8, 7)          # int4 범위
  litertlm_block.replace(pack_int4(scale, patched_ints))
```

Patcher는 `.litertlm` FlatBuffers container를 파싱해 audio-encoder와 LM weight block을 찾고, Google의 INT2/INT4 packing에서 dequantize한 뒤 LoRA delta를 적용하고, 같은 grid로 다시 양자화한 다음, 동일 크기의 TFLite section을 in-place로 다시 씁니다.

### 3.4 Graft + sidecar 분리

| 부분 | 방식 | 이유 |
|---|---|---|
| LM 쪽 | `.litertlm` 안의 LM weight에 graft patch | 134-utt eval에서 안정적, Adreno 740 prefill 통과 |
| Audio 쪽 | alpha-8 LoRA sidecar를 runtime `loraPath`로 연결 | `.litertlm` 안의 audio encoder weight를 직접 건드리면 Adreno 740 prefill kernel이 불안정 (xnnpack 컴파일 시점의 int4 codebook과 mismatch) |

Sidecar는 `flutter_gemma`의 `loraPath`로 runtime에 연결되며, fork의 Patch 4가 필요합니다 ([`patches/README.md`](../patches/README.md) 참고). Upstream `flutter_gemma` 0.14.5는 FFI `.litertlm` 경로에서 `loraPath`가 들어오면 throw 합니다. 공개 C API에 LiteRT-LM internal `SessionConfig::SetScopedLoraFile` setter가 없기 때문입니다. 그래서 앱쪽 shim(`libgilbeot_litertlm_lora.so`)을 두고, 내부 C++ symbol을 `dlsym`한 다음 `ConversationConfig`가 snapshot 하기 전에 opaque `SessionConfig`에 sidecar를 붙입니다.

### 3.5 Cache invalidation 함정

프로젝트에서 가장 demoralizing 했던 하루입니다.

LiteRT-LM은 첫 모델 로드 시 `/data/data/<pkg>/app_files/...` 아래에 컴파일된 kernel을 캐시합니다(`xnnpack/`, `mldrift/` subdir). **이 캐시 키가 모델 파일 이름이고, content hash가 아닙니다.** 같은 파일명으로 patch된 `.litertlm`을 push하면 다음 launch는 *예전* weight로 컴파일된 캐시 kernel을 조용히 재사용합니다. 시도한 byte-level patch가 전부 무효처럼 보였고 — 격리 환경에서 분명히 옳다고 검증된 patch도 포함해서 — patcher 방향 자체가 막힌 것 아닌가 의심까지 갔습니다.

해결은 ADB 한 줄입니다.

```bash
adb shell run-as com.psymon.gilbeot.real \
    rm -rf app_files/xnnpack app_files/mldrift
```

이후 모든 device CER eval은 이 명령으로 시작합니다.

---

## 4. Cactus — 병행 조사

Cactus는 `.litertlm`을 거치지 않는 별도 경로입니다. 우리 patcher가 LiteRT-LM runtime을 그대로 두고 weight만 patch하는 반면, Cactus는 자체 모바일 추론 엔진을 두고 Gemma 4를 자기 포맷으로 다시 변환합니다. 같은 문제("Android에서 Gemma 4 audio 돌리기")를 두 다른 layer에서 푸는 셈입니다.

### 4.1 Byte-fallback detokenization 버그 (PR #635)

비교를 돌리던 중 Cactus가 literal byte-token 문자열을 transcript에 그대로 흘리고 있다는 점을 발견했습니다. Gemma 4는 codepoint가 단일 SentencePiece piece에 없을 때 `<0xEA><0xB9><0xB0>` 같은 byte-fallback 토큰을 emit 하는데 — 한국어에선 거의 매번 — Cactus는 이를 UTF-8로 재조합하지 않고 literal 문자열로 통과시키고 있었습니다. "잠 깰 수 있는" 같은 문장이 "잠 `<0xEA><0xB9><0xB0>` 수 있는"으로 출력됐습니다.

한 줄짜리 detokenization fix가 134-sample 한국어 평가에서 CER을 다음과 같이 끌어내렸습니다.

| Backend | 수정 전 CER | 수정 후 CER |
|---|---:|---:|
| Cactus INT8 | 4.638% | 3.949% |
| Cactus INT4 | 5.545% | 4.856% |

수정은 [cactus-compute/cactus#635](https://github.com/cactus-compute/cactus/pull/635)에 upstream PR로 올렸습니다. §4.2의 정확도 수치는 모두 post-fix 값입니다.

### 4.2 Accuracy vs latency

아래 latency 수치는 **compiler-board anchor**이고, S10e mic flow의 실측 시간이 아닙니다. 네 backend 모두 같은 `scripts/gemma4_edge_compiler_board.py` harness 위에서 134-sample 한국어 held-out 셋을 돌렸기 때문에, sample당 초에는 board overhead(sample 간 reload, fixture I/O, eval driver)가 pure inference 위에 더 얹혀 있습니다. 이 표가 말하려는 건 backend 사이의 *상대 격차*이고, 절대 초 자체는 사용자가 실 mic flow에서 보는 시간과 다를 수 있습니다.

| Backend | CER | Board median/sample | 판단 |
|---|---:|---:|---|
| HF merged reference | 3.619% | 0.691초 | 클라우드 참고선 |
| LiteRT graft + alpha-8 sidecar | 5.003% | 12.48초 | production 채택 |
| Cactus INT8 | 3.949% | 41.94초 | 정확도 우위, 같은 harness에서 ~3.4배 느림 |
| Cactus INT4 | 4.856% | 45.32초 | 더 작지만 latency는 더 나쁨 |

정확도만 보면 Cactus는 매력적이었습니다. 막힌 곳은 *상대* latency입니다. 같은 board harness에서 Cactus는 sample당 LiteRT 경로의 ~3.4배 시간이 걸렸습니다. 이 비율을 2019년형 S10e의 Gemma 4 E2B audio(102MB encoder + 781MB LM prefill/decode) mic flow에 옮겨 보면, LiteRT 경로가 얼마가 걸리든 Cactus는 그것을 곱해 어르신이 목적지를 말하고 첫 안내가 시작되기까지 기다릴 수 있는 범위를 한참 넘기게 됩니다. 그래서 production 앱은 LiteRT-LM `.litertlm` + LoRA sidecar 경로를 유지했습니다.

이 결론은 우리가 검증한 두 단말 S10e(2019)와 S23(2023) 범위 안에서만 성립합니다. 더 강력한 컴퓨팅 환경 — 최신 플래그십이나 NPU 가속이 풍부한 단말 — 에서는 Cactus의 정확도 우위는 그대로 남고 latency 우려는 줄어들 수 있고, 그런 경우엔 Cactus가 여전히 좋은 선택입니다.

---

## 5. Vision — 좌/우 안전성

### 5.1 문헌으로 보고된 VLM 약점

길벗의 사진 안내는 "사진을 설명하는 기능"이 아니라 "지금 어디로 가야 하는지 말해 주는 기능"입니다. 그래서 모델은 예쁜 caption이 아니라 어르신 눈앞의 실제 단서를 안전하게 읽어야 합니다.

문제는 VLM이 기본 공간 추론, 특히 좌/우에 취약하다는 점입니다. 이건 작은 모델만의 문제가 아니라 VLM 전반에서 관찰되는 약점입니다. Kamath, Hessel, Chang의 EMNLP 2023 논문 [*What's "up" with vision-language models? Investigating their struggle with spatial reasoning*](https://aclanthology.org/2023.emnlp-main.568/)는 18개 VLM을 평가했고, 모두 좌/우 pair에서 50% chance 수준에 머무는 반면 사람은 약 99%였다고 보고합니다. 좌/우에 더 직접적인 근거는 Hoehing, Rushe, Ventresque의 [*What's left can't be right — The remaining positional incompetence of contrastive vision-language models*](https://arxiv.org/abs/2311.11477)로, CLIP 계열 모델의 좌/우 관계를 집중 분석하고 대규모 데이터셋에서도 실패 양상이 예측 가능하다고 보여 줍니다.

Gemma 4도 출구 번호와 표지판은 잘 읽지만, 가로 화살표에서는 "left"와 "right"를 가끔 뒤집었습니다. 어르신 길안내에서 이건 작은 오류가 아닙니다.

Chain-of-thought prompting도 시도했는데 **해결되지 않았습니다**. 작은 VLM은 JSON schema에 instruction 한 줄만 들어가는 상황에서, 내부 hidden state로 CoT를 끌고 가지 못합니다.

### 5.2 bbox-style output

해결책은 Gemma 4의 native bbox output을 활용하는 것이었습니다. 응답 schema에 두 개의 optional field를 추가했습니다.

```jsonc
{
  ...,
  "arrow_tip_x": null,   // 0=좌측 가장자리, 1=우측 가장자리
  "arrow_tail_x": null,  // 가로 화살표가 없으면 null
  "instruction": "...",
  ...
}
```

모델은 가로 화살표가 보일 때만 이 두 값을 채웁니다. Dart 코드는 두 x 좌표를 비교해 **instruction 안의 좌/우 단어를 픽셀 산술 기준으로 덮어씁니다**.

```dart
final pixelDir = tipX < tailX ? 'left' : 'right';
if (RegExp(r'\b' + opposite(pixelDir) + r'\b').hasMatch(instruction)) {
  instruction = instruction.replaceAll(opposite(pixelDir), pixelDir);
}
```

이건 답을 prompt에 심는 것과는 **다릅니다**. 모델에는 화살표를 *localize* 하라고 — vision encoder가 잘하는 일을 — 부탁하고, 좌/우 판정은 deterministic code가 plain 산술로 합니다. 모델은 우리 코드가 어느 쪽을 "left"로 보는지 알 수 없습니다. 좌표는 그저 숫자입니다.

사진당 비용은 prefill 약 +0.3초(field 두 개 declaration 추가), decode 약 0초(화살표가 있을 때만 채워짐) 정도입니다.

### 5.3 단어 빈도 prior 버그

bbox 경로를 다듬다가 *일관되게* "to your right"가 나오는 현상을 만났는데, random confusion이 아니라 한쪽으로 쏠린 bias였습니다. 경험적 probe로, schema 예시 instruction(`"I can see the exit sign right ahead..."`)에서 단어 `right`만 빼서(`"...directly ahead..."`) 다시 돌렸더니, 일관 "right"이던 출력이 일관 *hedging*("follow the indicated direction", 방향 단어 자체가 사라짐) 으로 뒤집혔습니다.

작은 VLM은 prompt 안의 자기 example 출력을 강한 vocabulary prior로 다룹니다. 모델은 "right ahead"(부사)의 `right`를 directional 출력으로 끌어다 쓰고 있었습니다. 수정 방향은, 우선 prompt에서 불필요한 "right"/"left"를 모두 제거하고, *대칭*인 LEFT/RIGHT example phrasing을 추가하는 것이었습니다.

```text
Example phrasings for arrow photos (pick ONE direction word):
- "The arrow points to the LEFT — please walk to the left..."
- "The arrow points to the RIGHT — please walk to the right..."
```

명시적 룰도 추가했습니다: *"가로 화살표가 보이면 instruction에 'left' 또는 'right' 중 하나가 들어가야 한다. 픽셀 좌표 arrow_tip_x/arrow_tail_x와 단어가 충돌하면 앱이 단어를 덮어쓴다."*

### 5.4 Mirror experiment

마지막 확인으로, test photo 를 EXIF-aware하게 좌우 반전해서(렌더되는 장면이 실제로 뒤집힘) 화살표가 오른쪽을 가리키게 만든 뒤 demo flow를 다시 돌렸습니다.

```text
Photo (mirrored):
  arrow_tip_x: 0.75, arrow_tail_x: 0.25,
  instruction: "I can see a yellow sign with an arrow pointing
                to the right. Please walk to the right..."
```

좌표는 원본 (0.15, 0.45)에서 mirror (0.75, 0.25)로 옮겨갔고, 방향 단어도 함께 바뀌었습니다. 모델이 상수 답을 외운 게 아니라 실제로 이미지를 보고 있다는 확인이었습니다.

### 5.5 S23에서 관찰된 세 가지 regime

| 좌표 반환? | 모델 wording | 최종 spoken instruction |
|---|---|---|
| no | "...the arrow points to the left..." | 그대로 (정답) |
| yes, 모델이 "right" | "...follow the arrow to your right..." | **override → "...to your left..."** |
| yes, 모델이 "left" | "...follow the arrow to your left..." | 그대로 |

Override(`home_screen._applyArrowBboxOverride`)는 demo 빌드와 한국 production 빌드 양쪽에 연결되어 있습니다. 총 추가량은 Dart 약 25줄과 field 3개 declaration, 그리고 두 prompt template의 대칭 example phrasing입니다.

---

## 6. Latency — 단말별 backend 정책

여기서 속도는 편의 기능이 아니라 접근성입니다. 길 위에서 사진을 찍고 너무 오래 기다려야 한다면, 모델이 정확해도 실제 제품으로는 쓰기 어렵습니다.

기준 단말로 최신 플래그십을 잡지 않았습니다. 길벗의 핵심 사용자인 어르신들은 매년 새 폰으로 갈아타는 부류가 아니라, 한 번 산 폰을 몇 년씩 그대로 쓰는 경우가 많습니다. 단말 사양 범위도 그 현실에 맞게 보수적으로 잡아야 했고, 상한선을 Galaxy S23(2023), 하한선을 Galaxy S10e(2019)로 두었습니다.

### 6.1 MTP 연결 (fork patches 1–3)

Gemma 4의 MTP(Multi-Token Prediction)는 작은 drafter가 여러 token을 먼저 제안하고 main model이 병렬로 검증하는 speculative decoding 방식입니다.

배포된 `libLiteRtLm.so`(native-v0.10.2-b)는 이미 `litert_lm_engine_settings_set_enable_speculative_decoding`을 export하고 MTP runtime도 bundle하고 있습니다. `flutter_gemma` 0.14.5가 단지 이를 호출하지 않을 뿐입니다. Fork에 작은 3개 edit(purely additive — [`patches/README.md`](../patches/README.md) 참고)를 더해 헤더에 심볼을 선언하고, FFI binding을 추가하고, backend-aware default로 `initialize()`에서 호출하게 했습니다.

```dart
bool? enableSpeculativeDecoding,   // null → backend 별 auto
...
final enableSpec = enableSpeculativeDecoding ?? (backend != 'cpu');
b.litert_lm_engine_settings_set_enable_speculative_decoding(
    settings, enableSpec);
```

Native bump는 필요 없었습니다. 이 surface에선 0.10.2-b prebuilt가 0.11.0-b와 C-API 심볼 동일입니다.

### 6.2 S23 GPU — MTP 이득 (~1.5×), 그리고 더 못 간 이유

S23 A/B(demo-normal flow, 같은 13-rule prompt + 1280px 입력):

| | baseline | MTP on |
|---|---:|---:|
| prefill | ~4.9초 | ~5.06–6.53초 |
| decode | ~17.8초 | ~11.34–11.80초 |
| chunks | ~128 | ~55–65 |
| total | ~22초 | ~16.4–18.3초 |

Decode 17.8초 → ~11.5초는 **1.5–1.6×**입니다. chunk 개수가 절반으로 줄어든 걸 보면 MTP가 chunk당 약 2 토큰을 묶고 있습니다. 품질은 유지됩니다 (speculative decoding은 정의상 lossless).

Google이 공개한 Gemma 4 MTP의 base 모델 stock target은 약 **2.87×** 이고, vLLM의 QLoRA + MTP 측정은 LoRA가 적용됐을 때 **stock speedup의 약 92%가 유지**된다고 보고합니다. 즉 fine-tuned 모델이라도 적절한 setup이면 이만큼 떨어지지 않습니다. 우리는 base-only 대비 acceptance가 거의 **절반** 으로 떨어졌고, 그 격차의 원인이 뭔지 한 번은 들여다보고 싶었습니다.

S23 GPU + MTP 위에서 4-way controlled 측정 (각 row는 같은 demo flow의 vision 사진들만 평균, acceptance length는 `chunks/sec ÷ sequential baseline ~5.6 tok/sec` 에서 역산):

| Setup | 모델 파일 | Sidecar | Vision chunks/sec | Avg total | Implied acceptance |
|---|---|---|---:|---:|---|
| Base only | `gemma-4-E2B-it.litertlm` | — | 14.4 | 8.9초 | ~2.6 tok/step |
| Graft only (sidecar 제거) | graft `.litertlm` | — | 7.0 | 14.2초 | ~1.25 tok/step |
| Graft + sidecar (production) | graft | alpha 8 | 7.4 | 13.2초 | ~1.3 tok/step |
| Base + sidecar | base | alpha 8 | 7.4 | 13.8초 | ~1.3 tok/step |

마지막 세 row가 거의 평평한 게 핵심 발견입니다. **LoRA path가 어떤 형태로든 — baked-in graft, runtime sidecar, 둘 다 — 활성화되는 순간 chunks/sec가 base-only의 절반 수준으로 떨어지고, 어느 path를 쓰든 결과가 비슷합니다.** 경로 자체가 아니라 LoRA의 *존재* 가 영향을 줍니다.

처음에는 이걸 "`tf_lite_mtp_drafter`에 LoRA가 없으니 drafter의 예측이 LoRA-적용된 main LM과 어긋난다" 정도로 정리했었는데, 이건 약간 oversimplify한 framing 입니다. Gemma 4의 drafter는 fully separate model이 아니라 main LM과 input embedding table, KV cache를 공유하고, 핵심적으로 main LM의 final-layer activation에 *conditioning* 됩니다 (drafter input ≈ token embedding + main LM final hidden state → down-projection → drafter 자체 transformer layer). Drafter 자신의 ~42MB weight는 untuned이지만, LoRA가 activation에 미친 영향은 conditioning 채널을 통해 drafter까지 propagate 됩니다 — 그래서 vLLM이 collapse가 아니라 92% 유지를 봅니다.

결국 진짜 질문은 "drafter에 LoRA가 없어서" 가 아니라 "왜 vLLM의 일반적 setup은 92%를 유지하는데 우리 specific LoRA 구성은 acceptance가 ~2.6 → ~1.3 으로 절반이 되는가" 입니다. 직접 검증은 안 됐지만, 네 가지 가설이 있습니다.

1. **LoRA magnitude와 coverage.** 우리 sidecar는 full-LM `atten_mlp_alpha8` adapter 입니다 — 모든 attention layer, 모든 MLP layer, alpha 8. 일반 PEFT QLoRA case들은 더 좁고 alpha도 낮습니다. activation drift가 클수록 drafter의 conditioning input이 pretraining 시 본 분포 범위에서 멀어집니다.
2. **Cross-task generalization.** LoRA는 한국어 audio transcription용으로 학습됐는데, 이 측정은 vision 사진 benchmark에서 영어 길안내 instruction을 생성하는 상황입니다. Drafter는 한국어 오디오 LoRA의 분포 신호가 영어 vision prompt에 얹힌 hidden state를 보게 되는데, 이 조합은 drafter pretraining에도, LoRA training에도 들어 있지 않습니다.
3. **Patcher × runtime conditioning 상호작용.** 우리 `patch_gemma4_lora_requant.py`는 `prefill_decode`와 `audio_encoder_hw`를 patch합니다. LiteRT-LM의 `SetScopedLoraFile` runtime 경로가 Google의 (미공개) audio-LoRA exporter와 drafter conditioning에 동일하게 interact 하는지는 확인되지 않았습니다. Patcher의 same-grid INT4 re-quantization 자체가 근사이고, conditioning 채널에 누적되는 numerical drift가 acceptance를 더 깎을 수 있습니다.
4. **Graft baseline 측정 부재.** Graft 선택 sweep을 돌릴 때 chunks/sec는 측정하지 않았고 CER만 봤습니다. 다른 graft 후보들이 다르게 동작할 수 있고, 현재의 `...-qat-g11875-v2sidecar` graft가 LoRA × MTP trade-off 면 위의 유일한 점이라는 보장은 없습니다.

원인을 좁히는 데 도움이 됐을 measurement 몇 가지는 *돌리지 않은* 채로 두었습니다. per-step acceptance 직접 측정(LiteRT-LM INFO log가 accept/reject event를 노출하지 않음), lower-alpha sidecar(alpha 4, alpha 2), drafter-side LoRA 부착, audio-only vs LM-only LoRA가 chunks/sec에 미치는 영향 분리, 같은 모델에서 vLLM과 apples-to-apples 비교 등이 그것입니다.

따라서 S23의 1.5–1.6× 는 구조적 상한이라기보다 우리의 현재 LoRA 구성이 이 specific 배포 경로 위에서 내는 수치입니다. vLLM의 92% 유지 수치와의 격차는 분명히 있고, 이건 "해결됐다" 가 아니라 계속 손볼 문제로 남겨두고 있습니다.

### 6.3 S10e CPU — MTP가 손해

S10e CPU+XNNPACK에서 MTP가 처음 돌았을 때 사진당 total이 50~70초였습니다. 2-step A/B로 사진당 시간을 절반으로 줄였습니다.

```text
Baseline (MTP on, English DEMO prompt 3.74 KB):
  prefill avg 40.8초 + decode avg 21.9초 = 62.7초/사진
  decode rate 2.7 chunks/sec

Step A — CPU backend에서 MTP auto-disable
  prefill avg 30.5초 + decode avg 10.0초 = 40.5초/사진
  decode rate 9.5 chunks/sec
  Δ = -22.2초/사진 (-35%)

Step B — prompt에서 [Good example] / [Bad example] block 제거
  prefill avg 21.4초 + decode avg 8.9초 = 30.2초/사진
  decode rate 9.7 chunks/sec
  누적 Δ = -32.5초 (-52%, 62.7 → 30.2)
```

진단은 이렇습니다. CPU에서는 MTP drafter의 cycle당 forward-pass가 acceptance gain으로 상쇄되지 않습니다. GPU에서는 drafter가 빠르고 accepted draft가 target 연산을 건너뛰어 net 1.5×가 나오지만, CPU에서는 두 cost가 같은 산술 단위로 떨어지고 거기에 LoRA-drafter mismatch가 acceptance를 낮춰서 net negative가 됩니다.

### 6.4 최종 실기기 수치

| 단말 | Backend | MTP | 사진당 |
|---|---|---|---:|
| Galaxy S23 | GPU (Adreno 740) | on | ~12초 |
| Galaxy S10e | CPU (XNNPACK) | off (auto) | ~50초 |

Prefill은 prompt size에 scale 합니다(English DEMO prompt ~3.3KB / ~900 text token + 768px 기준 ~256 visual token). Decode는 output length에 scale 합니다(사진당 typical 50~105 chunk).

정책은 단순합니다. 가능하면 GPU + MTP, 오래된 단말에서는 CPU-safe 경로로 느려도 끝까지 완주.

---

## 7. Demo APK — 무엇을 고정했고 무엇을 실제로 돌렸나

해외 심사자가 한국어 발화자, 한국 지도 API 키, 한국 GPS, 한국 지하철 환경을 한꺼번에 갖추기는 어렵습니다. 그래서 별도 Demo APK가 유일한 제출 경로였습니다.

원칙은 단순합니다. **단말 위의 Gemma 4 모델은 모든 사진·음성 입력에 대해 실제로 돈다.** 한국 영토 입력(지도 API, live GPS) 만 pre-bake.

| Build | Package id | Role |
|---|---|---|
| Judge Demo | `com.psymon.gilbeot.demo` | 해외 심사 |
| Korea Production | `com.psymon.gilbeot.real` | 한국 실 사용자 |

### 7.1 고정한 것 vs 실제로 도는 것

| 구성요소 | Judge mode | Korea production |
|---|---|---|
| Gemma 4 audio (STT) | **Real** (bundled WAV → 실 모델) | Real (mic → 실 모델) |
| Gemma 4 vision | **Real** (bundled photo → 실 모델) | Real (camera → 실 모델) |
| Gemma 4 text generation | **Real** | Real |
| 목적지 → place lookup | hardcoded (송파보건소) | live T-Map POI API |
| Routing | cached T-Map polyline (asset) | live ODsay + T-Map API |
| Origin (현재 위치) | 사진별 EXIF GPS, stepwise | live Geolocator GPS |
| Off-route guard | EXIF distance vs polyline | live GPS Haversine |
| 도착 감지 | model `is_arrival` + 마지막 사진 fallback | model + GPS Haversine ≤ 30m |

한국 production 빌드(`com.psymon.gilbeot.real`)는 이런 대체가 없고 끝까지 live로 돌아갑니다.

### 7.2 BYO photo flow와 EXIF gate

합리적인 reviewer라면, 단말 Gemma 4 vision 모델이 *진짜로* 일반화하는지, 아니면 prompt engineering 중에 모델이 본 사진만 cherry-pick 했는지를 물을 수 있습니다.

DEMO_MODE에서 카메라 버튼을 길게 누르면 system gallery가 열립니다. 고른 사진은 canned demo와 *같은* 단말 vision 경로를 탑니다. 특수 tuning도, asset overlap도 없습니다. 심사자는 폰에 있는 무엇이든 넣어 봐서 모델이 4-photo lookup table이 아니라는 점을 확인할 수 있습니다.

임의 사진은 그냥 두면 모델이 hardcoded 송파보건소 목적지로 방향을 상상하게 만들 수 있습니다("walk left to exit 10"). 사진이 다른 도시에서 찍혔어도 마찬가지입니다. 그래서 EXIF GPS를 추출하고 계획된 T-Map polyline과의 최소 수직 거리를 계산해서, 사진별 step context를 분기합니다.

| EXIF 거리 (경로 대비) | Step context branch |
|---|---|
| ≤ 200m | "on or near the route — 짧은 장면 설명 + 간단한 Exit 10 안내" |
| > 200m | "NOT on the route — 필수 prefix 문장 + ONE describing 문장 + STOP. FORBIDDEN: `follow`, `walk toward`, `go to`, …" |
| 없음 | "generic — describe only, 목적지 방향 금지" |

Off-route branch는 중립적인 example JSON(`["bench", "tree", "grass"]` + "wooden bench under a leafy tree")을 ship 합니다. 이 어휘가 subway/landmark 사진 응답으로 새지 않도록 의도적으로 고른 것입니다. 모델은 example을 복사하지 못하고 실제로 vision을 해야 합니다.

---

## 8. References

- AI Hub — [명령어 음성(노인남녀)](https://www.aihub.or.kr/aihubdata/data/view.do?aihubDataSe=data&dataSetSn=94)
- Kamath, Hessel, Chang. [*What's "up" with vision-language models? Investigating their struggle with spatial reasoning*](https://aclanthology.org/2023.emnlp-main.568/). EMNLP 2023.
- Hoehing, Rushe, Ventresque. [*What's left can't be right — The remaining positional incompetence of contrastive vision-language models*](https://arxiv.org/abs/2311.11477).
- Cactus byte-fallback detokenization fix: [cactus-compute/cactus#635](https://github.com/cactus-compute/cactus/pull/635)
- LiteRT-LM v0.11.0 release notes (Gemma 4 MTP support): https://github.com/google-ai-edge/LiteRT-LM/releases/tag/v0.11.0
- LiteRT-LM C++ API (speculative decoding on mobile): https://ai.google.dev/edge/litert-lm/cpp
- 영문 버전: [`technical_report.md`](technical_report.md)
- 이 리포트가 참조하는 in-repo 문서: [`tools/README.md`](../tools/README.md), [`patches/README.md`](../patches/README.md), [`docs/architecture.md`](architecture.md).
