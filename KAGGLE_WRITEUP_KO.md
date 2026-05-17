# 길벗(Gilbeot) — 기계 좌표를 사람의 길안내로 바꾸는 온디바이스 AI

> **TL;DR.** 길벗은 고령 사용자를 위한 온디바이스 멀티모달 AI 길안내 안드로이드 앱입니다. 한국어로 목적지를 말하면 Gemma 4 audio가 사용자 발화를 분석해 경로를 만들고, 길에서 찍은 사진을 Gemma 4 vision이 읽어 "노란 10번 출구 표지판 아래에서 왼쪽으로" 같은 안내문으로 바꿔 TTS로 들려줍니다. 음성·사진·안내 생성 모두 단말 안의 Gemma 4 E2B 단일 모델에서 처리되어 네트워크 없이 동작하고, 음성·사진·위치는 서버로 나가지 않습니다. Galaxy S23(GPU+MTP, ~12초/사진)·S10e(CPU, ~50초/사진) 실기기 검증. 기술 디테일은 §5, 해외 심사 흐름은 §2.두 가지 APK 참고.

## 1. Motivation

고령자가 길을 잃는 순간은 지도 앱이 없는 순간이 아니라, 지도 앱의 말이 내 눈앞의 장면과 연결되지 않는 순간입니다.

지도와 내비게이션은 보통 좌표계와 도로를 기준으로 말합니다.

- "250m 앞에서 우회전"
- "삼거리에서 2시 방향"
- "북동쪽으로 이동"

하지만 많은 어르신에게 실제 길은 그런 방식으로 보이지 않습니다. 길은 "보이는 노란 표지판 아래", "약국 지나서 오른쪽", "계단 손잡이가 있는 쪽", "건물 입구 위 간판"처럼 **눈앞의 사물과 장면**으로 이해됩니다.

길벗은 이 차이를 줄이기 위해 만들었습니다.

> **길벗은 기계 좌표계를 인간 좌표계로 번역하는 온디바이스 AI 길안내 도우미입니다.**

우리가 줄이고 싶었던 간극은 단지 지도 앱 사용법의 간극이 아닙니다. 최신 멀티모달 AI는 음성, 이미지, 텍스트를 한 번에 이해할 수 있는 단계까지 왔지만, 그 기술이 어르신의 실제 외출 순간에는 아직 자연스럽게 닿지 않습니다. 길벗은 Gemma 4를 어르신이 쓰는 언어와 장면 위에 올려, 최신 AI 기술과 일상 이동 경험 사이의 거리를 줄이려는 시도입니다.

가상의 사용자는 이렇게 움직입니다. 사용자가 목적지를 말하면 앱이 경로를 만듭니다. 이동 중 헷갈리는 곳에서 사진을 찍으면 Gemma 4가 눈앞의 장면을 읽어, 어르신이 바로 이해할 수 있는 말로 바꿔 줍니다. 모든 핵심 추론은 단말 안에서 끝납니다.

## 2. Solution Approach

### 제품 흐름

```text
목적지 음성 입력
  → Gemma 4 audio가 한국어 발화 인식
  → T-Map / ODsay / Naver Map으로 경로 생성
  → 사용자가 헷갈리는 지점에서 사진 촬영
  → Gemma 4 vision이 표지판, 출구 번호, 화살표, 건물 입구를 해석
  → "노란 10번 출구 표지판 아래에서 왼쪽으로 가세요"처럼 말로 안내
  → GPS와 경로 polyline을 비교해 경로 이탈 여부를 확인
  → 경로에서 벗어나면 목적지 안내를 상상하지 않고 재탐색/재촬영을 요청
  → TTS로 읽어 줌
```

경로 이탈 처리는 제품 흐름에서 중요한 안전 장치입니다. 사용자가 계획된 polyline에서 멀어진 상태로 사진을 찍으면, 모델이 엉뚱한 사진을 보고도 "계속 가세요"라고 상상할 수 있습니다. 길벗은 production build에서 현재 GPS와 도보 경로 segment의 Haversine 거리를 계산해, 경로에서 벗어난 경우 안내 생성을 제한하고 재탐색 또는 새 사진 촬영을 요청합니다.

### 왜 온디바이스인가

이 서비스가 필요한 순간은 지하철역, 지하상가, 병원 주변, 버스정류장처럼 네트워크가 불안정하거나 주변이 복잡한 곳입니다. 또 음성과 사진은 사용자의 현재 위치와 이동 상황을 드러내는 민감한 데이터입니다.

그래서 길벗은 핵심 모델 추론을 서버가 아니라 Android 단말에서 실행합니다.

| 항목 | 길벗의 선택 |
|---|---|
| 음성 인식 | Gemma 4 audio를 단말에서 실행 |
| 사진 이해 | Gemma 4 vision을 단말에서 실행 |
| 안내문 생성 | Gemma 4 text generation을 단말에서 실행 |
| 지도/경로 | 한국 production은 T-Map, ODsay, Naver Map API 사용 |
| 개인정보 | 음성/사진/장면 맥락을 모델 서버로 보내지 않음 |

Android를 우선한 이유도 분명합니다. 한국갤럽의 2025년 스마트폰 조사에서 60대이상 고령자의 92%가 삼성 갤럭시를 사용 중이었습니다. 70대 이상에서는 갤럭시 점유율이 82%로 낮아지지만 여전히 압도적입니다. 즉 한국 고령 사용자를 위해 실기기에서 검증해야 한다면, 첫 번째 기준점은 iOS가 아니라 삼성 갤럭시 Android 단말입니다. [한국갤럽 2025 스마트폰 관련 조사](https://www.gallup.co.kr/dir/GallupReport/GallupReport%2820250707%29_%EC%8A%A4%EB%A7%88%ED%8A%B8%ED%8F%B0.pdf)

### 두 가지 APK

한국 production 앱은 한국 지도 API와 실제 위치를 사용합니다. 하지만 Kaggle 심사자는 한국어 발화자, 한국 지도 API 키, 한국 위치, 한국 지하철 환경을 갖고 있지 않을 수 있습니다. 그래서 별도 Demo APK를 만들었습니다.

| 빌드 | 패키지 ID | 역할 |
|---|---|---|
| Judge Demo | `com.psymon.gilbeot.demo` | 해외에서도 동일 시나리오 재현 |
| Korea Production | `com.psymon.gilbeot.real` | 한국 지도 API와 실제 마이크/카메라 사용 |

Demo APK는 경로와 일부 입력을 고정하지만, 핵심 AI 기능은 실제로 실행합니다. 마이크 데모의 WAV도 Gemma 4 audio가 인식하고, 네 장의 역 사진도 Gemma 4 vision이 매번 읽습니다. 즉 심사용 편의 장치가 있어도, 모델 추론은 장식이 아니라 앱 상태를 바꾸는 실제 동작입니다.

## 3. Development Process

개발 과정은 기능을 하나씩 붙이는 방식보다, **어르신이 실제로 사용할 수 있는 이동 루프**를 완성하는 방식으로 진행했습니다.

### 3.1 서비스 아키텍처

```text
Flutter UI
  ├─ 마이크 입력
  │    └─ Gemma 4 audio → 목적지 transcript
  ├─ 지도/경로
  │    ├─ T-Map POI + 도보
  │    ├─ ODsay 대중교통
  │    └─ Naver Map 표시
  └─ 카메라 입력
       └─ route/off-route context
            └─ Gemma 4 vision/text
                 └─ JSON {instruction, landmarks, is_arrival, arrow_tip_x, ...}
                      └─ bbox 좌/우 보정
                           └─ TTS 안내
```

### 3.2 단말 성능 범위 설정

고령 사용자는 매년 최신 스마트폰을 변경하는 고객군이 아닙니다. 따라서 우리 서비스의 목표 단말 설정도 보수적으로 잡아야 했습니다. 

| 항목 | Galaxy S23 | Galaxy S10e |
|---|---:|---:|
| 출시 시기 | 2023년 | 2019년 |
| 검증 역할 | 현실적 상한선 | 오래된 폰 하한선 |
| SoC | Snapdragon 8 Gen 2 Mobile Platform for Galaxy | Exynos 9820 (한국 모델 SM-G970N) |
| CPU | 64-bit octa-core, 최대 3.36GHz | octa-core, 2.73GHz Mongoose M4 + 2.31GHz / 1.95GHz Cortex |
| GPU | Adreno 740 | Mali-G76 MP12 |
| RAM | 8GB | 6GB |
| 주요 backend | GPU, Adreno 740 | CPU, XNNPACK |
| MTP 정책 | on | off |

## 4. Google AI Edge

이 프로젝트의 핵심은 Gemma 4를 단순 API가 아니라 **제품의 감각기관**으로 쓴다는 점입니다. 음성으로 목적지를 듣고, 사진으로 장면을 보고, 텍스트로 어르신에게 말합니다. 이 세 가지가 하나의 온디바이스 멀티모달 모델 안에서 이어집니다.

길벗이 Edge AI 프로젝트인 이유는 추론 위치만 단말이어서가 아닙니다. 클라우드 기반 STT나 VLM은 표면적으로 더 쉬운 선택일 수 있지만, 특정 서비스를 구독하고 고령 사용자 위치와 음성를 매번 서버로 보내는 구조는 길벗의 서비스 의도와 맞지 않습니다.

Gemma 4와 LiteRT-LM 덕분에 길벗은 다음 원칙을 지킬 수 있었습니다.

- 목적지 발화, 사진, 장면 맥락을 모델 서버로 보내지 않습니다.
- 모델 다운로드 이후 핵심 안내 루프는 오프라인에서도 돌아갑니다.
- 좋은 폰에서는 GPU/MTP로 빠르게, 오래된 폰에서는 CPU-safe 경로로 느리지만
  끝까지 동작합니다.
- 멀티모달 처리를 하나의 모델 family 안에 묶어 제품 구조를 단순하게
  유지합니다.

### 4.1 왜 Gemma 4가 최선의 선택인가

| Gemma 4 기능 | 길벗에서의 역할 |
|---|---|
| Audio understanding | 한국어 목적지 발화를 transcript로 변환 |
| Vision understanding | 사진 속 간판, 출구 번호, 화살표, 건물 입구 이해 |
| Text generation | 어르신이 이해할 수 있는 짧은 안내문 생성 |
| Native bbox-style output | 화살표 tip/tail 좌표를 받아 좌/우 보정 |
| LiteRT-LM deployment | 서버 없이 Android 단말에서 실행 |
| MTP / speculative decoding | S23 GPU에서 decode 속도 개선 |

Gemma 4가 특히 좋았던 지점은 "길안내에 필요한 입력이 원래 멀티모달"이라는 사실과 맞아떨어졌다는 점입니다. 목적지는 음성이고, 길 위의 불확실성은 사진이고, 최종 출력은 짧은 자연어 안내입니다. 여러 모델을 조합하면 각 단계의 오류와 지연이 누적되지만, Gemma 4 E2B는 같은 모델 family 안에서 이 세 가지를 처리할 수 있었습니다.

### 4.2 모델 번들 스펙

길벗이 사용하는 Gemma 4 E2B `.litertlm` bundle은 약 2.4GB입니다.

| 내부 구성요소 | 역할 | 크기 |
|---|---|---:|
| `tf_lite_vision_encoder` | 이미지 인코딩 | 약 208MB |
| `tf_lite_vision_adapter` | vision embedding adapter | 약 4.5MB |
| `tf_lite_audio_encoder_hw` | 오디오 인코딩 | 약 102MB |
| `tf_lite_audio_adapter` | audio embedding adapter | 약 9MB |
| `tf_lite_prefill_decode` | LM prefill/decode | 약 781MB |
| `tf_lite_per_layer_embedder` | MTP 조건 구성요소 | 약 1.2GB |
| `tf_lite_mtp_drafter` | speculative decoding drafter | 약 42MB |

이 구조 덕분에 하나의 모델 파일이 음성, 이미지, 텍스트를 모두 처리합니다. 반대로 말하면, 이 모델을 모바일 앱에서 실제로 쓰려면 메모리와 backend 선택을 매우 신중하게 해야 합니다.

## 5. Challenges I faced

길벗 서비스를 개발하면서 겪은 문제점과 해결 과정을 간략히 공유합니다.

### 5.1 한국어 오디오 학습이 필요했다

음성 인식은 길벗의 첫 문입니다. 목적지를 잘못 들으면 이후 모든 경로가 틀어집니다. 특히 한국어 지명, 시설명, 고령자 발화의 부정확도, 느린 속도와 불규칙한 휴지는 base 모델만으론 안정적이지 않았습니다.

내부 평가는 134개 한국어 목적지/일상 발화 샘플로 진행했습니다. 지표는 CER(Character Error Rate)입니다.

| 모델/경로 | CER | 의미 |
|---|---:|---|
| Base gemma-4-E2B-it-litert-lm(no LoRA) | 13.14% | 목적지 입력용으로 부적합 |
| 길벗 deployed LiteRT-LM path | **5.00%** | Android 배포 경로에서 사용 가능 |
| Improvement vs device base	| ~2.6 × reduction  | (−62 % relative) |

5.00%는 단순 benchmark 숫자가 아닙니다. 이 정도까지 내려와야 "송파구 보건소", "강남역", "국립중앙박물관" 같은 목적지 입력을 제품 흐름 안에서 다룰 수 있습니다.

#### 5.1.1 데이터를 얼마나, 어떻게 모았나

학습 데이터는 약 45,823개 utterance입니다 (총 50시간, 화자 단위 stratified subset; 품질 필터 후 train 43,431 / eval 2,291). 핵심 원천은 AI Hub의 **명령어 음성(노인남녀)** 데이터였습니다. 이 데이터셋은 한국인 노인남녀의 음성 명령어를 문자로 바꾸고 문맥을 이해하는 한국어 음성언어처리 기술 개발을 목표로 구축된 공개 데이터입니다.

| 데이터 원천 | 목적 |
|---|---|
| AI Hub 명령어 음성(노인남녀) | 고령 화자의 명령형 발화, 느린 말속도, 억양, 휴지 분포 반영 |
| 길벗용 선별/정규화 subset | 목적지 요청, 시설명, 짧은 길안내 의도에 맞는 utterance 중심 구성 |

길벗은 일반 받아쓰기 앱이 아니라 목적지 입력 앱이므로, 공개 데이터 안에서 "길안내 첫 문장"에 가까운 명령형/요청형 발화를 선별하고 전사 텍스트를 평가와 학습에 맞게 정규화했습니다. [AI Hub 명령어 음성(노인남녀)](https://www.aihub.or.kr/aihubdata/data/view.do?aihubDataSe=data&dataSetSn=94)

#### 5.1.2 어떻게 학습시켰나

모델은 Gemma 4 E2B이고, audio encoder와 LM 쪽 attention/MLP에 rank-8 LoRA를 적용했습니다. 학습은 Unsloth FastModel + bitsandbytes 4-bit + PEFT 경로로 진행했습니다.

중요한 세부사항은 Unsloth FastModel입니다. Gemma 4 audio 쪽에는 `Gemma4ClippableLinear` 같은 custom layer가 있고, 일반 PEFT 경로는 이 일부를 조용히 건너뛰었습니다. 겉으로는 loss가 줄어도 실제 audio 출력은 바뀌지 않는 상황이 생겼습니다. Unsloth 경로는 이 layer들을 정확히 처리했고, HF checkpoint 수준에서 CER 3.06%까지 도달했습니다.

### 5.2 모바일 탑재 시도들

학습보다 어려웠던 것은 배포였습니다. 당시 공개 toolchain은 Gemma 4 audio LoRA를 Android `.litertlm` bundle에 넣는 경로를 제공하지 않았습니다.

#### 5.2.1 `.litertlm` 공식 지원 부재와 해결법

제출 시점의 공개 경로를 확인했을 때, 상황은 다음과 같았습니다.

| 경로 | 확인한 문제 |
|---|---|
| LiteRT-LM 공식 변환기 | text-side LoRA 중심이며 audio path는 unsupported shape 문제 |
| `litert-torch-nightly` | LM export는 가능하지만 audio encoder 배포 경로가 맞지 않음 |
| MediaPipe LLM Inference | 공개 LoRA path가 attention-only/GPU runtime 중심 |
| 단순 HF adapter 배포 | Android `.litertlm` runtime에 직접 연결할 공식 방법 없음 |

그래서 직접 `.litertlm` patcher를 만들었습니다. 시도한 방식은 세 가지였습니다.

| 시도 | 방식 | 결과 |
|---|---|---|
| Fresh quantization | LoRA delta를 더한 뒤 새 scale로 int4 재양자화 | 작은 delta가 사라지고 scale mismatch로 불안정 |
| Quant-projected flip | 기존 int4 grid에서 큰 delta 방향으로 일부 값만 한 step 이동 | 일부 개선, layer별 튜닝이 불안정 |
| Same-grid requant | 기존 per-row scale 유지, dequant → delta add → 같은 scale로 requant | 최종 채택 |

최종 방식의 핵심은 scale을 새로 만들지 않는 것입니다.

```text
기존 .litertlm block:
  scale, int4_weights

patch:
  base_fp32     = dequantize(scale, int4_weights)
  patched_fp32  = base_fp32 + LoRA_delta
  patched_int4  = round(patched_fp32 / scale)   # 기존 scale 그대로 사용
  patched_int4  = clip(patched_int4, -8, 7)
  write back
```

이렇게 하면 살아남는 변화는 LoRA 방향으로만 남고, runtime kernel이 기대하는 quantization grid를 최대한 보존할 수 있습니다.

최종 deployed path는 두 부분으로 나뉩니다.

| 부분 | 방식 |
|---|---|
| LM 쪽 | `.litertlm` 안의 LM weight에 graft patch 적용 |
| audio 쪽 | alpha-8 LoRA sidecar를 runtime `loraPath`로 연결 |

audio encoder 자체를 `.litertlm` 안에서 과하게 건드리면 Adreno 740 prefill kernel에서 불안정해졌습니다. 그래서 audio-side delta는 sidecar로 분리하고, LiteRT-LM session config에 연결하는 경로를 추가했습니다.

#### 5.2.2 Cactus의 우회법과 INT4/INT8 한계

Cactus는 `.litertlm`을 거치지 않는 별도 경로입니다. 우리가 만든 patcher처럼 LiteRT-LM 바이너리 안의 양자화 weight를 직접 건드리는 대신, Cactus는 자체 모바일 추론 엔진을 두고 Gemma 4 모델을 자기 포맷으로 다시 변환합니다. 즉 같은 "Android에서 Gemma 4 audio를 돌린다"는 문제를, 우리는 기존 runtime을 그대로 둔 채 weight만 patch하는 쪽으로 풀었고 Cactus는 runtime 자체를 갈아끼우는 쪽으로 풀었습니다.

정확도만 보면 Cactus는 꽤 매력적이었습니다.

| backend | CER | S10e median/sample | 판단 |
|---|---:|---:|---|
| HF merged reference | 3.619% | 0.691초 | 클라우드 참고선 |
| LiteRT best, graft + alpha-8 sidecar | 5.003% | 12.48초 | production 채택 |
| Cactus INT8 | 3.949% | 41.94초 | 정확도 우위, 3.4배 느림 |
| Cactus INT4 | 4.856% | 45.32초 | 더 작지만 latency 더 나쁨 |

문제는 지연시간이었습니다. S10e에서 LiteRT 경로는 마이크 입력 1회 기준 prefill 11,992ms + decode 3,389ms = total 15,381ms로 이미 빠르다고 말하기 어려운 수치였는데, 같은 환경에서 Cactus는 sample당 42~45초가 나왔습니다. 어르신이 목적지를 말하고 안내가 시작되기까지 기다릴 수 있는 범위를 넘었습니다.

그래서 최종 submission 앱은 LiteRT-LM `.litertlm` + LoRA sidecar 경로를 유지했습니다. 단, 이 결론은 우리가 검증한 두 단말 S10e(2019)/S23(2023)에 한정된 판단입니다. 더 강력한 컴퓨팅 파워를 쓸 수 있는 환경 — 예컨대 최신 플래그십이나 NPU 가속이 풍부한 단말 — 에서는 Cactus의 정확도 우위는 그대로 살아남으면서 latency 우려는 줄어들 수 있고, 그 경우 Cactus는 여전히 좋은 선택입니다.

한 가지 주의할 점은, 위 정확도 수치는 Cactus의 byte-fallback detokenization 버그를 수정한 뒤의 값이라는 것입니다. Cactus는 Gemma 4가 한국어처럼 SentencePiece 단일 piece에 없는 codepoint를 emit할 때 사용하는 `<0xEA><0xB9><0xB0>` 같은 byte-fallback 토큰을 UTF-8로 재조합하지 않고 literal 문자열로 흘려, "잠 깰 수 있는" 같은 문장을 "잠 `<0xEA><0xB9><0xB0>` 수 있는"으로 출력했습니다. 이 한 줄짜리 detokenization 버그가 134-sample 한국어 평가에서 INT8 CER을 4.638%에서 3.949%로, INT4를 5.545%에서 4.856%로 끌어내렸습니다. 수정은 [cactus-compute/cactus#635](https://github.com/cactus-compute/cactus/pull/635)에 PR로 올려두었습니다.

### 5.3 VLM의 좌/우 판별 한계와 bbox 해결법

길벗의 사진 안내는 "사진을 설명하는 기능"이 아니라 "지금 어디로 가야 하는지 말해 주는 기능"입니다. 그래서 모델은 예쁘게 caption을 쓰는 것보다, 눈앞의 실제 단서를 안전하게 읽어야 합니다.

문제는 VLM이 left/right 등 기본 공간해석에 취약하다는 것입니다. 이는 작은 모델만의 문제가 아니라 VLM 전반에서 관찰되는 약점입니다. Kamath, Hessel, Chang의 EMNLP 2023 논문 **"What's 'up' with vision-language models? Investigating their struggle with spatial reasoning"**는 18개 vision-language model을 평가했고, 모두 기본 공간 관계에서 낮은 성능을 보였다고 보고합니다. [ACL Anthology](https://aclanthology.org/2023.emnlp-main.568/)

좌/우 문제에 더 직접적인 근거도 있습니다. Hoehing, Rushe, Ventresque의 **"What's left can't be right -- The remaining positional incompetence of contrastive vision-language models"**는 CLIP 계열 모델의 simple left-right positional relation을 집중적으로 분석하며, 대규모 데이터셋에서도 이 실패 양상이 예측 가능하다고 설명합니다. [arXiv:2311.11477](https://arxiv.org/abs/2311.11477)

Gemma 4도 출구 번호와 표지판은 잘 읽었지만, 가로 화살표를 보고 "left"와 "right"를 가끔 뒤집었습니다. 어르신 길안내에서 이는 작은 오류가 아닙니다.

해결은 Gemma 4의 native bbox-style output을 활용하는 것이었습니다. 응답 JSON에 다음 두 필드를 추가했습니다.

```json
{
  "arrow_tip_x": null,
  "arrow_tail_x": null
}
```

가로 화살표가 보이면 모델은 tip과 tail의 x 좌표를 0~1 범위로 냅니다. 앱은 이 숫자를 비교합니다.

```text
tip_x < tail_x  →  왼쪽 화살표
tip_x > tail_x  →  오른쪽 화살표
```

그리고 모델의 안내문이 픽셀 좌표와 반대 방향을 말하면 앱이 단어를 교정합니다. 이는 답을 prompt에 심는 방식이 아닙니다. 모델에게는 localization을 맡기고, 좌/우 판정은 deterministic code가 수행하는 방식입니다.

마지막으로 같은 사진을 좌우 반전해 검증했습니다. 원본에서는 왼쪽 화살표, 반전본에서는 오른쪽 화살표로 좌표와 문장이 함께 바뀌었습니다. 모델이 상수 답을 외운 것이 아니라 이미지를 실제로 보고 있음을 확인한 것입니다.

### 5.4 추론 속도를 높이기 위한 노력들

속도는 편의 기능이 아니라 접근성입니다. 길 위에서 사진을 찍고 너무 오래 기다려야 한다면, 모델이 정확해도 실제 제품으로 쓰기 어렵습니다.

우리는 최신 최고 사양 폰을 기준으로 잡지 않았습니다.

| 기준 | 기기 | 목표 |
|---|---|---|
| 상한선 | Galaxy S23, 2023년형 | 최신폰이 아니어도 쾌적한 경험 |
| 하한선 | Galaxy S10e, 2019년형 | 오래된 폰에서도 끝까지 동작 |

Gemma 4의 MTP(Multi-Token Prediction)는 작은 drafter가 여러 token을 먼저 제안하고 main model이 검증하는 speculative decoding 방식입니다. GPU에서는 품질 손실 없이 decode 시간을 줄일 수 있습니다.

S23에서는 MTP가 확실한 이득이었습니다.

| 항목 | baseline | MTP on |
|---|---:|---:|
| decode 시간 | 약 17.8초 | 약 11.5초 |
| speedup | - | 약 1.5~1.6배 |
| 품질 | 유지 | 유지 |

하지만 S10e CPU에서는 반대였습니다. CPU에서는 drafter와 main model이 같은 연산 자원을 나눠 쓰고, fine-tune된 main LM 과 base drafter 사이의 분포 차이로 acceptance rate도 낮아집니다. 그래서 MTP overhead가 이득보다 컸습니다.

최종 정책:

| Backend | MTP 기본값 | 이유 |
|---|---|---|
| GPU/NPU | on | S23에서 quality-neutral speedup |
| CPU | off | S10e에서 MTP-off가 더 빠름 |

최종 실기기 성능은 다음과 같습니다.

| 기기 | Backend | MTP | 사진 안내 시간 | 사용성 |
|---|---|---|---:|---|
| Galaxy S23 | GPU | on | 약 12초 | 데모/실사용 모두 쾌적 |
| Galaxy S10e | CPU | off | 약 50초 | 느리지만 오래된 폰에서도 완주 가능 |

### 5.5 심사용 고정 흐름과 정직성

ODsay, T-Map, Naver Map은 한국 환경에 강하게 묶여 있습니다. 해외 심사자가 한국 API 키, 한국 GPS, 한국 지하철 사진, 한국어 발화를 모두 준비하기는 어렵습니다.

해결책은 Demo APK였습니다. 단, 단순 scripted demo가 아니라 다음 원칙을 지켰습니다.

| 항목 | Demo APK 처리 |
|---|---|
| 경로 | 한국에서 미리 만든 cached route 사용 |
| 음성 입력 | bundled Korean WAV 사용 |
| STT | Gemma 4 audio가 실제 인식 |
| 사진 안내 | Gemma 4 vision이 실제 생성 |
| 목적지 검증 | transcript가 Songpa Health Center를 가리킬 때만 route 시작 |
| 경로 밖 사진 | long-press gallery flow에서 EXIF 정보로 GPS동작 대체 |

카메라 버튼을 길게 누르면 폰의 갤러리가 열립니다. 무엇이든 고를 수 있고, 골라낸 이미지는 캔드 데모와 같은 온디바이스 Gemma 4 vision path를 탑니다. EXIF GPS가 계획 경로에서 멀면 앱은 안내를 상상하지 않고 "이 사진은 계획된 경로 위로 보이지 않는다"는 식으로 설명만 합니다. 이 기능은 demo가 네 장의 고정 사진 lookup table이 아니라 실제 vision 모델 흐름이라는 점을 보여줍니다.

## 6. Future Work

길찾기는 시작점입니다. 길벗의 더 큰 목적지는 고령 사용자와 AI 기술 사이에 있는 장벽을 허무는 것입니다.

고령 사용자가 스마트폰 앞에서 겪는 어려움은 길찾기에만 머물지 않습니다. 폰뱅킹, 공문서 처리, 병원 예약, 약 복용 안내, 복지 서비스 신청, 가족에게 보낼 사진과 문서 정리처럼 생활의 중요한 문제들이 점점 스마트폰 안으로 들어오고 있습니다. 이 문제들은 대부분 개인정보, 건강정보, 금융정보를 포함하므로 클라우드로 아무렇게나 보낼 수 없습니다. 반드시 온디바이스 환경에서 처리해야 하는 이유가 있습니다.

다음 단계의 길벗은 다음 방향으로 확장할 수 있습니다.

| 영역 | 미래 과제 |
|---|---|
| 길찾기 | 더 넓은 지역, 실시간 경로 이탈 재탐색, 현장 사용자 테스트 |
| 폰뱅킹 보조 | 화면을 읽고 "이 버튼을 누르세요"가 아니라 위험한 이체/피싱을 먼저 막는 온디바이스 안내 |
| 공문서 처리 | 주민센터/건강보험/복지 문서를 사진으로 읽고 쉬운 말로 요약 |
| 병원 예약 | 병원 앱, 문자, 예약 확인서를 읽고 일정과 준비물을 단말 안에서 정리 |
| 가족 연결 | 자녀에게 보낼 현재 위치/상황 설명을 사용자가 승인한 뒤 생성 |

이 프로젝트에서 길안내를 선택한 이유는 길 위의 문제가 가장 눈에 잘 보였기 때문입니다. 하지만 진짜 문제는 "AI가 할 수 있는 일"과 "어르신이 실제로 스마트폰에서 할 수 있는 일" 사이의 간극입니다. 길벗은 그 간극을 줄이는 온디바이스 AI 인터페이스의 첫 번째 사례입니다.

## 7. Thanks

이 글의 마지막은 특별한 감사 인사로 채우려 합니다.

누구보다 적극적으로 제품 테스트에 참여해 주시고 데모 영상 촬영에도 기꺼이 응해 주신 장인어른, 당신이 계셔서 개발을 끝까지 마칠 수 있었습니다. 길벗을 개발하는 동안 늘 곁에서 귀중한 피드백을 주시고, 변함없이 응원해 주셔서 진심으로 감사드립니다.

정년퇴직 후 새로운 삶을 시작하시는 당신의 앞길에 늘 행복이 가득하시기를.

## 8. License

This Writeup has been released under the Attribution 4.0 International (CC BY 4.0) license.

## 9. Citation

Psymon. 길벗(Gilbeot) — 고령 사용자를 위한 온디바이스 멀티모달 AI 길안내 앱. https://www.kaggle.com/competitions/gemma-4-good-hackathon/writeups/gilbeot. 2026. Kaggle.
