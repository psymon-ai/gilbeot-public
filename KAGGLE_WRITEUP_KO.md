# 길벗(Gilbeot) — 고령 사용자를 위한 온디바이스 멀티모달 AI 길안내 앱

> **TL;DR.** 길벗은 고령 사용자를 위한 온디바이스 멀티모달 AI 길안내 안드로이드 앱입니다. 한국어로 목적지를 말하면 Gemma 4 audio가 발화를 인식하고 앱이 경로를 만들며, 길에서 찍은 사진을 Gemma 4 vision이 읽어 "노란 10번 출구 표지판 아래에서 왼쪽으로" 같은 안내문으로 바꿔 TTS로 들려줍니다. 모델 다운로드 이후 음성·사진·안내 생성은 단말 안의 Gemma 4 E2B 단일 모델에서 처리되고, 음성·사진·위치는 서버로 나가지 않습니다. Galaxy S23(GPU+MTP, ~12초/사진)·S10e(CPU, ~50초/사진) 실기기 검증.

## 1. Motivation

고령자가 길을 잃는 순간은 지도 앱이 없는 순간이 아니라, 지도 앱의 말이 내 눈앞의 장면과 연결되지 않는 순간입니다. 지도와 내비게이션은 "250m 앞에서 우회전", "삼거리에서 2시 방향" 같은 좌표·도로 기준으로 말하지만, 어르신에게 길은 "보이는 노란 표지판 아래", "약국 지나서 오른쪽", "계단 손잡이가 있는 쪽"처럼 **눈앞의 사물과 장면**으로 이해됩니다.

> **길벗은 기계 좌표계를 인간 좌표계로 번역하는 온디바이스 AI 길안내 도우미입니다.**

최신 멀티모달 AI는 음성·이미지·텍스트를 한 번에 이해할 수 있는 단계까지 왔지만, 그 기술이 어르신의 실제 외출 순간에는 아직 자연스럽게 닿지 않습니다. 길벗은 Gemma 4를 어르신이 쓰는 언어와 장면 위에 올려, 최신 AI와 일상 이동 경험 사이의 거리를 줄이려는 시도입니다.

## 2. Solution Approach

```text
목적지 음성 → Gemma 4 audio → 한국어 transcript
            → T-Map / ODsay 경로 생성 + Naver Map 표시
            → 헷갈리는 지점에서 사진 촬영
            → Gemma 4 vision → 표지판/출구번호/화살표/입구 해석
            → "노란 10번 출구 표지판 아래에서 왼쪽으로" 안내문
            → GPS-경로 polyline Haversine 비교로 이탈 감지
            → 이탈 시 안내 생성 금지 + 재탐색/재촬영 요청
            → TTS 음성 안내
```

경로 이탈 처리는 중요한 안전 장치입니다. 사용자가 계획된 polyline에서 벗어난 채로 사진을 찍으면 모델이 엉뚱한 사진을 보고도 "계속 가세요"라고 상상할 수 있어, production은 현재 GPS와 도보 segment 사이 Haversine 거리를 계산해 이탈 시 안내 생성을 막고 재탐색을 요청합니다.

### 왜 온디바이스인가

이 서비스가 필요한 순간은 지하철역·지하상가·병원 주변처럼 네트워크가 불안정하거나 주변이 복잡한 곳입니다. 또 음성과 사진은 사용자 위치·이동 상황을 드러내는 민감 데이터입니다. 한국갤럽 2025 조사에서 60대는 92%, 70대 이상은 82%가 Galaxy를 사용 — 한국 고령 사용자를 위한 첫 기준점은 iOS가 아닌 Android입니다. [한국갤럽 2025](https://www.gallup.co.kr/dir/GallupReport/GallupReport%2820250707%29_%EC%8A%A4%EB%A7%88%ED%8A%B8%ED%8F%B0.pdf)

### 두 가지 APK

한국 production은 한국 지도 API와 실제 위치가 필요해 해외 심사자가 그대로 시도하기 어렵습니다. Demo APK는 시나리오를 고정하지만 모든 사진·음성에 대해 단말 내 Gemma 4 추론은 실제로 매번 실행됩니다.

| 빌드 | Package ID | 용도 |
|---|---|---|
| Judge Demo | `com.psymon.gilbeot.demo` | 해외 심사자 |
| Korea Production | `com.psymon.gilbeot.real` | 한국 실 사용자 |

## 3. Development Process

검증 단말은 보수적으로 잡았습니다. 한국 고령자는 매년 새 폰을 사는 고객군이 아닙니다.

| 항목 | Galaxy S23 | Galaxy S10e |
|---|---:|---:|
| 출시 | 2023년 | 2019년 |
| 역할 | 현실적 상한선 | 오래된 폰 하한선 |
| SoC | Snapdragon 8 Gen 2 for Galaxy | Exynos 9820 |
| GPU | Adreno 740 | Mali-G76 MP12 |
| 주요 backend | GPU | CPU (XNNPACK) |
| MTP 정책 | on | off |

전체 아키텍처: [`docs/architecture.md`](https://github.com/psymon-ai/gilbeot-public/blob/main/docs/architecture.md).

## 4. Google AI Edge

이 프로젝트의 핵심은 Gemma 4를 단순 API가 아닌 **제품의 감각기관**으로 쓴다는 점입니다. 음성으로 목적지를 듣고, 사진으로 장면을 보고, 텍스트로 어르신에게 말합니다. 세 가지가 하나의 온디바이스 멀티모달 모델 안에서 이어집니다.

| Gemma 4 기능 | 길벗에서의 역할 |
|---|---|
| Audio understanding | 한국어 목적지 발화 → transcript |
| Vision understanding | 간판/출구번호/화살표/입구 이해 |
| Text generation | 어르신용 한국어 안내문 생성 |
| Native bbox-style output | 화살표 tip/tail 좌표 → 좌/우 보정 |
| LiteRT-LM deployment | 서버 없이 Android 실행 |
| MTP / speculative decoding | S23 GPU decode 가속 |

번들 ~2.4GB. 주요 구성: vision encoder ~208MB, audio encoder ~102MB, LM prefill/decode ~781MB, per-layer embedder ~1.2GB, MTP drafter ~42MB. 하나의 모델 파일이 음성·이미지·텍스트를 모두 처리하지만, 그만큼 모바일 메모리·backend 선택을 신중하게 해야 했습니다.

## 5. Engineering Challenges

이 프로젝트의 어려움은 모델 성능 하나가 아니라, **한국어 오디오 → 모바일 배포 → 안전한 시각 안내 → 실기기 속도**가 한 제품 루프로 이어져야 한다는 점이었습니다. 각 문제는 따로 해결할 수 없었습니다. 오디오 정확도가 좋아도 Android에 얹히지 않으면 제품이 아니고, 시각 안내가 그럴듯해도 좌/우를 틀리면 어르신에게 위험합니다.

### 5.1 Audio: 고령 한국어 발화

목적지를 잘못 들으면 모든 경로가 틀어집니다. Base Gemma 4 audio는 134발화 한국어 held-out 셋에서 CER **13.14%**였고, 목적지 입력용으로는 부족했습니다.

| 문제 | 대응 | 결과 |
|---|---|---|
| 고령 발화의 느린 속도/휴지 | AI Hub *명령어 음성(노인남녀)* 45,823발화, 약 50시간 사용 | 고령 명령형 발화 반영 |
| Gemma 4 audio custom layer | Unsloth FastModel + PEFT + bitsandbytes 4-bit | `Gemma4ClippableLinear`까지 학습 |
| 모바일 배포 정확도 | audio encoder + LM attention/MLP rank-8 LoRA | deployed CER **5.00%** |

[AI Hub](https://www.aihub.or.kr/aihubdata/data/view.do?aihubDataSe=data&dataSetSn=94)

| 경로 | CER |
|---|---:|
| Base `gemma-4-E2B-it-litert-lm` (no LoRA) | 13.14% |
| 길벗 deployed LiteRT-LM 경로 | **5.00%** |
| 단말 base 대비 개선 | ~2.6× 감소 (−62%) |

여기서 5.00%는 단순한 benchmark 숫자가 아닙니다. "송파구 보건소", "강남역", "국립중앙박물관" 같은 목적지를 앱 흐름 안에서 다룰 수 있게 만드는 기준선이었습니다.

### 5.2 Deployment: `.litertlm`에 LoRA 넣기

학습보다 어려운 것은 배포였습니다. 제출 시점에 Gemma 4 audio LoRA를 Android `.litertlm` bundle에 넣는 공식 경로가 없었습니다.

| 경로 | 막힌 지점 |
|---|---|
| LiteRT-LM 변환기 | audio path shape unsupported |
| `litert-torch-nightly` | LM export 중심 |
| MediaPipe LLM | attention-only/GPU runtime 중심 |

그래서 직접 patcher를 만들었습니다. 핵심은 새 quant scale을 만들지 않는 **Same-grid requant**입니다.

```text
patched_int4 = clip(round((base_fp32 + LoRA_delta) / scale), -8, 7)  # 기존 scale 그대로
```

```text
HF LoRA delta ──► same-grid requant ──► LM weight graft in .litertlm
audio adapter ─► alpha-8 sidecar ─────► runtime loraPath
```

이 방식은 LoRA 방향 변화만 남기고 runtime kernel이 기대하는 quantization grid를 보존합니다. audio encoder를 `.litertlm` 안에서 과하게 건드리면 Adreno 740 prefill kernel이 불안정해져, audio adapter는 sidecar로 분리했습니다. 즉 최종 구조는 "가능한 부분은 bundle에 graft하고, 위험한 부분은 runtime sidecar로 붙이는" 타협입니다. 패처 소스: [`tools/README.md`](https://github.com/psymon-ai/gilbeot-public/blob/main/tools/README.md).

**Cactus 비교**도 진행했습니다. Cactus는 같은 문제를 다른 방향에서 푸는 좋은 대안이었습니다. 자체 runtime으로 Gemma 4 audio를 실행했고, 정확도만 보면 매우 매력적이었습니다. 문제는 S10e latency가 제품 한도를 넘었다는 점입니다.

| Backend | CER | S10e median/sample |
|---|---:|---:|
| LiteRT (graft+sidecar) | 5.003% | 12.48초 |
| Cactus INT8 | 3.949% | 41.94초 |
| Cactus INT4 | 4.856% | 45.32초 |

위 Cactus 수치는 byte-fallback detokenization fix 후 값입니다. `<0xEA><0xB9><0xB0>` 같은 토큰을 UTF-8로 재조합하지 못하던 문제를 수정해 PR을 올렸습니다: [cactus-compute/cactus#635](https://github.com/cactus-compute/cactus/pull/635). 더 강력한 단말에선 Cactus가 좋은 선택일 수 있지만, S10e/S23 목표에서는 LiteRT가 더 현실적이었습니다.

### 5.3 Vision: 좌/우를 모델 말에만 맡기지 않기

VLM은 기본 공간 관계, 특히 left/right에 취약합니다([EMNLP 2023](https://aclanthology.org/2023.emnlp-main.568/), [arXiv:2311.11477](https://arxiv.org/abs/2311.11477)). Gemma 4도 출구번호·표지판은 잘 읽지만 가로 화살표에서 가끔 단어를 뒤집었습니다.

해결은 bbox-style output입니다. 모델은 좌/우 단어를 확정하지 않고 화살표 tip/tail x좌표를 냅니다. 앱이 deterministic code로 방향을 계산합니다.

```text
arrow_tip_x < arrow_tail_x  →  left
arrow_tip_x > arrow_tail_x  →  right
```

즉 localization은 모델이, 최종 좌/우 판정은 코드가 맡습니다. 안내문과 픽셀 좌표가 충돌하면 앱이 단어를 교정합니다. 이 보정은 모델을 불신해서가 아니라, 길안내에서 방향 오류의 비용이 너무 크기 때문에 넣은 안전장치입니다. 좌우 반전 이미지 테스트로 모델이 상수 답을 외운 것이 아니라 실제 이미지를 보고 있음을 확인했습니다.

### 5.4 Latency: 단말별 backend 정책

속도는 편의가 아닌 접근성입니다. S23 GPU에서는 MTP가 decode 17.8→11.5초로 1.5~1.6× 가속했습니다. S10e CPU에서는 반대로 drafter와 main model이 같은 자원을 나눠 쓰고, LoRA가 적용된 main LM과 base drafter 분포 차이로 acceptance가 낮아져 MTP overhead가 더 컸습니다.

| 단말 | Backend | MTP | 사진당 |
|---|---|---|---:|
| Galaxy S23 | GPU | on | ~12초 |
| Galaxy S10e | CPU | off | ~50초 |

그래서 정책은 단순합니다. 좋은 단말에서는 GPU와 MTP로 기다림을 줄이고, 오래된 단말에서는 느리더라도 CPU-safe 경로로 끝까지 동작하게 했습니다.

### 5.5 Demo: 고정 흐름이지만 실제 추론

해외 심사자가 한국 API 키·GPS·지하철 사진·한국어 발화를 모두 준비하기는 어렵습니다. 그래서 Demo APK는 입력 일부를 고정하되, 모델 판단은 실제로 실행합니다. 특히 bundled Korean WAV는 단순 재생용 음원이 아니라, Gemma 4 audio가 매번 실제로 듣고 transcript와 목적지를 뽑는 입력입니다.

| 고정한 것 | 실제 실행한 것 |
|---|---|
| bundled Korean WAV | Gemma 4 audio STT + 목적지 인식 |
| cached T-Map route | transcript 검증 후 route 시작 |
| demo photo sequence | Gemma 4 vision 안내 생성 |
| generated JSON | parsing, 좌/우 교정, 도착 인식 |
| gallery long-press | 임의 사진 + EXIF off-route 처리 |

카메라를 길게 누르면 갤러리에서 임의 사진을 고를 수 있습니다. 같은 vision path를 타고, EXIF GPS가 계획 경로에서 멀면 안내를 상상하지 않고 장면 설명만 합니다. 한국 production 빌드(`com.psymon.gilbeot.real`)에는 이런 발판이 없습니다.

## 6. Future Work

길찾기는 시작점입니다. 진짜 문제는 "AI가 할 수 있는 일"과 "어르신이 스마트폰에서 실제로 할 수 있는 일" 사이의 간극입니다. 폰뱅킹, 공문서 처리, 병원 예약, 약 복용, 복지 서비스 — 개인정보·건강정보·금융정보가 들어가는 문제들은 클라우드로 보낼 수 없기에 반드시 온디바이스여야 합니다. 길벗은 그 간극을 줄이는 온디바이스 AI 인터페이스의 첫 사례입니다.

## 7. Thanks

이 글의 마지막은 특별한 감사 인사로 채우려 합니다.

누구보다 적극적으로 제품 테스트에 참여해 주시고 데모 영상 촬영에도 기꺼이 응해 주신 장인어른, 당신이 계셔서 개발을 끝까지 마칠 수 있었습니다. 길벗을 개발하는 동안 늘 곁에서 귀중한 피드백을 주시고, 변함없이 응원해 주셔서 진심으로 감사드립니다.

정년퇴직 후 새로운 삶을 시작하시는 당신의 앞길에 늘 행복이 가득하시기를.

## 8. License

This Writeup has been released under the Attribution 4.0 International (CC BY 4.0) license.

_Gemma is a trademark of Google LLC._

## 9. Citation

Psymon. 길벗(Gilbeot) — 고령 사용자를 위한 온디바이스 멀티모달 AI 길안내 앱. https://www.kaggle.com/competitions/gemma-4-good-hackathon/writeups/gilbeot. 2026. Kaggle.
