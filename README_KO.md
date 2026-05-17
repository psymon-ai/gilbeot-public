# 길벗(Gilbeot) — 고령 사용자를 위한 온디바이스 길안내 도우미

> 고령 사용자를 위한 온디바이스 멀티모달 길안내 안드로이드 앱입니다. 한국어로 목적지를 말하면 Gemma 4 audio가 발화를 인식하고 앱이 경로를 만듭니다. 길에서 찍은 사진을 Gemma 4 vision이 읽어, 그 장면에 맞는 짧은 안내문을 만들어 TTS로 들려줍니다. 핵심 추론은 모두 단말 안에서 실행됩니다.

[Kaggle Gemma 4 Good Hackathon][hackathon] (Digital Equity & Inclusivity 트랙) 제출작입니다. 전체 글: [`KAGGLE_WRITEUP_KO.md`](KAGGLE_WRITEUP_KO.md) · English [`KAGGLE_WRITEUP.md`](KAGGLE_WRITEUP.md) · English README [`README.md`](README.md).

| | |
|---|---|
| **모델** | Gemma 4 E2B (audio + vision + text), LiteRT-LM으로 온디바이스 실행 |
| **스택** | Flutter • dart:ffi • LiteRT-LM • flutter_gemma 0.14.5 (MTP + LoRA sidecar용 소규모 로컬 fork 포함) |
| **한국어 STT** | APK에 LoRA sidecar 번들 — **CER 13.14 % → 5.00 %**, 134발화 한국어 held-out 셋 기준 (−62 %) |
| **검증 단말** | Galaxy S23 (GPU + MTP, ~12초/사진), S10e (CPU, MTP off, ~50초/사진) |
| **개인정보** | 최초 실행 이후 완전 오프라인 (모델은 최초 1회 다운로드, LoRA + canned 경로는 APK에 번들) |

[hackathon]: https://www.kaggle.com/competitions/gemma-4-good-hackathon

---

## 체험하기 — Judge Demo APK

한국 production 빌드는 한국 지도 API (ODsay / T-Map / Naver)와 한국어 발화자가 필요해 해외 심사자가 그대로 시도하기 어렵습니다. Judge Demo APK는 시나리오는 고정하되 **모든 사진과 모든 음성 입력에 대해 단말 내 Gemma 4 모델 추론은 매번 실제로 실행**합니다.

1. [GitHub Releases 페이지](https://github.com/psymon-ai/gilbeot-public/releases/tag/v1.0.0)에서 [`gilbeot-judge-demo.apk`](https://github.com/psymon-ai/gilbeot-public/releases/download/v1.0.0/gilbeot-judge-demo.apk) (201 MB) **설치**. Android 10+ 필요. `adb install` 또는 브라우저의 "알 수 없는 앱 설치 허용" 후 APK 탭.
2. **Gilbeot Demo** 실행. 최초 실행 시 Hugging Face에서 ~2.4 GB Gemma 4 번들 다운로드 ([`psymon/gilbeot-korean-audio-litertlm`](https://huggingface.co/psymon/gilbeot-korean-audio-litertlm); Wi-Fi 기준 약 3–5분).
3. **마이크** 버튼 1회 탭. 사전 녹음된 한국어 발화 (*"송파구보건소 가야 해"*)가 스피커로 재생됩니다. 단말 내 Gemma 4 audio 모델이 실시간으로 인식해 한국어/영어 텍스트를 함께 표시합니다.
4. **카메라** 버튼 탭. 번들된 지하철역 사진이 전체화면으로 열리고 하단에 셔터 버튼이 보입니다. 셔터를 누르면 단말 내 Gemma 4 vision 모델이 JSON 안내문 (`{ instruction, landmarks, is_arrival, arrow_tip_x, arrow_tail_x }`)을 만들고 TTS가 읽어 줍니다.
5. 사진 2 → 3 → 4까지 반복. 4번째 사진은 목적지 건물이며 모델이 `is_arrival=true`로 표시하면 길벗이 작별 인사를 합니다.
6. **(보너스) 카메라 버튼을 길게 누르기** — 갤러리가 열리고 *아무 사진이나* 선택할 수 있습니다. 선택한 사진에 대해서도 단말 내 Gemma 4 vision 모델이 실제로 실행됩니다. 만약 사진의 EXIF GPS가 캐시된 경로에서 멀리 떨어져 있으면 앱은 안내를 상상하지 않고 *"이 사진은 계획된 경로 위에 있는 것 같지 않습니다"* 라는 안내와 함께 장면만 묘사합니다. 경로 이탈 안전 가드가 라이브로 동작.

지도 버튼을 누르면 캐시된 T-Map 도보 polyline과 사진별로 한 단계씩 이동하는 origin 마커가 보입니다.

Demo APK에서 실제로 실행되는 것: Gemma 4 audio · vision · text 생성, JSON 파싱, bbox 기반 좌/우 교정, 도착 인식. 고정된 것: 경로 데이터 (캐시된 T-Map polyline), 목적지 발화 (녹음 WAV), GPS (사진별 EXIF). 분리 기준은 아래 [데모 환경 대체 사항](#데모-환경-대체-사항) 섹션에 정리했습니다.

## 데모 영상

📺 **[3분 워크스루 (YouTube)](https://www.youtube.com/watch?v=jg-FSf5QdNI)**

## 아키텍처 (한눈에)

```
음성 (한국어 WAV) ──► Gemma 4 audio ──► transcript
                                          │
                                          ▼
                            목적지 + canned 경로
                                          │
                                          ▼
사진 (카메라 또는 번들) ─► Gemma 4 vision (English systemPrompt)
                                          │
                                          ▼
JSON { instruction, landmarks, is_arrival,
       arrow_tip_x, arrow_tail_x } ────► English TTS
        │
        └─► tip_x vs tail_x → 결정론적 좌/우 교정
            (bbox grounding; VLM은 "left/right" 단어를 자주 뒤집음)
```

Gemma 4는 `dart:ffi` → LiteRT-LM C API → `libLiteRtLm.so` (arm64) 경로로 실행됩니다. Release APK는 forked `flutter_gemma` 0.14.5를 사용합니다 — 네 군데 인라인 추가: 세 군데는 Gemma 4 MTP 와이어링 (C 심볼은 `.so`에 이미 export되어 있는데 upstream이 선언만 안 함), 한 군데는 LoRA sidecar를 FFI 클라이언트에 통과시키는 부분 (upstream은 `.litertlm` 경로의 `loraPath`를 거부). 자세한 내용: [`patches/README.md`](patches/README.md).

Speculative decoding (Gemma 4 MTP)은 backend별로 분기:

| Backend | MTP | 이유 |
|---|---|---|
| GPU (S23 Adreno 740) | on  | decode 1.5–1.6× 가속, 품질 손실 없음 측정. |
| CPU (S10e XNNPACK)   | off | drafter 오버헤드 > acceptance 이득. LoRA로 target 분포가 옮겨져 acceptance가 더 낮음. MTP-off 대비 사진당 ~10–15초 느려짐. |

전체 다이어그램 + 구성요소 사이즈: [`docs/architecture.md`](docs/architecture.md).

## 한국어 STT — 핵심 동작 원리

기본 Gemma 4 audio 경로는 134발화 한국어 held-out 셋 (목적지 + 일상 발화, 고령 발화자)에서 **CER 13.14 %**를 냅니다. 목적지 입력 용도로는 쓰기 어려운 수치입니다. audio encoder와 LM의 attention / MLP 레이어에 rank-8 LoRA를 학습했습니다 (Unsloth FastModel + PEFT + bitsandbytes 4-bit; AI Hub *명령어 음성(노인남녀)* 데이터셋에서 ~45,823 발화 / ~50 시간 사용). 그 후 same-grid requant patcher를 직접 만들어 Android에 배포했습니다 — 제출 시점에 공식 Gemma 4 audio LoRA → `.litertlm` exporter가 존재하지 않습니다.

| 경로 | CER | 비고 |
|---|---:|---|
| Base `gemma-4-E2B-it-litert-lm` (LoRA 없음) | 13.14 % | 단말 baseline |
| 길벗 deployed (graft + alpha-8 sidecar)     | **5.00 %** | Android 배포 경로 |
| 단말 base 대비 개선                          | 약 2.6× 감소 | (상대 −62 %) |
| HF 참조 (merged, fp16)                       |  3.06 % | HF checkpoint, 배포 전 merge; 배포 경로와의 양자화 갭 ~2 pp |

배포 번들은 두 부분으로 분리되어 있습니다. LM 가중치는 `.litertlm` 안에서 in-place로 graft-patch 되어 단일 `.litertlm` 파일로 Hugging Face ([`psymon/gilbeot-korean-audio-litertlm`](https://huggingface.co/psymon/gilbeot-korean-audio-litertlm))에 업로드되고, audio adapter는 alpha-8 LoRA sidecar로 APK의 `assets/lora/`에 번들됩니다. `.litertlm` 내부에서 audio encoder를 너무 공격적으로 건드리면 Adreno 740 prefill 커널이 불안정해져 — encoder 바이트는 손대지 않는 분리 구조를 택했습니다.

Patcher 소스 + 작업 기록: [`tools/README.md`](tools/README.md).

## 소스에서 빌드하기

[`docs/build.md`](docs/build.md) 참고. 사이드-바이-사이드 빌드 두 종류:

| 빌드 | Package ID | 라벨 | 용도 |
|---|---|---|---|
| Judge demo       | `com.psymon.gilbeot.demo` | Gilbeot Demo | 해외 심사자 대상 |
| Korea production | `com.psymon.gilbeot.real` | Gilbeot      | 한국 실 사용자 대상 |

`applicationId`가 다르므로 두 APK가 같은 단말에 동시 설치됩니다.

## Repo 구조

```
gilbeot-public/
├── app/                 Flutter 프로젝트 (Dart + Android)
│   ├── lib/             ~25개 .dart 파일
│   ├── assets/
│   │   ├── env_config.example   ← env_config로 복사 후 키 입력
│   │   ├── demo/                ← 번들 사진 / WAV / 캐시된 경로 polyline
│   │   ├── demo_photos/         ← 엄선한 역 사진 4장
│   │   └── lora/                ← 한국어 audio LoRA sidecar (50.7 MB, 번들)
│   └── pubspec.yaml
├── third_party/         앱이 사용하는 patched flutter_gemma 0.14.5 fork
├── patches/             flutter_gemma fork 수정점 (MTP + LoRA sidecar) 문서
├── scripts/             build_install_demo_apk.py + build_install_realuse_apk.py
├── tools/               .litertlm LoRA patcher (한국어 audio adapter, byte-level)
└── docs/
    ├── architecture.md
    └── build.md
```

## 데모 환경 대체 사항

Demo APK는 한국 밖에서는 end-to-end로 실행할 수 없는 두 가지를 대체합니다. 모두 비-한국 심사자를 위한 self-contained 데모 동작에 필수적인 조치이며, 무언가를 가리는 의도가 아닙니다.

1. **실 GPS 대신 사진별 EXIF GPS 사용** — 심사자가 잠실역에 있지 않으므로 실 GPS가 무의미합니다. Demo 사진 4장이 실제 EXIF GPS를 갖고 있어, 도보 진행에 맞춰 지도 위 origin 마커를 한 단계씩 진행시킵니다. Production 빌드는 실 Geolocator 사용.
2. **마지막 사진 도착 fallback** — Production 빌드는 도착 판정에 두 독립 신호를 결합합니다: 목적지까지의 GPS-Haversine + vision 모델의 `is_arrival` 플래그. 데모는 모델 신호 한 축만 가지므로 `if (isLastDemoPhoto || textArrival) llmArrival = true;` 한 줄로 단 한 번의 오인식으로 데모가 멈추지 않도록 합니다. 실제 S23 reviewer 시뮬레이션에서는 모델이 photo 4를 스스로 인식해 fallback이 발동하지 않았습니다.

한국 production 빌드 (`com.psymon.gilbeot.real`)는 두 항목 모두 대체하지 않습니다 — end-to-end 모두 실제 동작입니다.

## 주석 언어 정책

도메인 로직 파일 (`app/lib/services/`, `app/lib/screens/`)은 한국어 주석을 사용합니다 — 한국 고령자 STT 품질, 한국 지도 API 특이사항, 한국 발화 패턴, 한국 접근성 UX 결정 같이 한국어가 도메인의 자연 언어인 부분이기 때문입니다. 범용 UI 컴포넌트 (`app/lib/widgets/`)는 영어 주석 — 한국 도메인 맥락 밖에서 재사용을 의도한 코드입니다. 의도된 분리입니다.

## License

MIT — [LICENSE](LICENSE) 참조.

_Gemma is a trademark of Google LLC._
