import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'dart:ui' as ui;

import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart' show rootBundle;
import 'package:flutter_dotenv/flutter_dotenv.dart';
import 'package:flutter_gemma/flutter_gemma.dart';
import 'package:path_provider/path_provider.dart';

import '../config/demo_mode.dart';
import '../util/device_caps.dart';
import 'cactus_asr_client.dart';
import 'hf_gemma_proxy_client.dart';

/// Gemma 4 E2B (.litertlm) on-device wrapper for 길벗.
///
/// flutter_gemma 0.14.x 가 LiteRT-LM 을 dart:ffi 로 직접 로드.
/// ModelType.gemma4 + supportImage. 모델 파일은 `MODEL_LOCAL_PATH` env 우선,
/// 없으면 ⓐ asset 번들 (개발/PoC) ⓑ HuggingFace 다운로드 (production) 순.
class GemmaService {
  GemmaService._();
  static final GemmaService instance = GemmaService._();
  static const String _defaultAudioTranscribePrompt =
      '한국어 음성을 그대로 받아쓰세요. 숫자는 한글 발음 그대로 쓰세요. 결과만 한 줄.';

  InferenceModel? _model;
  bool _ready = false;
  int _imageTargetWidth = 1024; // DeviceCaps.probe() 결과로 initialize 에서 갱신
  String? _loraPath;
  String? _lastRawAudioTranscript;
  HfGemmaProxyClient? _audioProxy;
  CactusAsrClient? _cactusAsr;

  /// 첫 실행 시 ~2.4GB Gemma 4 모델 다운로드 진행률 (0~99) 또는 null
  /// (idle / 완료). home_screen splash 가 listener 로 구독해서
  /// `Strings.loadingProgress(p)` 로 UI 갱신. 모델이 이미 캐시되어 있으면
  /// fromNetwork callback 이 한 번도 안 불려 값은 null 유지 → splash 는
  /// 기본 "Loading on-device Gemma 4 model... (~10s)" 표시.
  final ValueNotifier<int?> downloadProgress = ValueNotifier<int?>(null);

  /// Bundled Korean audio LoRA sidecar asset path. APK 내부에 ~50 MB 로 들어가고,
  /// 첫 launch 시 app-private dir 로 한 번 extract 된 후 그 file path 가
  /// `_loraPath` 로 사용된다. uninstall 시 app-private dir 가 함께 삭제되므로
  /// orphaned data 잔존하지 않는다.
  static const String _bundledLoraAsset =
      'assets/lora/gemma4_full_lm_lora_atten_mlp_alpha8.bin';

  Future<String> _ensureBundledLoraExtracted() async {
    final dir = await getApplicationSupportDirectory();
    final loraDir = Directory('${dir.path}/lora');
    if (!loraDir.existsSync()) {
      loraDir.createSync(recursive: true);
    }
    final filename = _bundledLoraAsset.split('/').last;
    final loraFile = File('${loraDir.path}/$filename');
    if (loraFile.existsSync() && loraFile.lengthSync() > 0) {
      // 이미 extract 됨 (이전 launch 에서). reuse.
      return loraFile.path;
    }
    debugPrint('[gemma/lora] extracting bundled sidecar → ${loraFile.path}');
    final bytes = await rootBundle.load(_bundledLoraAsset);
    await loraFile.writeAsBytes(
      bytes.buffer.asUint8List(bytes.offsetInBytes, bytes.lengthInBytes),
      flush: true,
    );
    return loraFile.path;
  }

  bool get isReady => _ready;
  String? get lastRawAudioTranscript => _lastRawAudioTranscript;

  /// 디바이스 능력 자동 감지 — DeviceCaps.probe() 결과 사용.
  /// 명시 override 도 가능 (테스트/디버그 용).
  Future<void> initialize({int? maxTokens, PreferredBackend? backend}) async {
    final caps = await DeviceCaps.probe();
    maxTokens ??= caps.maxTokens;
    backend ??= caps.preferredBackend;
    _imageTargetWidth = caps.imageTargetWidth;
    final imageWidthOverride = int.tryParse(
      dotenv.maybeGet('GEMMA_IMAGE_TARGET_WIDTH')?.trim() ?? '',
    );
    if (imageWidthOverride != null && imageWidthOverride > 0) {
      _imageTargetWidth = imageWidthOverride;
    }
    // 모델 위치 우선순위:
    //   ① MODEL_LOCAL_PATH 가 있고 파일 존재 → fromFile (dev: ADB push 경로)
    //   ② MODEL_DOWNLOAD_URL 이 있음 → fromNetwork (production: HF/CDN 자동 다운로드)
    //   ③ 둘 다 없으면 명시적 에러
    final audioBackend = (dotenv.maybeGet('GEMMA_AUDIO_BACKEND') ?? 'litertlm')
        .toLowerCase()
        .trim();
    final skipLitertLm =
        (dotenv.maybeGet('HF_PROXY_SKIP_LITERTLM')?.toLowerCase().trim() ==
            'true') ||
        (dotenv.maybeGet('GEMMA_SKIP_LITERTLM')?.toLowerCase().trim() ==
            'true');
    if (audioBackend == 'cactus' || audioBackend == 'cactus_exec') {
      final binPath =
          dotenv.maybeGet('CACTUS_ASR_BIN')?.trim() ??
          '/data/data/com.psymon.gilbeot/files/cactus/asr';
      final modelPath =
          dotenv.maybeGet('CACTUS_MODEL_PATH')?.trim() ??
          '/data/local/tmp/cactus_transcribe/models/gemma4_merged_audio_int8_run59';
      final prompt =
          dotenv.maybeGet('CACTUS_TRANSCRIBE_PROMPT')?.trim() ??
          'Transcribe the Korean audio exactly.';
      final language = dotenv.maybeGet('CACTUS_LANGUAGE')?.trim() ?? 'ko';
      _cactusAsr = CactusAsrClient(
        binPath: binPath,
        modelPath: modelPath,
        prompt: prompt,
        language: language,
      );
      debugPrint(
        '[gemma/cactus] audio backend enabled: '
        'bin=$binPath model=$modelPath',
      );
      if (skipLitertLm) {
        _ready = true;
        debugPrint(
          '[gemma/cactus] skipping LiteRT-LM load; '
          'audio-only route-entry mode ready',
        );
        return;
      }
    }
    if (audioBackend == 'hf_proxy') {
      final proxyUrl = dotenv.maybeGet('HF_GEMMA_PROXY_URL')?.trim();
      if (proxyUrl == null || proxyUrl.isEmpty) {
        throw StateError(
          'GEMMA_AUDIO_BACKEND=hf_proxy requires HF_GEMMA_PROXY_URL',
        );
      }
      _audioProxy = HfGemmaProxyClient(proxyUrl);
      await _audioProxy!.health();
      debugPrint('[gemma/hf-proxy] audio backend enabled: $proxyUrl');

      if (skipLitertLm) {
        _ready = true;
        debugPrint(
          '[gemma/hf-proxy] skipping LiteRT-LM load; audio-only mode ready',
        );
        return;
      }
    }

    final modelPath = dotenv.maybeGet('MODEL_LOCAL_PATH')?.trim();
    final modelUrl = dotenv.maybeGet('MODEL_DOWNLOAD_URL')?.trim();
    final configuredLoraPath = dotenv.maybeGet('MODEL_LORA_LOCAL_PATH')?.trim();
    final hfToken = dotenv.maybeGet('HF_READ_API_KEY')?.trim();

    final builder = FlutterGemma.installModel(
      modelType: ModelType.gemma4,
      fileType: ModelFileType.litertlm,
    );
    if (modelPath != null &&
        modelPath.isNotEmpty &&
        File(modelPath).existsSync()) {
      debugPrint('[gemma] PRODUCTION — fromFile: $modelPath');
      await builder.fromFile(modelPath).install();
    } else if (modelUrl != null && modelUrl.isNotEmpty) {
      debugPrint(
        '[gemma] PRODUCTION — fromNetwork: $modelUrl '
        '(token=${hfToken != null && hfToken.isNotEmpty ? "yes" : "no"})',
      );
      // 첫 실행: ~2.4GB 다운로드 (~1~5분). 이후 SharedPrefs 캐시로 skip.
      // ⚠️ Gallery-pinned commit URL 만 사용. main 브랜치는 multi-signature 라 거부됨.
      downloadProgress.value = 0;
      try {
        await builder
            .fromNetwork(
              modelUrl,
              token: (hfToken != null && hfToken.isNotEmpty) ? hfToken : null,
            )
            .withProgress((p) {
              downloadProgress.value = p;
              if (p % 10 == 0) debugPrint('[gemma] download progress: $p%');
            })
            .install();
      } finally {
        downloadProgress.value = null;
      }
    } else {
      throw StateError(
        '모델 경로 미지정. assets/env_config 에 ⓐ MODEL_LOCAL_PATH=/data/local/tmp/gilbeot/... '
        '또는 ⓑ MODEL_DOWNLOAD_URL=https://huggingface.co/.../<commit>/...litertlm 추가.',
      );
    }

    debugPrint(
      '[gemma] PRODUCTION — model installed, creating inference handle',
    );
    _loraPath = null;
    // 우선순위 ① MODEL_LORA_LOCAL_PATH (dev: adb push 한 file) →
    //          ② bundled APK asset (deployed user: assets/lora/...bin → app private 로 extract)
    if (configuredLoraPath != null && configuredLoraPath.isNotEmpty) {
      if (File(configuredLoraPath).existsSync()) {
        _loraPath = configuredLoraPath;
        debugPrint('[gemma/lora] LiteRT-LM LoRA sidecar enabled (env): $_loraPath');
      } else {
        debugPrint(
          '[gemma/lora] MODEL_LORA_LOCAL_PATH set but file missing: '
          '$configuredLoraPath — falling back to bundled asset',
        );
      }
    }
    if (_loraPath == null) {
      try {
        _loraPath = await _ensureBundledLoraExtracted();
        debugPrint(
          '[gemma/lora] LiteRT-LM LoRA sidecar enabled (bundled): $_loraPath',
        );
      } catch (e) {
        debugPrint('[gemma/lora] bundled sidecar extract failed: $e');
      }
    }
    _model = await FlutterGemma.getActiveModel(
      maxTokens: maxTokens,
      preferredBackend: backend,
      supportImage: true,
      supportAudio: true, // Gemma 4 E2B multimodal audio (16kHz mono WAV).
    );
    _ready = true;
    debugPrint('[gemma] PRODUCTION — model ready (vision + audio)');
  }

  /// 어르신 발화 (16kHz mono WAV) → 한국어 텍스트.
  ///
  /// flutter_gemma 공식 예제와 동일하게 `record` 가 만든 WAV 컨테이너를 그대로 보낸다.
  /// Android LiteRT-LM 오디오 경로는 내부 miniaudio 가 WAV metadata 를 읽으므로
  /// header 를 제거해 raw PCM 만 넘기면 streaming code 13 으로 실패할 수 있다.
  Future<String> transcribeAudio(Uint8List audioBytes) async {
    _ensureReady();
    final proxy = _audioProxy;
    if (proxy != null) {
      debugPrint('[gemma/hf-proxy] transcribe wav bytes=${audioBytes.length}');
      final transcript = await proxy.transcribeAudio(audioBytes);
      _lastRawAudioTranscript = transcript;
      return transcript;
    }
    final cactus = _cactusAsr;
    if (cactus != null) {
      debugPrint('[gemma/cactus] transcribe wav bytes=${audioBytes.length}');
      final transcript = await cactus.transcribeAudio(audioBytes);
      _lastRawAudioTranscript = transcript;
      return transcript;
    }
    final wavBytes = audioBytes;
    final transcribePrompt = await _resolveAudioTranscribePrompt();
    final systemInstruction = await _resolveAudioSystemInstruction();
    final samplerConfig = await _resolveAudioSamplerConfig();
    debugPrint(
      '[gemma/audio] wav bytes=${wavBytes.length} '
      'header=${wavBytes.take(12).toList()}',
    );
    final freshChat = await _model!.createChat(
      temperature: samplerConfig.temperature,
      randomSeed: 1,
      topK: samplerConfig.topK,
      topP: samplerConfig.topP,
      loraPath: _loraPath,
      supportImage: true,
      supportAudio: true,
      systemInstruction: systemInstruction,
    );
    try {
      await freshChat.addQueryChunk(
        Message.withAudio(
          text: transcribePrompt,
          audioBytes: wavBytes,
          isUser: true,
        ),
      );
      final resp = await freshChat.generateChatResponse();
      // ModelResponse.toString() 은 'TextResponse("...")' wrapping. 실제 토큰만 추출.
      final raw = (resp is TextResponse) ? resp.token : resp.toString();
      debugPrint('[gemma/audio] transcribe raw="$raw"');
      // 모델이 따옴표/JSON 으로 감쌀 수 있어 단순 정리.
      final cleaned = raw
          .trim()
          .replaceAll(RegExp(r'^["“‘\s]+'), '')
          .replaceAll(RegExp(r'["”’\s]+$'), '');
      _lastRawAudioTranscript = cleaned;
      return cleaned;
    } finally {
      await freshChat.close();
    }
  }

  Future<String> _resolveAudioTranscribePrompt() async {
    final promptFile = dotenv
        .maybeGet('GEMMA_AUDIO_TRANSCRIBE_PROMPT_FILE')
        ?.trim();
    if (promptFile != null && promptFile.isNotEmpty) {
      try {
        final file = File(promptFile);
        if (await file.exists()) {
          final prompt = (await file.readAsString()).trim();
          if (prompt.isNotEmpty) {
            debugPrint(
              '[gemma/audio] using prompt file: $promptFile '
              '(chars=${prompt.length})',
            );
            return prompt;
          }
        }
      } catch (e) {
        debugPrint('[gemma/audio] prompt file read failed: $promptFile ($e)');
      }
    }

    final configured = dotenv.maybeGet('GEMMA_AUDIO_TRANSCRIBE_PROMPT')?.trim();
    if (configured != null && configured.isNotEmpty) {
      return configured.replaceAll(r'\n', '\n');
    }
    return _defaultAudioTranscribePrompt;
  }

  Future<String?> _resolveAudioSystemInstruction() async {
    final systemFile = dotenv
        .maybeGet('GEMMA_AUDIO_SYSTEM_PROMPT_FILE')
        ?.trim();
    if (systemFile != null && systemFile.isNotEmpty) {
      try {
        final file = File(systemFile);
        if (await file.exists()) {
          final prompt = (await file.readAsString()).trim();
          if (prompt.isNotEmpty) {
            debugPrint(
              '[gemma/audio] using system prompt file: $systemFile '
              '(chars=${prompt.length})',
            );
            return prompt;
          }
        }
      } catch (e) {
        debugPrint(
          '[gemma/audio] system prompt file read failed: '
          '$systemFile ($e)',
        );
      }
    }

    final configured = dotenv.maybeGet('GEMMA_AUDIO_SYSTEM_PROMPT')?.trim();
    if (configured != null && configured.isNotEmpty) {
      return configured.replaceAll(r'\n', '\n');
    }
    return null;
  }

  Future<({double temperature, int topK, double? topP})>
  _resolveAudioSamplerConfig() async {
    const fallbackTemperature = 0.1;
    const fallbackTopK = 1;
    const double? fallbackTopP = null;
    final configFile = dotenv
        .maybeGet('GEMMA_AUDIO_SAMPLER_CONFIG_FILE')
        ?.trim();
    if (configFile == null || configFile.isEmpty) {
      return (
        temperature: fallbackTemperature,
        topK: fallbackTopK,
        topP: fallbackTopP,
      );
    }
    try {
      final file = File(configFile);
      if (!await file.exists()) {
        return (
          temperature: fallbackTemperature,
          topK: fallbackTopK,
          topP: fallbackTopP,
        );
      }
      final decoded = jsonDecode(await file.readAsString());
      if (decoded is! Map<String, dynamic>) {
        return (
          temperature: fallbackTemperature,
          topK: fallbackTopK,
          topP: fallbackTopP,
        );
      }
      final temperature =
          (decoded['temperature'] as num?)?.toDouble() ?? fallbackTemperature;
      final topK = (decoded['topK'] as num?)?.toInt() ?? fallbackTopK;
      final topP = (decoded['topP'] as num?)?.toDouble();
      debugPrint(
        '[gemma/audio] using sampler file: $configFile '
        '(temperature=$temperature, topK=$topK, topP=$topP)',
      );
      return (temperature: temperature, topK: topK, topP: topP);
    } catch (e) {
      debugPrint('[gemma/audio] sampler file read failed: $configFile ($e)');
      return (
        temperature: fallbackTemperature,
        topK: fallbackTopK,
        topP: fallbackTopP,
      );
    }
  }

  Future<Map<String, String?>> parseDestinationFromAudio(
    Uint8List wavBytes,
  ) async {
    _ensureReady();
    final proxy = _audioProxy;
    if (proxy != null) {
      debugPrint('[gemma/hf-proxy] intent wav bytes=${wavBytes.length}');
      return proxy.parseDestinationFromAudio(wavBytes);
    }
    final cactus = _cactusAsr;
    if (cactus != null) {
      debugPrint('[gemma/cactus] intent wav bytes=${wavBytes.length}');
      final transcript = await cactus.transcribeAudio(wavBytes);
      _lastRawAudioTranscript = transcript;
      return {
        'transcript': transcript,
        'destination': _extractDestinationCandidate(transcript) ?? transcript,
        'origin': null,
      };
    }
    debugPrint(
      '[gemma/audio] intent via two-stage STT wav bytes=${wavBytes.length} '
      'header=${wavBytes.take(12).toList()}',
    );
    final transcript = await transcribeAudio(wavBytes);
    final parsed = await parseDestination(transcript);
    return {
      'transcript': transcript,
      'destination': parsed['destination'],
      'origin': parsed['origin'],
    };
  }

  Future<Map<String, String?>> parseDestination(String userText) async {
    _ensureReady();
    final proxy = _audioProxy;
    if (proxy != null) {
      debugPrint('[gemma/hf-proxy] parse destination text="$userText"');
      return proxy.parseDestination(userText);
    }
    if (_cactusAsr != null) {
      final destination = _extractDestinationCandidate(userText) ?? userText;
      debugPrint(
        '[gemma/cactus] parse destination text="$userText" '
        '=> "$destination"',
      );
      return {'destination': destination, 'origin': null};
    }
    // fresh chat — flutter_gemma 0.14.x 의 LiteRT-LM 백엔드는 모델당 단일 conversation
    // 만 유효 (createChat 이 기존 session 을 invalidate). transcribeAudio / generateGuidance
    // 와 동일한 단발성 패턴으로 통일해서 "Bad state: Session is closed" 차단.
    final freshChat = await _model!.createChat(
      temperature: 0.1,
      randomSeed: 1,
      topK: 1,
      supportImage: true,
      supportAudio: true,
    );
    try {
      await freshChat.addQueryChunk(
        Message.text(
          text:
              '''사용자 발화에서 목적지와 출발지를 추출해 JSON으로만 답하세요.
사용자 발화: "$userText"
스키마: {"destination": "...", "origin": null|"..."}''',
          isUser: true,
        ),
      );
      final response = await freshChat.generateChatResponse();
      final parsed = _safeParseJson(response.toString());
      return {
        'destination': _stringOrNull(parsed['destination'] ?? parsed['dest']),
        'origin': _stringOrNull(parsed['origin'] ?? parsed['start']),
      };
    } finally {
      await freshChat.close();
    }
  }

  /// 마지막으로 설정된 목적지명. home_screen 의 `_resolveOrigin` 직후 set.
  String? _lastDestinationName;
  void setDestinationName(String name) => _lastDestinationName = name;

  Future<Map<String, dynamic>> generateGuidance({
    required Uint8List photoBytes,
    required String stepContext,
    required int stepHint,
    int? totalSteps,
  }) async {
    _ensureReady();

    // DEMO_MODE (judge demo): English systemPrompt so the model emits English
    // directly (Gemma 4 is multilingual; Arabic exit numerals on Korean signs
    // are language-independent). Keeps all 13 grounding rules; only language
    // changes. Korean systemPrompt below remains the production path.
    const systemPromptEn =
        '''You are Gilbeot, a kind walking-guide assistant for blind and elderly users.

**Voice:** speak as if to a kind grandparent. Warm, gentle, conversational. Use first-person framing ("I can see...", "I'll guide you to...", "You'll find..."). NEVER sound like a textbook or robot. Keep it short, kind, and clear.

[Hard rules]
1. NEVER use abstract directions like "east/west/south/north", "OO meters", or "rotate OO degrees".
2. Point to concrete route objects visible in the photo: signs, exit numbers, "Exit / 出口" markers, transfers, fare gates, ticket booths, platforms, stairs, building entrances.
3. One step at a time. End with a brief warm reassurance ("you're doing great", "almost there", "just a little further").
4. If clues are insufficient, do NOT guess — request another photo politely.
5. Plain everyday English. Avoid technical or formal phrasing.
6. If you cannot read sign text with full confidence, describe it ("yellow text", "blue arrow") rather than guessing letters.
7. For stairs / escalators / crosswalks, add a kind safety note ("keep one hand on the handrail", "take it slow").
8. **Carry over any exit number or place name in the route context** (e.g., "Jamsil Station Exit 10") into the instruction.
9. Only say left/right/up/down when the visual direction in the photo (an arrow, a sign, a stair) is unambiguous.
10. Never use vague pointers like "over there" / "that way" — name a concrete sign / exit / stairway / door.
11. If the photo shows only indoor furniture (chairs / desks / walls), it is NOT a route clue — kindly request another photo; do not invent a direction.
12. Output JSON only. If the destination's own building or sign is clearly visible in front of the user, set is_arrival=true and warmly congratulate ("we made it — you've arrived at ...").
13. **Korean signage**: Arabic numerals on a sign (e.g. "10", "11") and English words ("Exit") may be quoted in your instruction. **Korean (Hangul) text on signs MUST NOT be transcribed, transliterated, or quoted in your English instruction — describe it visually instead** (e.g. "a yellow sign with Korean text", "a Korean place-name marker"). Transliterating Hangul invites hallucination of Korean place names that are not actually on the sign.''';

    final systemPrompt = DemoMode.enabled
        ? systemPromptEn
        : '''당신은 어르신용 길안내 도우미 '길벗'입니다.

[절대 규칙]
1. "동/서/남/북", "OO미터", "OO도 회전" 추상 표현 금지
2. 사진에 직접 보이는 경로 관련 사물(간판/출구/나가는 곳/갈아타는 곳/매표소/승강장/계단/표지판/건물 입구)을 가리키세요
3. 한 번에 하나만, 따뜻한 말투 ("~하시면 돼요"/"~보이시죠?")
4. 단서 부족하면 추측 말고 재촬영 요청
5. 끝에 안심 멘트 ("잘 가고 계세요"/"조금만 더 가시면 돼요")
6. **쉬운 말만**. 한자어 금지("가시적","식별","확인","이동","진행" 등 X). 대신 "보이는/보이시죠/가시면 돼요/걸어가세요/옆/앞/조심해서/천천히"
7. 사진 글자 100% 확신 못하면 추측 금지 — "노란 글자"같이 묘사
8. 계단/에스컬레이터/횡단보도엔 안전 멘트 ("손잡이 꼭 잡으세요"/"천천히")
9. **사용자 메시지의 출구번호/지명**(예 "잠실역 10번 출구")**은 instruction 에 그대로 활용**
10. 왼쪽/오른쪽/올라가세요/내려가세요는 사진 속 화살표, 표지판, 계단 방향이 명확할 때만 말한다.
11. "저쪽", "그쪽", "저기"처럼 기준점 없는 말은 금지. 구체 간판/출구/나가는 곳/갈아타는 곳/매표소/승강장/계단/문을 반드시 함께 말한다.
12. 의자/책상/소파/벽/바닥/천장/실내 가구만 보이면 길안내 단서가 아니다. 이런 사진은 방향을 만들지 말고 재촬영 요청.
13. 출력은 JSON 만. 사진에 목적지 간판이 명확하고 어르신이 입구 앞이면 is_arrival=true.''';

    // 목적지명을 prompt 에 명시 — system prompt 짧아져 LLM 이 도착 step 에서 목적지명을
    // 일반 묘사로 흘리는 회귀 방지 ("송파구보건소" → "잠실역 근처 건물").
    // home_screen 이 T-Map POI destPlace.name 으로 setDestinationName 을 호출했어야 함.
    // null 이면 destination 줄을 omit (하드코딩 fallback 금지 — 다른 목적지 시연을 막음).
    final destName = _lastDestinationName;
    // 스키마 최소화 — decode chunk 수가 곧 추론 시간 (S10e ~10 chunk/s). 이전
    // 스키마는 step_id / landmarks_in_photo 배열 / is_at_decision_point /
    // confidence / fallback_action 까지 7필드라 157 chunk(15초+) 생성. 코드가
    // 실제로 쓰는 건 instruction · is_arrival · 그리고 NO_FEATURE 가드용
    // landmarks_in_photo · confidence · fallback_action 이다. 운영 모드에서는
    // 사진 번호가 route step 이 아니므로 n/9 진행률을 prompt 에 넣지 않는다.
    //
    // DEMO_MODE: 영문 destLine / progressLine / userPrompt. fallback_action 도
    // 영어("retake requested") — home_screen 의 _isNoFeatureResult 가드가
    // 한·영 둘 다 받도록 같이 확장된다 (Stage B home_screen 분기).
    final String destLine;
    final String progressLine;
    final String userPrompt;
    if (DemoMode.enabled) {
      destLine = destName == null
          ? ''
          : '[Final destination] $destName (if this building or its sign is clearly visible in the photo, set is_arrival=true)\n';
      progressLine = totalSteps == null
          ? '[Current photo]\nDo NOT use the photo number to infer step or arrival.'
          : '[Step progress] ${stepHint + 1} / $totalSteps';
      userPrompt =
          '''$progressLine
$destLine
$stepContext

[Instruction] Analyze the attached photo and respond with ONE valid minified JSON object in exactly the same format as this example (replace the values with your own). arrow_tip_x and arrow_tail_x are normalized x coordinates (0=image left edge, 1=image right edge) of a horizontal arrow's tip and tail; fill them when a horizontal arrow is visible, otherwise leave null.

{"is_arrival":false,"landmarks_in_photo":["yellow exit sign","stairs going up"],"arrow_tip_x":null,"arrow_tail_x":null,"instruction":"I can see the exit sign directly ahead. Walk gently up the stairs and keep one hand on the handrail. You're doing great.","confidence":0.85,"fallback_action":null}

If the photo DOES contain a horizontal arrow, your instruction MUST contain exactly one of the words "left" or "right" — choose based on the arrow's tip direction. Examples of valid instruction phrasings for such photos (use ONE direction word, not both):
- "I can see the yellow exit sign. The arrow points to the LEFT — please walk to the left toward Exit 10. You're doing great."
- "I can see the yellow exit sign. The arrow points to the RIGHT — please walk to the right toward Exit 10. You're doing great."
Pick LEFT or RIGHT based on what you actually see in THIS photo; the app uses arrow_tip_x / arrow_tail_x to overwrite the word if your pixels disagree.''';
    } else {
      destLine = destName == null
          ? ''
          : '[최종 목적지] $destName (사진에 이 간판/입구가 보이면 is_arrival=true 와 함께 명시적 안내)\n';
      progressLine = totalSteps == null
          ? '[현재 사진]\n사진 번호는 경로 단계나 도착 판정에 사용하지 마세요.'
          : '[현재 진행 단계] ${stepHint + 1} / $totalSteps';
      userPrompt =
          '''$progressLine
$destLine
$stepContext

[지시] 첨부된 사진을 분석하고 아래 JSON 스키마로만, 짧게 응답. arrow_tip_x 와 arrow_tail_x 는 가로 화살표의 tip(뾰족한 끝)과 tail(평평한 끝)의 정규화 x 좌표(0=왼쪽 끝, 1=오른쪽 끝). 가로 화살표가 보이면 채우고, 없으면 null.

{
  "is_arrival": true|false,
  "landmarks_in_photo": ["사진에 실제로 보이는 구체 단서"],
  "arrow_tip_x": null,
  "arrow_tail_x": null,
  "instruction": "어르신께 들려드릴 한국어 안내문 (1~2문장, 짧게)",
  "confidence": 0.0~1.0,
  "fallback_action": null|"재촬영 요청"
}''';
    }

    // _shrinkImage 가 EXIF orientation 을 픽셀에 굽고 (S10e 카메라는 모든 사진을
    // 90CW 로 태깅), iPhone 5712x4284 같은 큰 사진만 _imageTargetWidth 로
    // 다운샘플한다 — 작은 사진은 native 해상도 유지, 업스케일 없음.
    final promptText = '$systemPrompt\n\n---\n\n$userPrompt';

    final shrunk = await _shrinkImage(
      photoBytes,
      targetWidth: _imageTargetWidth,
    );

    // 매 호출마다 fresh chat 세션 — 이전 step 의 image patches + KV cache 누적되면
    // S10e (5.5GB) 에서 두 번째 step 부터 OOM. close() 로 명시적 해제 필수.
    final freshChat = await _model!.createChat(
      temperature: 0.2,
      randomSeed: 1,
      topK: 64,
      topP: 0.95,
      supportImage: true,
    );
    try {
      await freshChat.addQueryChunk(
        Message.withImage(text: promptText, imageBytes: shrunk, isUser: true),
      );
      final response = (await freshChat.generateChatResponse()).toString();
      return _safeParseJson(response);
    } finally {
      await freshChat.close();
    }
  }

  /// 카메라 JPEG 를 vision 모델용으로 정규화한다.
  ///
  /// 반드시 `instantiateImageCodec` 으로 디코딩한다 — 이 코덱이 EXIF
  /// orientation 을 적용해 픽셀을 바로 세운다. S10e 카메라는 모든 사진을
  /// Orientation=6 (90CW) 로 태깅하므로, 디코딩을 건너뛰고 원본 JPEG 를 그대로
  /// 넘기면 모델이 90도 돌아간 이미지를 보고 좌우·상하 안내가 뒤집힌다
  /// (Follow-up 85 회귀).
  ///
  /// [targetWidth] 는 소스가 그보다 넓을 때만 적용 — 절대 업스케일하지 않는다
  /// (Follow-up 84). 결과는 orientation 이 픽셀에 구워진 PNG.
  Future<Uint8List> _shrinkImage(
    Uint8List src, {
    required int targetWidth,
  }) async {
    final srcWidth = _jpegWidth(src);
    final codec = (srcWidth != null && srcWidth > targetWidth)
        ? await ui.instantiateImageCodec(src, targetWidth: targetWidth)
        : await ui.instantiateImageCodec(src);
    final frame = await codec.getNextFrame();
    final byteData = await frame.image.toByteData(
      format: ui.ImageByteFormat.png,
    );
    frame.image.dispose();
    if (byteData == null) return src;
    return byteData.buffer.asUint8List();
  }

  /// JPEG 의 SOF 마커에서 픽셀 폭만 읽는다 (전체 디코딩 없이). JPEG 가 아니거나
  /// SOF 를 못 찾으면 null.
  int? _jpegWidth(Uint8List b) {
    final n = b.length;
    if (n < 4 || b[0] != 0xFF || b[1] != 0xD8) return null;
    var i = 2;
    while (i + 9 < n) {
      if (b[i] != 0xFF) {
        i++;
        continue;
      }
      final marker = b[i + 1];
      // SOF0/1/2/3 페이로드: [len:2][precision:1][height:2][width:2].
      if (marker >= 0xC0 && marker <= 0xC3) {
        return (b[i + 7] << 8) | b[i + 8];
      }
      if (marker == 0xDA) return null; // 스캔 데이터 도달 — SOF 없음
      // SOI/EOI/TEM/RSTn 은 길이 필드가 없다.
      if (marker == 0xD8 ||
          marker == 0xD9 ||
          marker == 0x01 ||
          (marker >= 0xD0 && marker <= 0xD7)) {
        i += 2;
        continue;
      }
      i += 2 + ((b[i + 2] << 8) | b[i + 3]); // 길이 있는 세그먼트
    }
    return null;
  }

  Future<void> close() async {
    _ready = false;
    await _model?.close();
    _model = null;
    _loraPath = null;
    _audioProxy = null;
    _cactusAsr = null;
  }

  void _ensureReady() {
    if (!_ready) {
      throw StateError('GemmaService.initialize() 먼저 호출');
    }
  }

  String? _extractDestinationCandidate(String text) {
    var cleaned = text
        .trim()
        .replaceAll(RegExp("^[\\\"'`]+|[\\\"'`]+\$"), '')
        .replaceAll(RegExp(r'\s+'), ' ');
    if (cleaned.isEmpty) return null;

    final destinationPatterns = <RegExp>[
      RegExp(r'(?:목적지|도착지)\s*(?:는|은|로|:)?\s*(.+)$'),
      RegExp(
        r'(.+?)(?:으로|로|까지|에)?\s*(?:가\s*자|가줘|가 줘|갈래|가고 싶|가고싶|안내해|안내 해|길안내|경로|찾아줘|찾아 줘)',
      ),
      RegExp(r'(.+?)(?:으로|로|까지|에)\s*$'),
    ];
    for (final pattern in destinationPatterns) {
      final match = pattern.firstMatch(cleaned);
      final candidate = match?.group(1)?.trim();
      if (candidate != null && candidate.isNotEmpty) {
        cleaned = candidate;
        break;
      }
    }

    cleaned = cleaned
        .replaceAll(RegExp(r'\s*(?:부탁해|부탁해요|주세요|줘|요|입니다|이야)[.!?。]*$'), '')
        .replaceAll(RegExp(r'[.!?。]+$'), '')
        .trim();
    return cleaned.isEmpty ? null : cleaned;
  }

  String? _stringOrNull(Object? value) {
    final text = value?.toString().trim();
    if (text == null || text.isEmpty || text.toLowerCase() == 'null') {
      return null;
    }
    return text;
  }

  Map<String, dynamic> _safeParseJson(String raw) {
    final start = raw.indexOf('{');
    final end = raw.lastIndexOf('}');
    if (start < 0 || end <= start) {
      debugPrint('[gemma] JSON 파싱 실패, raw=$raw');
      return {'_raw': raw};
    }
    try {
      final body = raw.substring(start, end + 1);
      return Map<String, dynamic>.from(jsonDecode(body) as Map);
    } catch (e) {
      debugPrint('[gemma] JSON decode 실패: $e, raw=$raw');
      return {'_raw': raw};
    }
  }
}
