import 'demo_mode.dart';

/// 사용자 노출 텍스트의 ko/en 분기.
///
/// `DemoMode.enabled` (= 심사위원 데모) 일 때 영문, 아니면 한글. 4일 마감
/// 기준 사용자 텍스트가 ~30~50개라 본격 i18n(ARB+gen-l10n) 파이프라인 대신
/// 단순 헬퍼로 처리 — 추가 텍스트가 생기면 이 파일에 getter/메서드만 추가.
///
/// 모델 출력 자체는 이 헬퍼와 무관 — DEMO_MODE 일 때 `gemma_service`가 영문
/// systemPrompt 를 사용해 모델이 영어를 직접 생성한다.
class Strings {
  const Strings._();

  static bool get _en => DemoMode.enabled;

  // ── 부팅 / Splash ─────────────────────────────────────────────────────
  static String get bootingStatus => _en ? 'Starting...' : '준비 중...';
  static String get permissionsCheck =>
      _en ? 'Checking permissions...' : '권한 확인 중...';
  static String get voiceSetup =>
      _en ? 'Setting up voice synthesis...' : '음성 합성 준비 중...';
  static String modelLoadFailed(Object e) =>
      _en ? 'Model load failed: $e' : '모델 로딩 실패: $e';

  /// 부팅 완료 후 첫 안내 — **화면 표시용**. 영문판은 브랜드를 한글 "길벗"
  /// 그대로 표기 (브랜드 인지도 + 시각적 정체성). TTS 음성 출력은 별도
  /// [welcomeIntroSpeak] 사용 — 영어 TTS 가 한글을 못 읽기 때문.
  static String get welcomeIntro => _en
      ? 'Welcome to 길벗. Tap the microphone below to play the sample destination request.'
      : '안녕하세요. 어디로 가실래요? 말하기 버튼을 누르고 말씀해 주세요.';

  /// Welcome intro — **TTS speak 전용**. 한글 "길벗" → 영어 음운 근사
  /// "Gil-but" 로 치환 (Samsung en-US TTS 가 한글을 "길비오스" 식으로 잘못
  /// 발음하는 사고 방지). "Gil-but" 은 "GIL but" 로 음절 분리되어 한국어
  /// [kil.bʌt] 에 가장 가깝게 들린다.
  static String get welcomeIntroSpeak => _en
      ? 'Welcome to Gil-but. Tap the microphone below to play the sample destination request.'
      : '안녕하세요. 어디로 가실래요? 말하기 버튼을 누르고 말씀해 주세요.';

  // ── Splash / 모델 로딩 ────────────────────────────────────────────────
  static String get loading =>
      _en ? 'Loading on-device Gemma 4 model... (~10s)' : '모델 로딩 중... (10초)';

  static String get loadingDownload => _en
      ? 'First launch: downloading ~2.4GB Gemma 4 model from Hugging Face. ~3-5 min on Wi-Fi.'
      : '첫 실행 — Hugging Face 에서 모델 다운로드 중 (~2.4GB, Wi-Fi 3-5분)';

  static String loadingProgress(int percent) =>
      _en ? 'Downloading model... $percent%' : '모델 다운로드 중... $percent%';

  static String get ready =>
      _en ? 'Ready. Tap the microphone to set your destination.' : '준비 완료';

  // ── 마이크 / 음성 입력 ───────────────────────────────────────────────
  static String get micTapToStart =>
      _en ? 'Tap the microphone to begin.' : '마이크를 누르고 말씀해 주세요';

  static String get listening => _en
      ? 'Listening... tap again when finished.'
      : '듣고 있어요... 말씀하신 뒤 다시 눌러주세요';

  static String get micPlayingDemo =>
      _en ? 'Playing the sample destination request...' : '샘플 발화 재생 중...';

  static String get analysingSpeech =>
      _en ? 'Analysing speech...' : '음성 분석 중...';

  static String get inferenceStatus =>
      _en ? 'Generating guidance...' : '안내 생성 중...';

  // ── BYO photo (DEMO_MODE long-press camera = gallery pick) ──────────
  static String get byoPickingPhoto =>
      _en ? 'Pick a photo from your gallery...' : '갤러리에서 사진 선택...';

  static String get byoAnalysing => _en
      ? 'Analysing your own photo with on-device Gemma 4...'
      : '직접 선택한 사진을 분석 중...';

  /// BYO mode replaces the canned step-context. GPS-less case: photo's EXIF
  /// has no GPS lock (스크린샷, GPS 꺼진 카메라, 또는 0/0 placeholder values).
  /// 위치를 알 수 없으므로 destination 방향 안내 금지 — off-route stepContext
  /// 와 같은 톤이지만 "GPS 없음" 사유로 분기. cherry-picking 방어상 GPS 없는
  /// 사진에서 "Exit 10 가세요" 같은 잘못된 안내가 가장 사고 위험 높다.
  static String get byoStepContext =>
      'You are looking at a photo the user picked from their gallery.\n'
      "The photo has NO GPS metadata (screenshot, GPS-off camera, or messenger-forwarded image), so we cannot verify whether it's on the user's planned walking route.\n"
      '\n'
      'REQUIRED instruction format (3 parts, in this order):\n'
      '1) START the instruction with EXACTLY this sentence: '
      '"I can\'t tell from this photo whether it\'s on your planned route to Songpa Health Center."\n'
      '2) Then ONE short sentence describing what is actually visible in THIS photo (objects only — do NOT borrow words from the example).\n'
      '3) STOP.\n'
      '\n'
      'FORBIDDEN — do NOT use any of these words/phrases in the instruction:\n'
      '"follow", "walk toward", "go to", "head toward", "if you are looking for", '
      '"you might want to", "take the stairs", "turn left/right", "Exit 10", "Jamsil".\n'
      '\n'
      'Example (copy ONLY the prefix sentence and the overall shape — REPLACE landmarks_in_photo and the description with what is actually visible in THIS photo):\n'
      '{"is_arrival":false,"landmarks_in_photo":["bench","tree","grass"],'
      '"instruction":"I can\'t tell from this photo whether it\'s on your '
      'planned route to Songpa Health Center. I can see a wooden bench under '
      'a leafy tree on green grass.","confidence":0.8,"fallback_action":null}\n'
      '\n'
      'ONE-SHOT — no questions, the user cannot reply.';

  /// BYO mode — photo EXIF GPS is FAR from the planned route. Describe-only,
  /// NO directional guidance toward the destination. Strict template with
  /// neutral example so the model copies the SHAPE (prefix + describe + stop)
  /// without leaking domain vocabulary into the actual response.
  ///
  /// **Cherry-picking 방지 노트**: 예시의 landmarks/instruction 은 의도적으로
  /// subway/exit/sign/stairs 같은 시연 사진 어휘를 피해서 일반 풍경(벤치/나무)
  /// 으로 골랐다. 모델이 예시 단어를 그대로 복사하는 cargo-cult 패턴 방지 —
  /// 실제 사진이 무엇이든 모델이 자기 vision 으로 묘사하도록 강제.
  static String byoStepContextOffRoute(int meters) =>
      'You are looking at a photo the user picked from their gallery.\n'
      "EXIF GPS shows it was taken about $meters meters away from the planned route — this photo is NOT on the user's actual walking route.\n"
      '\n'
      'REQUIRED instruction format (3 parts, in this order):\n'
      '1) START the instruction with EXACTLY this sentence: '
      '"This photo doesn\'t look like it\'s on your planned route to Songpa Health Center."\n'
      '2) Then ONE short sentence describing what is actually visible in THIS photo (objects only — do NOT borrow words from the example).\n'
      '3) STOP.\n'
      '\n'
      'FORBIDDEN — do NOT use any of these words/phrases in the instruction:\n'
      '"follow", "walk toward", "go to", "head toward", "if you are looking for", '
      '"you might want to", "take the stairs", "turn left/right".\n'
      '\n'
      'Example (copy ONLY the prefix sentence and the overall shape — REPLACE landmarks_in_photo and the description with what is actually visible in THIS photo):\n'
      '{"is_arrival":false,"landmarks_in_photo":["bench","tree","grass"],'
      '"instruction":"This photo doesn\'t look like it\'s on your planned route '
      'to Songpa Health Center. I can see a wooden bench under a leafy tree '
      'on green grass.","confidence":0.9,"fallback_action":null}\n'
      '\n'
      'ONE-SHOT — no questions, the user cannot reply.';

  /// BYO mode — photo EXIF GPS appears to be ON or near the planned route.
  /// Brief description + relevant guidance OK.
  static String byoStepContextOnRoute(int meters) =>
      'You are looking at a photo the user picked from their gallery. Its '
      'EXIF GPS shows it is about $meters meters from the planned route '
      '(Jamsil Station → Songpa Health Center) — likely on or near the '
      'actual route. Describe what you see in one or two warm sentences and '
      'give brief walking guidance toward Exit 10 / the destination if the '
      'photo shows a relevant scene. ONE-SHOT — do not ask any questions.';

  static String get speakerLabel => _en ? 'Korean speaker' : '한국어 발화';

  /// 마이크 버튼 idleLabel (재 발화용).
  static String get micRetryLabel => _en ? 'Speak\nagain' : '다시\n말하기';

  /// 마이크 버튼 라벨 (3-state).
  static String get micIdleLabel => _en ? 'Speak' : '말하기';
  static String get micRecordingLabel => _en ? 'Stop' : '끝내기';
  static String get micBusyLabel => _en ? 'Processing...' : '처리 중...';

  /// 카메라 버튼 라벨.
  static String get cameraLabel => _en ? 'Take photo' : '사진 찍기';

  /// 지도 버튼 라벨.
  static String get mapLabel => _en ? 'View\nmap' : '지도\n보기';

  /// DEMO_MODE 에서 모델 prompt 에 노출되는 영문 destination 명. Place.name 은
  /// 한국어 '송파구보건소' (지도 marker 용) 그대로 두고, prompt 만 영문 사용.
  static String get demoDestinationNameEn => 'Songpa Health Center';

  // ── 데모 카메라 preview ───────────────────────────────────────────────
  static String demoPreviewCaption(int n, int total) => _en
      ? 'Demo photo $n of $total\nThis image was pre-loaded for the demo. Tap the camera button below to send it to Gemma 4 for analysis.'
      : '데모 사진 $n / $total\n시연용 이미지입니다. 아래 카메라 버튼을 다시 눌러 분석을 시작해 주세요.';

  // ── NO_FEATURE / off-route / 사람-도움 ────────────────────────────────
  static String offRouteWarning(int meters) => _en
      ? 'You appear to be $meters meters off the planned route. Please head back to the original direction.'
      : '경로에서 $meters미터 벗어나신 것 같아요. 처음 안내드린 방향으로 돌아가세요.';

  static String get retryHint => _en
      ? "I couldn't find a clear route clue in this photo. Tap the camera again with a clearer view of an exit sign, stairs, or the destination building."
      : '사진에서 길안내 단서가 잘 안 보여요. 주변 사물이 잘 보이도록 다시 비춰주세요.';

  static String humanHelpMessage(String target) => _en
      ? "I'm having trouble reading the surroundings. Please ask someone nearby for directions to $target."
      : '주변 분께 $target 가는 길을 여쭤보세요.';

  static String get instructionFallback =>
      _en ? 'Could not generate guidance.' : '안내 생성 실패';

  // ── 라우팅 / 카메라 ───────────────────────────────────────────────────
  static String get findingDestination =>
      _en ? 'Looking up destination...' : '목적지 위치 찾는 중...';

  static String get gettingOrigin =>
      _en ? 'Getting current location...' : '현재 위치 확인 중...';

  static String get takePhotoHint =>
      _en ? 'Tap the camera when you need guidance.' : '길이 헷갈리시면 카메라를 눌러주세요';

  static String get analysingPhoto =>
      _en ? 'Running on-device Gemma 4 inference (~16s)...' : '사진 분석 중...';

  // ── 도착 / 마무리 ─────────────────────────────────────────────────────
  /// 도착 직후 farewell — **화면 표시용**. 한글 "길벗" 노출.
  /// TTS 출력은 [farewellAtSpeak] 사용.
  static String farewellAt(String destName) => _en
      ? 'Thank you for using 길벗. Have a good visit at $destName.'
      : '$destName 에 도착하셨어요. 오시느라 수고 많으셨어요.';

  /// Farewell — **TTS speak 전용**. "길벗" → "Gil-but" 치환.
  static String farewellAtSpeak(String destName) => _en
      ? 'Thank you for using Gil-but. Have a good visit at $destName.'
      : '$destName 에 도착하셨어요. 오시느라 수고 많으셨어요.';
}
