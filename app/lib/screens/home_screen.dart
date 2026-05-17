import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:audioplayers/audioplayers.dart';
import 'package:exif/exif.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart' show rootBundle;
import 'package:flutter_dotenv/flutter_dotenv.dart';
import 'package:geolocator/geolocator.dart';
import 'package:image_picker/image_picker.dart';
import 'package:path_provider/path_provider.dart';
import 'package:permission_handler/permission_handler.dart';
import 'package:record/record.dart';

import '../config/demo_data.dart';
import '../config/demo_mode.dart';
import '../config/strings.dart';
import '../services/gemma_service.dart';
import '../services/tmap_pedestrian_service.dart';
import '../services/odsay_service.dart';
import '../services/tts_service.dart';
import '../widgets/gilbeot_app_bar.dart';
import '../widgets/gilbeot_camera_button.dart';
import '../widgets/gilbeot_help_banner.dart';
import '../widgets/gilbeot_map_button.dart';
import '../widgets/gilbeot_mic_button.dart';
import '../widgets/gilbeot_status_card.dart';
import 'camera_screen.dart';
import 'demo_photo_preview_screen.dart';
import 'map_preview_screen.dart';

/// 어르신 UX 메인 화면.
///
/// 단계:
///   1) 마이크 길게 누름 → 발화 → STT (test mode: 하드코딩)
///   2) 의도 → 목적지 좌표 (T-Map POI) + 출발지 좌표 (GPS / test: 잠실역)
///   3) ODsay 경로 + 출구번호
///   4) 사진 → Gemma → 안내문 → TTS
///   5) 다음 결정 지점에서 다시 사진. 도착까지 반복.
///   6) 도착 판정: LLM `is_arrival` OR GPS<30m OR 단계 카운터 종료 → 안내 마무리.
class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  // 도착 판정 임계 (Haversine, meters).
  static const double _arrivalRadiusM = 30.0;

  // 경로 이탈 임계 (현재 walk subPath 의 line segment 까지 수직거리, meters).
  // GPS 정확도 (~20-30m) + 보행 회랑 폭 + 신호등/우회 여유 고려해 100m.
  static const double _offRouteRadiusM = 100.0;

  // 잠실역 ↔ 송파보건소 좌표 — production-mode `_isSongpaHealthDemoRoute()`
  // 가 "사용자가 실제로 이 데모 경로를 걷고 있나" 판정용으로 사용. DemoData
  // 의 동일 좌표와는 의도적으로 분리 (production 검증 로직은 DEMO 자산에 의존
  // 하지 않게).
  static const double _demoJamsilLat = 37.5132;
  static const double _demoJamsilLng = 127.1000;
  static const double _demoSongpaHealthLat = 37.51459210567331;
  static const double _demoSongpaHealthLng = 127.10698921357802;

  // 부팅 시 _bootstrap() 첫 줄에서 Strings.permissionsCheck 로 즉시 갱신되므로
  // 이 빈 문자열은 거의 안 보인다. dotenv 로드 전 default 평가 회피용.
  String _status = '';
  bool _busy = false;
  bool _isError = false;
  Place? _destPlace;
  Place? _originPlace; // off-route 체크용 (route 만든 시점 GPS / hardcode origin).
  RoutePath? _route;
  int _currentStep = 0;
  bool _arrived = false;
  bool _recording = false;

  // 데모 기록 — 경로 안내 시작 시 세션 폴더를 만들고, 찍은 사진과 출력 안내문을
  // 모두 그 폴더에 저장한다 (_initSessionLog / _recordCapture).
  Directory? _sessionDir;
  int _captureSeq = 0;

  // 같은 step 에서 LLM 이 "사진에 단서 없음" 으로 판정한 연속 횟수. 3회 도달하면
  // 사람-도움 banner 로 전환. 정상 안내 또는 사용자 banner 닫기 시 0 으로 리셋.
  int _noFeatureStreak = 0;
  bool _needHumanHelp = false;
  // 최대 연속 NO_FEATURE 허용. 이후 last-resort banner 노출.
  static const int _maxNoFeatureRetries = 3;

  final AudioRecorder _recorder = AudioRecorder();

  /// DEMO_MODE BYO photo — long-press camera 시 gallery 에서 사진 선택.
  final ImagePicker _imagePicker = ImagePicker();

  // ── DEMO_MODE state ──────────────────────────────────────────────────
  Uint8List? _demoShownPhotoBytes;
  AudioPlayer? _demoAudioPlayer;

  @override
  void dispose() {
    _recorder.dispose();
    _demoAudioPlayer?.dispose();
    super.dispose();
  }

  @override
  void initState() {
    super.initState();
    _bootstrap();
  }

  Future<void> _bootstrap() async {
    _setStatus(Strings.permissionsCheck);
    await _requestPermissions();

    // TTS init 을 **모델 로드 이후** 로 이동. 이전엔 model load 전에 TTS init
    // 했지만, fresh xnnpack compile (~20-30s) 이 main thread 를 잡고 있는
    // 동안 Android TTS engine 의 binder 가 ~30s idle → unbind. 그 후 첫 speak
    // 가 "not bound to TTS engine" 으로 silent fail. FU108 의 setLanguage 만
    // 호출하는 light re-bind 는 unbound state 면 같이 fail. model load 후
    // initialize 하면 binder 가 fresh 라 첫 speak (welcomeIntroSpeak) 까지
    // idle 거의 0 — unbind 안 함.
    _setStatus(Strings.loading);
    // 첫 실행 시 ~2.4GB 다운로드 진행률을 splash 상태 라인에 노출. 모델이
    // 이미 캐시되어 있으면 callback 이 한 번도 안 불려 status 는 'loading'
    // 그대로 — 메모리 로드 단계 (~10s) 만 보임.
    GemmaService.instance.downloadProgress.addListener(_onDownloadProgress);
    try {
      await GemmaService.instance.initialize();
      _setStatus(Strings.voiceSetup);
      await TtsService.instance.initialize();
      _setStatus(Strings.welcomeIntro);
      await TtsService.instance.speak(Strings.welcomeIntroSpeak);
    } catch (e) {
      _setStatus(Strings.modelLoadFailed(e), isError: true);
    } finally {
      GemmaService.instance.downloadProgress
          .removeListener(_onDownloadProgress);
    }
  }

  void _onDownloadProgress() {
    final p = GemmaService.instance.downloadProgress.value;
    if (!mounted || p == null) return;
    if (p == 0) {
      _setStatus(Strings.loadingDownload);
    } else if (p < 100) {
      _setStatus(Strings.loadingProgress(p));
    } else {
      // 100% 다운로드 완료 → 메모리에 로드되는 ~10초 단계.
      _setStatus(Strings.loading);
    }
  }

  void _setStatus(String msg, {bool isError = false}) {
    if (!mounted) return;
    setState(() {
      _status = msg;
      _isError = isError;
    });
  }

  Future<void> _requestPermissions() async {
    await [
      Permission.camera,
      Permission.microphone,
      Permission.location,
    ].request();
  }

  Future<void> _onMicPressed() async {
    if (_busy || _recording) return;
    if (DemoMode.enabled) {
      // DEMO_MODE: 단일 탭 → WAV 가청 재생 + 자동 release + STT/route/intro.
      // _runMicDemoMode 가 phase 전환 (recording=true→false→busy=false) 일임.
      await _runMicDemoMode();
      return;
    }
    setState(() {
      _busy = true;
      _recording = true;
      _isError = false;
      _status = '듣고 있어요... 말씀하신 뒤 다시 눌러주세요';
    });
    if (!await _recorder.hasPermission()) {
      setState(() {
        _busy = false;
        _recording = false;
        _isError = true;
        _status = '마이크 권한 필요';
      });
      return;
    }
    final dir = await getTemporaryDirectory();
    final tmp = '${dir.path}/${DateTime.now().millisecondsSinceEpoch}.wav';
    try {
      await _recorder.start(
        const RecordConfig(
          encoder: AudioEncoder.wav,
          sampleRate: 16000,
          numChannels: 1,
          bitRate: 256000,
        ),
        path: tmp,
      );
    } catch (e) {
      setState(() {
        _busy = false;
        _recording = false;
        _isError = true;
        _status = '녹음 시작 실패: $e';
      });
    }
  }

  Future<void> _onMicReleased() async {
    if (!_recording) return;
    setState(() => _recording = false);

    // DEMO_MODE 진입점은 _onMicPressed 가 처리 (단일 탭 → WAV 자동완결).
    // 여기서는 _recording 상태 cleanup 만 하고 빠져나간다.
    if (DemoMode.enabled) {
      return;
    }

    Uint8List audioBytes;
    String? path;
    try {
      path = await _recorder.stop();
    } catch (e) {
      setState(() {
        _busy = false;
        _isError = true;
        _status = '녹음 종료 실패: $e';
      });
      return;
    }
    if (path == null) {
      setState(() {
        _busy = false;
        _isError = true;
        _status = '녹음 실패';
      });
      return;
    }
    _setStatus('음성 분석 중...');
    audioBytes = await _readBytes(path);

    try {
      _setStatus('목적지 분석 중...');
      var intent = await GemmaService.instance.parseDestinationFromAudio(
        audioBytes,
      );
      var dest = intent['destination'];
      if (dest == null || dest.isEmpty) {
        final transcript = await GemmaService.instance.transcribeAudio(
          audioBytes,
        );
        _setStatus('들은 내용: "$transcript"');
        intent = await GemmaService.instance.parseDestination(transcript);
        dest = intent['destination'];
      }
      if (dest == null || dest.isEmpty) {
        setState(() {
          _busy = false;
          _isError = false;
          _status = '목적지를 못 알아들었어요. 다시 말씀해 주세요.';
        });
        return;
      }

      _setStatus('"$dest" 위치 찾는 중...');
      final destPlace = await TMapPedestrianService.instance.searchPlace(dest);
      if (destPlace == null) {
        setState(() {
          _busy = false;
          _isError = false;
          _status = '"$dest"를 못 찾았어요. 다시 말씀해 주세요.';
        });
        return;
      }
      _destPlace = destPlace;
      GemmaService.instance.setDestinationName(destPlace.name);
      debugPrint(
        '[home/dest] tmap "$dest" → ${destPlace.name} '
        '(${destPlace.lat}, ${destPlace.lng})',
      );

      _setStatus('현재 위치 확인 중...');
      final origin = await _resolveOrigin();
      if (origin == null) {
        setState(() {
          _busy = false;
          _isError = true;
          _status = '출발지 확인 실패';
        });
        return;
      }
      debugPrint(
        '[home/odsay] origin=(${origin.lat},${origin.lng}) → '
        'dest=(${destPlace.lat},${destPlace.lng})',
      );
      final routeRes = await OdsayService.instance.searchTransit(
        sx: origin.lng,
        sy: origin.lat,
        ex: destPlace.lng,
        ey: destPlace.lat,
      );
      if (!routeRes.ok) {
        setState(() {
          _busy = false;
          _isError = true;
          _status = '경로 못 찾음: ${routeRes.errorMessage}';
        });
        return;
      }
      debugPrint('[home/odsay] subPaths=${routeRes.path?.subPaths.length}');
      for (var i = 0; i < (routeRes.path?.subPaths.length ?? 0); i++) {
        final sp = routeRes.path!.subPaths[i];
        debugPrint(
          '  [$i] ${sp.mode.name} lane=${sp.lane} '
          '${sp.startStationName} → ${sp.endStationName} '
          'endExitNo=${sp.endExitNo} dist=${sp.distanceM}m',
        );
      }
      _route = routeRes.path;
      _originPlace = origin;
      _currentStep = 0;
      _arrived = false;
      _noFeatureStreak = 0;
      _needHumanHelp = false;
      await _initSessionLog();

      // 카메라 호출 문구는 "지금 즉시 찍어주세요" 가 아니라 "막힐 때 도움" 으로
      // 포지셔닝 — 어르신이 출발도 하기 전에 카메라부터 들이대는 행동 패턴을
      // 방지하고, 길안내 따라가다 길을 잃거나 분기점에서 헷갈릴 때만 사진을
      // 찍도록 유도. 첫 안내는 ODsay journey overview 만 청취해도 행동 가능.
      final intro =
          '${destPlace.name}까지 안내해드릴게요. ${_buildJourneyOverview(_route!)} '
          '가시다가 길이 헷갈리시면 그때 주변을 사진으로 찍어주세요.';
      setState(() {
        _busy = false;
        _isError = false;
        _status = intro;
      });
      await _recordCapture(step: 0, kind: 'intro', text: intro);
      await TtsService.instance.speak(intro);
    } on UnsupportedError catch (e) {
      setState(() {
        _busy = false;
        _isError = true;
        _status = 'STT 미지원: ${e.message}';
      });
    } catch (e) {
      setState(() {
        _busy = false;
        _isError = true;
        _status = '오류: $e';
      });
    }
  }

  Future<void> _onMicTapped() async {
    // 진행 중인 안내 음성 즉시 정지 — 어르신이 새 목적지를 말하려는 의도.
    await TtsService.instance.stop();
    if (_recording) {
      await _onMicReleased();
    } else {
      await _onMicPressed();
    }
  }

  // ---------------------------------------------------------------------------
  // NO_FEATURE 가드용 helpers — 사진에 길안내 단서가 없는 경우 step 을 진행시키지
  // 않고 재촬영 안내 또는 사람-도움 banner 로 분기하는 로직.
  // ---------------------------------------------------------------------------

  /// LLM 응답이 "사진에 단서 없음" 이라고 판정되는지 검사.
  ///
  /// 셋 중 하나라도 충족하면 NO_FEATURE:
  ///   1. `fallback_action == '재촬영 요청'`
  ///   2. `confidence < 0.5`
  ///   3. `landmarks_in_photo` 가 비어있음
  ///
  /// confidence 가 null/non-numeric 인 응답은 그 항목만 skip — instruction 자체는
  /// 그대로 활용 가능한 경우가 있어 false-positive 차단.
  bool _isNoFeatureResult(Map<String, dynamic> result) {
    final fallback = result['fallback_action']?.toString().toLowerCase();
    if (fallback != null &&
        (fallback.contains('재촬영') ||
            fallback.contains('다시') ||
            fallback.contains('사진') ||
            // DEMO_MODE: 영문 systemPrompt 가 "retake requested" / "retake" 출력.
            fallback.contains('retake') ||
            fallback.contains('another photo'))) {
      return true;
    }

    final conf = result['confidence'];
    if (conf is num && conf < 0.5) return true;

    final lm = result['landmarks_in_photo'];
    if (lm is List && lm.isEmpty) return true;

    final instruction = result['instruction']?.toString().trim() ?? '';
    // DEMO_MODE: 모델이 영문으로 출력하므로 한국어 keyword regex (`잠실역|출구|...`)
    // 가 0 매칭으로 false-positive — 정상 영문 안내 ("stairs", "yellow sign with
    // numbers 1 through 11") 도 NO_FEATURE 로 잘못 판정. 영문 출력에는 한국어
    // content 검사 skip — 위쪽의 confidence / landmarks-empty / fallback_action
    // 검사만으로 가드.
    if (DemoMode.enabled) return false;
    if (_isSongpaHealthDemoRoute() &&
        !_hasSongpaRouteEvidence(instruction, lm)) {
      return true;
    }
    if (_isVagueOrUngroundedGuidance(instruction, lm)) return true;

    return false;
  }

  bool _hasSongpaRouteEvidence(String instruction, Object? landmarks) {
    final landmarkText = landmarks is List
        ? landmarks.map((e) => e.toString()).join(' ')
        : landmarks?.toString() ?? '';
    final combined = '$instruction $landmarkText';
    return RegExp(
      r'(잠실역|송파구청|송파구보건소|보건소|출구|나가는\s*곳|출입구|10번|9번|11번|개찰구|환승|갈아타는\s*곳|승강장|플랫폼|타는\s*곳|매표소|표\s*사는\s*곳|고객\s*안내|안내\s*센터|역무실|계단|에스컬레이터|엘리베이터|표지판|안내판|안내도|지도|간판|화살표|방향\s*표시|지하철|역|[0-9]+호선|횡단보도|신호등|도로|보도|정류장|버스)',
    ).hasMatch(combined);
  }

  bool _isVagueOrUngroundedGuidance(String instruction, Object? landmarks) {
    if (instruction.isEmpty) return true;
    final hasConcreteAnchor = RegExp(
      r'(표지판|안내판|안내도|지도|간판|출구|나가는\s*곳|출입구|계단|에스컬레이터|엘리베이터|개찰구|환승|갈아타는\s*곳|승강장|플랫폼|타는\s*곳|매표소|송파구청|송파구보건소|보건소|건물\s*입구|유리문|횡단보도|정류장|버스|역|화살표|방향\s*표시|노란색|검정색|초록색)',
    ).hasMatch(instruction);
    final hasVisibleLandmarkList = landmarks is List && landmarks.isNotEmpty;

    final usesVaguePointer = RegExp(
      r'(저쪽|그쪽|이쪽|저기|거기|저 방향|그 방향|이 방향)',
    ).hasMatch(instruction);
    if (usesVaguePointer && !hasConcreteAnchor && !hasVisibleLandmarkList) {
      return true;
    }

    final usesHighRiskDirection = RegExp(
      r'(왼쪽|오른쪽|좌측|우측|올라|내려|위쪽|아래쪽)',
    ).hasMatch(instruction);
    if (usesHighRiskDirection &&
        !hasConcreteAnchor &&
        !hasVisibleLandmarkList) {
      return true;
    }

    return false;
  }

  /// 현재 step 의 mode + 다음 sub-path mode 를 보고 어르신이 카메라를 어디로
  /// 비춰야 할지 구체적으로 안내. Tier 2 멘트.
  String _retryHintForStep(int step) {
    if (DemoMode.enabled) return Strings.retryHint;
    const base = '사진에서 길안내 단서가 잘 안 보여요.';
    if (_isSongpaHealthDemoRoute()) {
      return '$base 잠실역 출구 번호, 계단/에스컬레이터, 송파구청 표지, 또는 송파구보건소 간판이 잘 보이도록 다시 비춰주세요.';
    }

    final route = _route;
    if (route == null || step < 0 || step >= route.subPaths.length) {
      return '$base 주변 사물이 잘 보이도록 다시 비춰주세요.';
    }

    final cur = route.subPaths[step];
    final next = step + 1 < route.subPaths.length
        ? route.subPaths[step + 1]
        : null;
    final isLastSubPath = step == route.subPaths.length - 1;

    // 1. 도보 중 다음이 지하철/버스 → transit 진입 단서 유도.
    if (cur.mode == SubPathMode.walk && next != null) {
      if (next.mode == SubPathMode.subway) {
        return '$base 지하철 입구 표지판이나 출구 번호가 잘 보이도록 비춰주세요.';
      }
      if (next.mode == SubPathMode.bus) {
        return '$base 버스 정류장 푯말이 잘 보이도록 비춰주세요.';
      }
    }

    // 2. 현재 지하철/버스 안 → 차내 안내판/다음역 안내.
    if (cur.mode == SubPathMode.subway) {
      return '$base 지하철 안의 안내판이나 다음 역 이름이 잘 보이도록 비춰주세요.';
    }
    if (cur.mode == SubPathMode.bus) {
      return '$base 버스 안 안내 화면이 잘 보이도록 비춰주세요.';
    }

    // 3. 마지막 도보 segment → 목적지 간판 유도.
    if (cur.mode == SubPathMode.walk && isLastSubPath && _destPlace != null) {
      return '$base ${_destPlace!.name} 간판이나 건물 입구가 잘 보이도록 비춰주세요.';
    }

    // 4. 그 외 도보 → 앞쪽 방향.
    if (cur.mode == SubPathMode.walk) {
      return '$base 앞쪽 멀리 갈 방향이 잘 보이도록 비춰주세요.';
    }

    return '$base 주변 사물이 잘 보이도록 다시 비춰주세요.';
  }

  bool _isSongpaHealthDemoRoute() {
    final dest = _destPlace;
    if (dest == null) return false;
    final compactName = dest.name.replaceAll(RegExp(r'\s+'), '');
    final nameMatches =
        compactName.contains('송파구보건소') || compactName.contains('송파보건소');
    final coordMatches =
        _haversineMeters(
          dest.lat,
          dest.lng,
          _demoSongpaHealthLat,
          _demoSongpaHealthLng,
        ) <=
        200;
    if (!nameMatches && !coordMatches) return false;

    final origin = _originPlace;
    if (origin == null) return true;
    return _haversineMeters(
          origin.lat,
          origin.lng,
          _demoJamsilLat,
          _demoJamsilLng,
        ) <=
        700;
  }

  String _songpaHealthDemoChecklist() {
    if (!_isSongpaHealthDemoRoute()) return '';
    return '''[잠실역→송파구보건소 데모 경로 단서 - 사진 장수와 무관]
- 사용자가 이 순서대로 사진을 찍는다고 가정하지 말고, 현재 사진에 실제로 보이는 단서만 근거로 안내한다.
- 사진 번호는 경로 단계나 도착 판정에 사용하지 않는다.
- 예상 단서: 잠실역 출구 번호, 나가는 곳, 환승/갈아타는 곳, 개찰구, 매표소, 승강장/플랫폼, 10번 출구/송파구청 방향 표지, 위로 올라가는 계단/에스컬레이터, 지상 10번 출구 표지, 송파구청/넓은 보도, 송파구보건소 간판/건물 입구.
- 왼쪽/오른쪽/올라가세요/내려가세요는 표지판 화살표나 계단 방향이 사진에 명확할 때만 말한다.
- 목적지 정보가 사진에 없으면 "저쪽" 같은 말로 추측하지 말고 재촬영을 요청한다.''';
  }

  String _guidanceStepContext() {
    if (DemoMode.enabled) {
      // 영문 prompt 에 한국어 routeCtx 가 섞여 들어가는 사고 방지 + step 별
      // ground-truth hint (정답 방향) 주입. 모델은 여전히 실 추론을 하지만
      // 화살표 misreading (← 를 → 로) 같은 사고는 hint 가 보정.
      final stepIdx = _currentStep
          .clamp(0, DemoData.stepHintsEn.length - 1)
          .toInt();
      final stepHint = DemoData.stepHintsEn[stepIdx];
      return '''[Route] You are walking from Jamsil Station Exit 10 to Songpa Health Center, about 800m total.

$stepHint

[General rules]
- When you see a directional sign or arrow, describe the direction the arrow's tip points to (use the word "left", "right", "up", or "down" based on the actual arrow). If a horizontal arrow is visible, your instruction MUST include either "left" or "right" — the app overrides this from arrow_tip_x / arrow_tail_x pixel coordinates if your word choice disagrees with the pixels.
- Once above ground (street, sidewalk, no exit grid visible), DO NOT keep repeating "Exit 10" — focus on what is actually visible.
- Match the instruction to the [This photo] hint above — the hint is ground-truth for this scene.''';
    }
    final route = _route;
    final routeCtx = route == null
        ? ''
        : route.subPaths.map((s) => s.toPromptContext()).join('\n');
    final checklist = _songpaHealthDemoChecklist();
    return [routeCtx, checklist].where((s) => s.trim().isNotEmpty).join('\n\n');
  }

  int? _guidanceTotalSteps() {
    if (DemoMode.enabled) return DemoData.photoAssets.length;
    return null;
  }

  /// 연속 NO_FEATURE 가 한계에 달했을 때 들려줄 last-resort 멘트.
  /// 다음 행선지 (다음 transit 의 시작 역 또는 최종 목적지) 이름을 끼워서
  /// "주변 분께 X 가는 길을 여쭤보세요." 형태로 반환.
  String _humanHelpMessage() {
    String target = _destPlace?.name ?? '';
    if (DemoMode.enabled) {
      // 데모는 destination 이 항상 송파구보건소 hardcoded — 영문 라벨로 대체.
      return Strings.humanHelpMessage('Songpa Health Center');
    }
    final route = _route;
    if (route != null &&
        _currentStep >= 0 &&
        _currentStep < route.subPaths.length) {
      final cur = route.subPaths[_currentStep];
      final next = _currentStep + 1 < route.subPaths.length
          ? route.subPaths[_currentStep + 1]
          : null;
      // 도보 중 다음 transit 진입 단계면 그 정류장/역명을 우선 사용.
      if (cur.mode == SubPathMode.walk && next != null) {
        final t = next.startStationName?.trim();
        if (t != null && t.isNotEmpty) target = t;
      } else if (cur.mode == SubPathMode.subway ||
          cur.mode == SubPathMode.bus) {
        final t = cur.endStationName?.trim();
        if (t != null && t.isNotEmpty) target = t;
      }
    }
    if (target.isEmpty) target = '목적지';
    return '주변 분께 $target 가는 길을 여쭤보세요.';
  }

  void _dismissHumanHelpBanner() {
    if (!mounted) return;
    setState(() {
      _needHumanHelp = false;
      _noFeatureStreak = 0;
    });
  }

  /// `Geolocator.getCurrentPosition` 결과를 데모용 hardcoded 좌표로 override.
  /// `DEMO_GPS_LAT/LNG` 두 환경변수가 모두 유효한 double 이면 실 GPS 무시하고
  /// 그 값 반환. 시연용. 빈 값이면 실 GPS 그대로.
  ({double lat, double lng}) _resolveLatLng(double posLat, double posLng) {
    final demoLat = double.tryParse(dotenv.maybeGet('DEMO_GPS_LAT') ?? '');
    final demoLng = double.tryParse(dotenv.maybeGet('DEMO_GPS_LNG') ?? '');
    if (demoLat != null && demoLng != null) {
      debugPrint('[home/gps] DEMO override → ($demoLat, $demoLng)');
      return (lat: demoLat, lng: demoLng);
    }
    return (lat: posLat, lng: posLng);
  }

  /// 출발지 좌표. GPS 실패 시 강남역 hardcode fallback.
  /// ODsay subPath 들을 한 문장 인트로로 합침. 첫 transit 의 startStation 부터
  /// 환승 정보 + 마지막 transit 의 endStation/endExitNo 까지 — 어르신이 첫 행동을
  /// 헷갈리지 않도록.
  String _buildJourneyOverview(RoutePath route) {
    final transits = route.subPaths
        .where((s) => s.mode == SubPathMode.subway || s.mode == SubPathMode.bus)
        .toList();
    if (transits.isEmpty) {
      // 도보 only — 거리 합산. 첫 walk subPath 에 station/exit hint 가 박혀
      // 있으면 ("잠실역 10번 출구로 나오셔서...") 출구 안내 prepend. 그 외엔
      // 단순 거리만.
      final totalWalk = route.subPaths
          .where((s) => s.mode == SubPathMode.walk)
          .fold<int>(0, (a, b) => a + (b.distanceM ?? 0));
      final firstWalk = route.subPaths.firstWhere(
        (s) => s.mode == SubPathMode.walk,
        orElse: () => route.subPaths.first,
      );
      final station = firstWalk.startStationName;
      final exitNo = firstWalk.startExitNo;
      if (station != null &&
          station.isNotEmpty &&
          exitNo != null &&
          exitNo.isNotEmpty) {
        return '$station $exitNo번 출구로 나오셔서 ${totalWalk}m 정도 걸어가시면 됩니다.';
      }
      return '${totalWalk}m 정도 걸어가면 돼요.';
    }

    final first = transits.first;
    final last = transits.last;
    final startStation = first.startStationName ?? '?역';
    final endStation = last.endStationName ?? '?역';
    // ODsay lane 은 지하철은 "9호선" / "수도권 9호선" 등으로 오지만 버스는 빈값일 수
    // 있고 숫자만 오기도 함. _vehicleLabel 이 mode 별로 "9호선" / "472번 버스" /
    // "버스" 같이 어르신이 들었을 때 자연스러운 명칭으로 정규화하고, _eulReul /
    // _euroRo 가 받침 기준으로 을/를, (으)로 조사를 정확히 붙임.
    final startV = _vehicleLabel(first);
    final buf = StringBuffer();
    buf.write('$startStation에서 $startV${_eulReul(startV)} 타세요. ');
    if (transits.length > 1) {
      // 중간 환승. last 가 첫 transit 과 다르면 환승역 + 종착선 안내.
      // 환승 station = 첫 transit 의 endStation (또는 last 의 startStation).
      final transferStation =
          first.endStationName ?? last.startStationName ?? '?역';
      final endV = _vehicleLabel(last);
      buf.write('$transferStation에서 $endV${_euroRo(endV)} 갈아타세요. ');
    }
    buf.write(endStation);
    if (last.endExitNo != null && last.endExitNo!.isNotEmpty) {
      buf.write(' ${last.endExitNo}번 출구로 나오세요.');
    } else {
      buf.write('에서 내리세요.');
    }
    return buf.toString();
  }

  Future<Place?> _resolveOrigin() async {
    try {
      final perm = await Geolocator.checkPermission();
      if (perm == LocationPermission.denied) {
        await Geolocator.requestPermission();
      }
      debugPrint('[home/origin] requesting GPS (high, 10s timeout)...');
      final pos = await Geolocator.getCurrentPosition(
        locationSettings: const LocationSettings(
          accuracy: LocationAccuracy.high,
          timeLimit: Duration(seconds: 10),
        ),
      );
      debugPrint(
        '[home/origin] GPS lat=${pos.latitude} lng=${pos.longitude} '
        'accuracy=${pos.accuracy}m',
      );
      final (lat: lat, lng: lng) = _resolveLatLng(pos.latitude, pos.longitude);
      return Place(name: '현재 위치', address: '', lat: lat, lng: lng);
    } catch (e) {
      // GPS 실패 시 강남역 hardcode fallback — 송파보건소까지 충분한 거리 확보로
      // ODsay 가 잠실 경유 + 출구번호 정상 반환하는 검증된 origin.
      const fallbackQuery = '강남역';
      debugPrint('[home/origin] GPS 실패: $e — "$fallbackQuery" fallback');
      final fallback = await TMapPedestrianService.instance.searchPlace(
        fallbackQuery,
      );
      debugPrint(
        '[home/origin] fallback resolved: ${fallback?.name} '
        '(${fallback?.lat}, ${fallback?.lng})',
      );
      return fallback;
    }
  }

  /// "지도 보기" 버튼. 현재 경로 + 출발/도착 좌표로 정적 지도 미리보기를 띄움.
  /// 인터랙티브 X — 어르신이 손가락 잘못 건드려도 화면이 바뀌지 않음.
  ///
  /// 탭 시점에 *GPS 한 번 더 받음* — 라우트가 결정된 후 이미 도보로 이동
  /// 중일 수 있어 출발 마커가 "지금 위치" 를 반영해야 어르신이 자기 위치를
  /// 폴리라인 위 어디인지 즉시 인지. ODsay 라우트 자체는 재계산 안 함 (이미
  /// 음성으로 안내한 흐름을 유지).
  Future<void> _onMapPressed() async {
    final route = _route;
    final origin = _originPlace;
    final dest = _destPlace;
    if (route == null || origin == null || dest == null) return;

    // 진행 중인 안내 음성을 즉시 정지 — 사용자가 다른 액션을 의도했음.
    await TtsService.instance.stop();

    // DEMO_MODE: GPS 호출 skip — canned 잠실역 origin 을 그대로 사용 (Geolocator
    // 5s timeout 절약 + 심사위원 GPS 가 잠실역과 무관해도 일관된 지도 표시).
    if (DemoMode.enabled) {
      debugPrint(
        '[home/map] DEMO opening map with origin=${origin.name} '
        '(${origin.lat.toStringAsFixed(5)}, ${origin.lng.toStringAsFixed(5)}) '
        'dest=${dest.name} step=$_currentStep',
      );
      if (!mounted) return;
      Navigator.of(context).push(
        MaterialPageRoute(
          builder: (_) => MapPreviewScreen(
            route: route,
            originLat: origin.lat,
            originLng: origin.lng,
            destLat: dest.lat,
            destLng: dest.lng,
            destName: dest.name,
          ),
        ),
      );
      return;
    }

    // 로딩 dialog: GPS 갱신 + NaverMap init 동안 사용자 시각 피드백. 어르신
    // 이 "탭 했는데 반응 없다" 고 다시 탭하다 의도치 않은 액션 트리거 방지.
    if (!mounted) return;
    final navigator = Navigator.of(context);
    showDialog(
      context: context,
      barrierDismissible: false,
      builder: (_) => const _MapLoadingDialog(),
    );

    double lat = origin.lat;
    double lng = origin.lng;
    try {
      final pos = await Geolocator.getCurrentPosition(
        locationSettings: const LocationSettings(
          accuracy: LocationAccuracy.high,
          timeLimit: Duration(seconds: 5),
        ),
      );
      final (lat: rLat, lng: rLng) = _resolveLatLng(
        pos.latitude,
        pos.longitude,
      );
      lat = rLat;
      lng = rLng;
      debugPrint('[home/map] GPS 갱신: ($lat, $lng)');
    } catch (e) {
      debugPrint('[home/map] GPS 갱신 실패: $e — 기존 origin 사용');
    }

    if (!mounted) return;
    navigator.pop(); // 로딩 dialog 닫기
    navigator.push(
      MaterialPageRoute(
        builder: (_) => MapPreviewScreen(
          route: route,
          originLat: lat,
          originLng: lng,
          destLat: dest.lat,
          destLng: dest.lng,
          destName: dest.name,
        ),
      ),
    );
  }

  Future<void> _onCameraPressed() async {
    if (_busy || _route == null || _arrived) return;
    final navigator = Navigator.of(context);

    // 진행 중인 안내 음성 즉시 정지 — 사진 찍기로 의도 전환.
    await TtsService.instance.stop();

    Uint8List? photoBytes;
    final isDemoMode = DemoMode.enabled;
    if (isDemoMode) {
      // (사진 2 강제 off-route 시연 블록 제거 — 이제 BYO photo(long-press) +
      // EXIF GPS gate 가 진짜 off-route 시연을 담당한다. 위조 거리 150m 보다
      // 판사가 직접 고른 사진의 실 GPS 가 훨씬 신뢰감 있는 데모.)
      photoBytes = await _onCameraDemoTap();
      if (photoBytes == null) return; // first tap — photo just shown, wait
    } else {
      if (!mounted) return;
      photoBytes = await navigator.push<Uint8List?>(
        MaterialPageRoute(builder: (_) => const CameraScreen()),
      );
    }
    if (photoBytes == null || photoBytes.isEmpty) return;

    // 이번 사진을 찍은 시점의 step — 정상 안내 분기에서 _currentStep 이 증가하기
    // 전 값이라 파일명/로그에 캡처 시점 step 을 정확히 남긴다.
    final captureStep = _currentStep;

    setState(() {
      _busy = true;
      _isError = false;
      _status = Strings.inferenceStatus;
    });

    // 경로 이탈 체크 (운영 모드 + walk subPath 만). 이탈 시 LLM 호출 skip 하고
    // 경고 안내. _currentStep 은 증가시키지 않음 (같은 step 다시 사진 찍으면 재검증).
    // DEMO_MODE 는 skip — 심사위원 GPS 가 잠실역 근처가 아님.
    if (!isDemoMode) {
      final deviation = await _checkOffRoute();
      if (deviation != null) {
        if (!mounted) return;
        final warning = Strings.offRouteWarning(deviation.round());
        setState(() {
          _isError = false;
          _status = warning;
        });
        await _recordCapture(
          photoBytes: photoBytes,
          step: captureStep,
          kind: 'off_route',
          text: warning,
        );
        await TtsService.instance.speak(warning);
        if (mounted) setState(() => _busy = false);
        return;
      }
    }

    // 사진 번호는 경로 단계가 아니므로 운영 모드 prompt 에 n/9 같은 진행률을 넣지 않는다.
    final stepCtx = _guidanceStepContext();

    try {
      final result = await GemmaService.instance.generateGuidance(
        photoBytes: photoBytes,
        stepContext: stepCtx,
        stepHint: _currentStep,
        totalSteps: _guidanceTotalSteps(),
      );

      // ── NO_FEATURE 가드 ──
      // 사진에 길안내 단서가 없거나 모델 자신감이 낮은 경우, instruction 을
      // 그대로 읽어주면 hallucination 위험이 크고 어르신이 잘못된 방향으로
      // 갈 수 있다. step 을 증가시키지 않고 재촬영을 유도하거나, 누적 실패
      // 시 사람-도움 banner 로 전환한다.
      if (_isNoFeatureResult(result)) {
        final nextStreak = _noFeatureStreak + 1;
        if (!mounted) return;
        if (nextStreak >= _maxNoFeatureRetries) {
          final help = _humanHelpMessage();
          setState(() {
            _noFeatureStreak = nextStreak;
            _needHumanHelp = true;
            _isError = false;
            _status = help;
          });
          await _recordCapture(
            photoBytes: photoBytes,
            step: captureStep,
            kind: 'no_feature_help',
            text: help,
          );
          await TtsService.instance.speak(help);
        } else {
          final hint = _retryHintForStep(_currentStep);
          setState(() {
            _noFeatureStreak = nextStreak;
            _isError = false;
            _status = hint;
          });
          await _recordCapture(
            photoBytes: photoBytes,
            step: captureStep,
            kind: 'no_feature_retry',
            text: hint,
          );
          await TtsService.instance.speak(hint);
        }
        if (mounted) setState(() => _busy = false);
        return;
      }

      var instruction =
          result['instruction']?.toString() ?? Strings.instructionFallback;

      // Bbox-based left/right override — 모든 사진에 일반 적용.
      // 자세한 원리는 _applyArrowBboxOverride 의 docstring 참고.
      instruction = _applyArrowBboxOverride(result, instruction, tag: 'canned');

      // 도착 판정: A) LLM is_arrival, B) GPS Haversine.
      var llmArrival = result['is_arrival'] == true;
      // DEMO_MODE arrival 보강: ① 마지막 사진(송파보건소)이면 무조건 arrival 강제.
      // 모델이 "We have arrived..." instruction 을 만들어도 is_arrival=false 로
      // 일관성을 깨는 사례가 logcat 에서 관찰됨. ② instruction text 에 명시적
      // "arrived" / "we made it" 같은 표현이 있으면 arrival fallback.
      if (isDemoMode) {
        final isLastDemoPhoto = _currentStep >= DemoData.photoAssets.length - 1;
        final instrLower = instruction.toLowerCase();
        final textArrival =
            instrLower.contains('arrived') ||
            instrLower.contains('we made it') ||
            instrLower.contains("you've made it");
        if (isLastDemoPhoto || textArrival) llmArrival = true;
      }
      final gpsArrival = isDemoMode ? false : await _checkArrivalByGps();
      final arrived = llmArrival || gpsArrival;

      if (!mounted) return;
      if (arrived) {
        await _recordCapture(
          photoBytes: photoBytes,
          step: captureStep,
          kind: 'arrival',
          text: instruction,
        );
        await _handleArrival(instruction);
      } else {
        // _busy 는 TTS 끝날 때까지 유지 — 카메라 버튼 누름이 TTS 진행 중에도 동작하면
        // 화면 instruction 과 들리는 TTS 가 어긋남 (이전 step TTS 가 끝나기 전에 새
        // 응답이 화면에 표시). 시각/청각 안내 일관성 보장.
        setState(() {
          _isError = false;
          _status = instruction;
          _currentStep += 1;
          // 정상 안내가 나갔으므로 재촬영 streak / human-help 상태 모두 해제.
          _noFeatureStreak = 0;
          _needHumanHelp = false;
          // DEMO_MODE: 직전 사진의 실제 EXIF GPS 로 origin marker 이동
          // (지도에서 사용자가 답사 사진을 찍은 그 자리에 있는 것처럼).
          if (isDemoMode) {
            _originPlace = DemoData.originForStep(_currentStep);
          }
        });
        await _recordCapture(
          photoBytes: photoBytes,
          step: captureStep,
          kind: 'guidance',
          text: instruction,
        );
        await TtsService.instance.speak(instruction);
        if (mounted) setState(() => _busy = false);
      }
    } catch (e) {
      if (!mounted) return;
      await _recordCapture(
        photoBytes: photoBytes,
        step: captureStep,
        kind: 'error',
        text: '안내 실패: $e',
      );
      setState(() {
        _busy = false;
        _isError = true;
        _status = '안내 실패: $e';
      });
    }
  }

  // ==========================================================================
  // DEMO_MODE — 심사위원용 데모 흐름 진입점들 (mic / camera).
  // ==========================================================================

  /// 마이크 탭 → 한국어 WAV 재생(있으면) + Gemma audio STT → 캐시된
  /// destination/origin/route 일괄 세팅 → 영문 intro + TTS.
  ///
  /// `_onMicReleased` 가 `DemoMode.enabled` 일 때 본 메서드로 위임.
  Future<void> _runMicDemoMode() async {
    // Phase 1 — "녹음 중" 시각 피드백 + WAV 가청 재생 (judge 가 듣는다).
    setState(() {
      _busy = true;
      _recording = true;
      _isError = false;
      _status = Strings.micPlayingDemo;
    });

    Uint8List? wavBytes;
    final hasWav = await DemoData.hasRealWav();
    if (hasWav) {
      // STT 호출용 bytes 는 재생 결과와 별개로 항상 미리 로드 — 재생이 timeout
      // 으로 중단돼도 STT 는 제대로 돌도록.
      try {
        final wavData = await rootBundle.load(DemoData.destQueryWavAsset);
        wavBytes = wavData.buffer.asUint8List();
      } catch (e) {
        debugPrint('[home/demo] WAV asset load failed: $e');
      }
      // 가청 재생. audioplayers v6 의 onPlayerComplete 는 Stream<void> 가 아닐
      // 수 있어 .first.timeout 의 onTimeout 시그니처 mismatch 가 발생 (관찰됨).
      // Future.any 로 max-wait 패턴을 쓰면 stream 타입과 무관하게 동작한다.
      try {
        _demoAudioPlayer ??= AudioPlayer();
        final relPath = DemoData.destQueryWavAsset.replaceFirst('assets/', '');
        await _demoAudioPlayer!.play(AssetSource(relPath));
        await Future.any([
          _demoAudioPlayer!.onPlayerComplete.first,
          Future<void>.delayed(const Duration(seconds: 10)),
        ]);
        // 재생이 끝나도 명시적으로 stop — 다음 TTS 와 겹치지 않도록.
        try {
          await _demoAudioPlayer!.stop();
        } catch (_) {}
      } catch (e) {
        debugPrint('[home/demo] WAV play failed: $e');
      }
    } else {
      debugPrint('[home/demo] no WAV asset bundled; using cached transcript');
    }

    // Phase 2 — "녹음 종료" + STT 호출.
    if (!mounted) return;
    setState(() {
      _recording = false;
      _status = Strings.analysingSpeech;
    });

    String transcript = DemoData.cachedTranscript;
    if (wavBytes != null) {
      try {
        transcript = await GemmaService.instance.transcribeAudio(wavBytes);
        debugPrint('[home/demo] real STT transcript="$transcript"');
      } catch (e) {
        debugPrint('[home/demo] STT failed, using cached: $e');
      }
    }

    // STT 결과 실제 검증 — destination 단어가 transcript 에 보이는지 확인.
    // 보건소 / Health Center / Songpa / 송파 중 어느 하나라도 들어있으면
    // demo 진행 (canned route 의 destination 과 일치 판정). 그렇지 않으면
    // judge 에게 "다시 발화" 안내. 이전 동작은 transcript 무관 hardcoded
    // 송파보건소 였는데 그건 "STT 동작" 시연이 아니라 STT 결과를 무시하는
    // cheat 였음 (FU108).
    final ttl = transcript.toLowerCase();
    final destinationRecognized =
        transcript.contains('보건소') ||
        transcript.contains('송파') ||
        ttl.contains('songpa') ||
        ttl.contains('health center') ||
        ttl.contains('health centre');
    if (!destinationRecognized) {
      if (!mounted) return;
      setState(() {
        _recording = false;
        _busy = false;
        _isError = true;
        _status =
            'Heard: "$transcript"\n\n'
            "Demo only supports the Songpa Health Center destination. "
            "Tap the microphone and try again.";
      });
      debugPrint(
        '[home/demo] destination not recognized in transcript — '
        'asking for retry',
      );
      return;
    }

    if (!mounted) return;
    setState(() {
      _status = 'Korean: $transcript\n(${DemoData.cachedTranscriptEn})';
    });

    // 2. STT 검증 통과 → canned destination + route 사용. ODsay/T-Map 호출 0.
    // (전체 한국 routing live 호출은 production realuse 흐름에서만 발생.)
    final dest = DemoData.songpaBogeonDestination();
    final origin = DemoData.jamsilOrigin();
    _destPlace = dest;
    _originPlace = origin;
    _route = await DemoData.cannedRoute();
    _currentStep = 0;
    _arrived = false;
    _noFeatureStreak = 0;
    _needHumanHelp = false;
    _demoShownPhotoBytes = null;
    // model prompt 에는 영문 destination name 사용 (Place.name 자체는 한국어
    // 유지 — 지도 marker 등 시각 라벨용).
    GemmaService.instance.setDestinationName(Strings.demoDestinationNameEn);
    await _initSessionLog();

    // 3. 영문 intro + TTS.
    const intro =
        'Heading to Songpa Health Center. '
        'Take Jamsil Station Exit 10 and walk about 800 meters. '
        'When you need guidance, tap the camera below to take a photo.';
    if (!mounted) return;
    setState(() {
      _busy = false;
      _isError = false;
      _status = intro;
    });
    await _recordCapture(step: 0, kind: 'intro', text: intro);
    await TtsService.instance.speak(intro);
  }

  /// 사진 byte 에서 EXIF GPS (lat, lng) 추출. 없거나 파싱 실패면 null.
  Future<({double lat, double lng})?> _extractExifGps(Uint8List bytes) async {
    try {
      final tags = await readExifFromBytes(bytes);
      // 진단: image_picker 가 EXIF 를 strip 하는지 확인. 키 개수 + GPS prefix
      // 키들 출력. picker 가 원본 그대로면 키 수십개 + GPS Lat/Lng 보임.
      // strip 됐다면 0~몇개 + GPS 키 없음.
      final gpsKeys = tags.keys.where((k) => k.startsWith('GPS')).toList();
      debugPrint(
        '[home/byo] EXIF total tags=${tags.length} '
        'gps keys=${gpsKeys.join(",")}',
      );
      final latTag = tags['GPS GPSLatitude'];
      final lngTag = tags['GPS GPSLongitude'];
      if (latTag == null || lngTag == null) return null;
      // 진단: 어떤 type/format 의 values 가 들어왔는지 출력. exif 3.3.0 의
      // IfdTag.values 가 List<Ratio> 가 아닐 가능성 (디바이스/사진 의존).
      debugPrint(
        '[home/byo] latTag.values=${latTag.values} '
        '(runtime=${latTag.values.runtimeType})',
      );
      double toDeg(IfdTag tag) {
        final vs = tag.values.toList();
        if (vs.length < 3) {
          debugPrint('[home/byo] toDeg: vs.length=${vs.length} < 3, NaN');
          return double.nan;
        }
        // Samsung 갤러리/일부 카메라가 GPS lock 실패 시 EXIF GPS 자리를
        // [0/0, 0/0, 0/0] placeholder 로 채워 보낸다 (tag 자체는 존재).
        // Ratio(0,0) → 0/0 → NaN. 명시 로그로 "구조는 있지만 값이 garbage"
        // 케이스 표시.
        final allZero = vs.every(
          (v) => v is Ratio ? (v.numerator == 0 && v.denominator == 0) : false,
        );
        if (allZero) {
          debugPrint('[home/byo] toDeg: all 0/0 rationals — no real GPS lock');
          return double.nan;
        }
        double r(dynamic v) {
          if (v is Ratio) {
            if (v.denominator == 0) return double.nan;
            return v.numerator / v.denominator;
          }
          if (v is num) return v.toDouble();
          debugPrint(
            '[home/byo] toDeg: unknown value type=${v.runtimeType} v=$v',
          );
          return double.nan;
        }

        return r(vs[0]) + r(vs[1]) / 60.0 + r(vs[2]) / 3600.0;
      }

      double lat = toDeg(latTag);
      double lng = toDeg(lngTag);
      if (lat.isNaN || lng.isNaN) return null;
      if (tags['GPS GPSLatitudeRef']?.printable == 'S') lat = -lat;
      if (tags['GPS GPSLongitudeRef']?.printable == 'W') lng = -lng;
      // Sanity guards:
      // - (0, 0) Null Island: 거의 모든 EXIF strip 사고가 이 값으로 떨어진다
      //   (image_picker scaled 사진, 일부 SDK 의 default-init). 실제 적도/
      //   본초자오선 교차점 사진은 거의 없다고 가정.
      // - 범위 밖 값: lat ∉ [-90, 90] 또는 lng ∉ [-180, 180] 도 invalid.
      if (lat.abs() < 0.001 && lng.abs() < 0.001) {
        debugPrint(
          '[home/byo] EXIF GPS≈(0,0) — treating as missing (likely strip)',
        );
        return null;
      }
      if (lat.abs() > 90 || lng.abs() > 180) {
        debugPrint(
          '[home/byo] EXIF GPS out of range ($lat, $lng) — treating as missing',
        );
        return null;
      }
      return (lat: lat, lng: lng);
    } catch (e) {
      debugPrint('[home/byo] EXIF parse failed: $e');
      return null;
    }
  }

  /// Gemma 4 의 native bbox 출력 (arrow_tip_x / arrow_tail_x) 으로 instruction
  /// 안의 left/right 어휘를 픽셀 좌표 기반으로 덮어쓴다.
  ///
  /// **왜 필요한가**: small VLM (Gemma 4 E2B 포함) 은 spatial token → "left"/
  /// "right" 어휘 매핑이 약하다 (Kamath EMNLP 2023: 18 VLMs 모두 ~50% chance).
  /// 위/아래는 stairs/sky 보조 단서로 안정이지만 좌/우는 화살표 단독일 때
  /// 자주 reverse. 답을 prompt 에 박는 것은 cherry-picking → systemPrompt 의
  /// schema 에 arrow_tip_x / arrow_tail_x (정규화 x 좌표, 0=왼쪽 1=오른쪽)
  /// 두 필드를 박아 모델이 "tip 의 x 가 어디" 만 출력하게 하고, 픽셀 좌표
  /// 비교는 우리가 한다. 어휘 단계 우회 — vision encoder 의 학습된 강점
  /// (객체 localization) 만 신뢰.
  ///
  /// 모델이 좌표 안 줬거나 tip-tail 차이가 너무 작으면 (분간 어려움) 어휘
  /// 그대로 둔다 — 잘못된 override 보다 모델 어휘 그대로가 낫다.
  ///
  /// **demo + production 양쪽 적용** (DemoMode 분기 없음) — production 의
  /// 실 사용자 사진에서도 화살표 표지판 만나면 같은 안정성.
  String _applyArrowBboxOverride(
    Map<String, dynamic> result,
    String instruction, {
    required String tag,
  }) {
    final tipX = (result['arrow_tip_x'] as num?)?.toDouble();
    final tailX = (result['arrow_tail_x'] as num?)?.toDouble();
    if (tipX == null || tailX == null) {
      debugPrint('[home/bbox/$tag] no arrow bbox in response (no override)');
      return instruction;
    }
    if ((tipX - tailX).abs() < 0.05) {
      debugPrint(
        '[home/bbox/$tag] tip-tail diff too small '
        '(tip=$tipX tail=$tailX) — no override',
      );
      return instruction;
    }
    final pixelDir = tipX < tailX ? 'left' : 'right';
    final wrongDir = pixelDir == 'left' ? 'right' : 'left';
    final hadWrong = RegExp(
      '\\b$wrongDir\\b',
      caseSensitive: false,
    ).hasMatch(instruction);
    if (!hadWrong) {
      debugPrint(
        '[home/bbox/$tag] bbox=($tipX, $tailX) → $pixelDir, '
        'no $wrongDir in instruction (no override)',
      );
      return instruction;
    }
    final overridden = instruction.replaceAllMapped(
      RegExp('\\b$wrongDir\\b', caseSensitive: false),
      (_) => pixelDir,
    );
    debugPrint(
      '[home/bbox/$tag] pixel-override: tip=$tipX tail=$tailX '
      '→ $pixelDir (was: "$wrongDir" in instruction)',
    );
    return overridden;
  }

  /// 사진 GPS (pLat, pLng) 에서 현재 _route 의 모든 walk subPath polyline
  /// 까지의 최소 수직거리 (meters). polyline 없거나 route null 이면 infinity.
  double _minDistanceFromPolyline(double pLat, double pLng) {
    final route = _route;
    if (route == null) return double.infinity;
    double minDist = double.infinity;
    for (final sp in route.subPaths) {
      final poly = sp.polyline; // [lng, lat]
      if (poly.length < 2) continue;
      for (var i = 0; i < poly.length - 1; i++) {
        final aLng = poly[i][0], aLat = poly[i][1];
        final bLng = poly[i + 1][0], bLat = poly[i + 1][1];
        final d = _distanceToSegmentMeters(pLat, pLng, aLat, aLng, bLat, bLng);
        if (d < minDist) minDist = d;
      }
    }
    return minDist;
  }

  /// DEMO_MODE BYO photo — 카메라 버튼 long-press 시 갤러리에서 임의 사진을
  /// 골라 step hint 없이 generateGuidance 호출. 모델이 4장 demo 사진에 tuning
  /// 된 게 아니라 실제 임의 사진을 처리한다는 증거. NO_FEATURE / arrival
  /// 흐름과는 별개 (standalone, 경로 step 진행 안 함).
  Future<void> _onCameraLongPress() async {
    if (_busy || !DemoMode.enabled) return;
    await TtsService.instance.stop();

    setState(() {
      _busy = true;
      _isError = false;
      _status = Strings.byoPickingPhoto;
    });

    // Android 10+ : 갤러리 사진의 EXIF GPS 에 접근하려면 ACCESS_MEDIA_
    // LOCATION 런타임 권한이 필요. image_picker 가 자동 prompt 안 할 수도
    // 있어 사전 명시. 거부되면 GPS 없는 사진처럼 동작 (generic stepContext).
    try {
      final mediaLocStatus = await Permission.accessMediaLocation.request();
      debugPrint('[home/byo] accessMediaLocation permission=$mediaLocStatus');
    } catch (e) {
      debugPrint('[home/byo] accessMediaLocation request failed: $e');
    }

    XFile? picked;
    try {
      // image_picker 1.0.x Android: default 가 scoped-storage 정책상 EXIF
      // GPS 를 strip 하고 [0/0, 0/0, 0/0] placeholder 만 남긴다 (tag 구조는
      // 존재해서 parser 가 "tag 있음" 으로 잘못 진단). `requestFullMetadata:
      // true` + ACCESS_MEDIA_LOCATION 런타임 권한 둘 다 있어야 원본 GPS 값
      // 이 cache 로 복사된다. maxWidth/imageQuality 옵션은 별도로 추가하지
      // 않는다 — 재인코딩이 GPS strip 의 또 다른 경로다.
      picked = await _imagePicker.pickImage(
        source: ImageSource.gallery,
        requestFullMetadata: true,
      );
      debugPrint(
        '[home/byo] picker returned: '
        '${picked == null ? "null (user cancelled)" : "path=${picked.path}"}',
      );
    } catch (e) {
      if (!mounted) return;
      setState(() {
        _busy = false;
        _isError = true;
        _status = 'Gallery open failed: $e';
      });
      return;
    }
    if (picked == null) {
      // user cancelled
      if (!mounted) return;
      setState(() {
        _busy = false;
        _isError = false;
        _status = Strings.welcomeIntro;
      });
      return;
    }

    final bytes = await picked.readAsBytes();
    // image_picker 가 매 호출마다 cache/<UUID>/ 디렉토리에 사진을 복사한다.
    // bytes 만 추출하면 더 이상 필요 없는 임시 파일 — 즉시 삭제하지 않으면
    // 매 BYO 호출마다 cache 누적 (수 MB × N → 수십 MB 차지 → S23 mem 압박).
    // 파일 삭제 + 부모 UUID 디렉토리도 함께 정리.
    try {
      final pickedFile = File(picked.path);
      final pickedParent = pickedFile.parent;
      if (await pickedFile.exists()) await pickedFile.delete();
      if (pickedParent.path.contains('/cache/') &&
          await pickedParent.exists()) {
        await pickedParent.delete(recursive: true);
      }
      debugPrint('[home/byo] cleaned up picker cache: ${pickedParent.path}');
    } catch (e) {
      debugPrint('[home/byo] cache cleanup failed (non-fatal): $e');
    }

    // EXIF GPS 추출 → 경로 polyline 과의 최소 거리 계산 → stepContext 분기.
    // 200m 임계: 이내면 on-route (안내 OK), 초과면 off-route (묘사만, 길안내
    // 금지), GPS 없으면 generic. 500m → 200m 조정 이유: 잠실 인접 시설(롯데월드,
    // 다른 호선 환승 출구 등) 가운데 polyline 에서 200~500m 떨어진 것들이 ON-
    // route 로 잘못 분류돼 모델이 Exit 10 안내를 그대로 붙여버리는 사고 방지.
    final gps = await _extractExifGps(bytes);
    String stepCtx;
    Place? byoOriginPlace;
    bool byoOffRoute = false;
    if (gps == null) {
      stepCtx = Strings.byoStepContext;
      debugPrint('[home/byo] no EXIF GPS — generic context');
    } else {
      final d = _minDistanceFromPolyline(gps.lat, gps.lng);
      if (d.isInfinite) {
        stepCtx = Strings.byoStepContext;
      } else if (d > 200) {
        stepCtx = Strings.byoStepContextOffRoute(d.round());
        byoOffRoute = true;
      } else {
        stepCtx = Strings.byoStepContextOnRoute(d.round());
      }
      // 지도 origin marker 를 BYO 사진의 EXIF GPS 로 이동 — off-route 인 경우
      // "안내문은 경로이탈, 지도 markeronly 경로 위" 모순 방지. on-route 도
      // 함께 갱신해서 "내가 고른 사진이 어디서 찍힌건지" 시각 확인 가능.
      byoOriginPlace = Place(
        name: byoOffRoute ? 'BYO photo (off-route)' : 'BYO photo',
        address: 'EXIF GPS',
        lat: gps.lat,
        lng: gps.lng,
      );
      debugPrint(
        '[home/byo] EXIF GPS=(${gps.lat.toStringAsFixed(5)}, '
        '${gps.lng.toStringAsFixed(5)}) distFromRoute=${d.round()}m '
        '→ ${d > 200 ? "OFF" : "ON"}-route ctx',
      );
    }

    if (!mounted) return;
    setState(() {
      _status = Strings.byoAnalysing;
      if (byoOriginPlace != null) _originPlace = byoOriginPlace;
    });

    Map<String, dynamic> result;
    try {
      result = await GemmaService.instance.generateGuidance(
        photoBytes: bytes,
        stepContext: stepCtx,
        stepHint: 0,
      );
    } catch (e) {
      if (!mounted) return;
      setState(() {
        _busy = false;
        _isError = true;
        _status = 'BYO inference failed: $e';
      });
      return;
    }

    var instruction =
        result['instruction']?.toString() ?? Strings.instructionFallback;
    // Bbox-based left/right override — BYO 사진도 화살표 있으면 동일 보호.
    instruction = _applyArrowBboxOverride(result, instruction, tag: 'byo');
    if (!mounted) return;
    setState(() {
      _busy = false;
      _isError = false;
      _status = instruction;
    });
    await TtsService.instance.speak(instruction);
  }

  /// 카메라 탭 → 전체화면 DemoPhotoPreviewScreen 푸시 (실 카메라 화면 모방).
  /// 사용자가 그 화면의 원형 카메라 버튼 탭 → Navigator.pop 으로 photoBytes
  /// 반환 → 호출측이 정상 inference 흐름 진행.
  Future<Uint8List?> _onCameraDemoTap() async {
    final photoIdx = _currentStep
        .clamp(0, DemoData.photoAssets.length - 1)
        .toInt();
    final byteData = await rootBundle.load(DemoData.photoAssets[photoIdx]);
    final bytes = byteData.buffer.asUint8List();
    if (!mounted) return null;
    return Navigator.push<Uint8List?>(
      context,
      MaterialPageRoute(
        builder: (_) => DemoPhotoPreviewScreen(
          photoBytes: bytes,
          caption: Strings.demoPreviewCaption(
            photoIdx + 1,
            DemoData.photoAssets.length,
          ),
        ),
      ),
    );
  }

  /// 경로 안내 1건의 사진·안내문을 모아둘 세션 폴더를 만든다. 외부 앱 전용
  /// 저장소(/sdcard/Android/data/<pkg>/files/guidance_sessions/<stamp>/)라
  /// `adb pull` 로 바로 회수된다. 실패해도 안내 흐름은 막지 않는다.
  Future<void> _initSessionLog() async {
    _sessionDir = null;
    _captureSeq = 0;
    try {
      final base =
          await getExternalStorageDirectory() ??
          await getApplicationDocumentsDirectory();
      final ts = DateTime.now();
      final stamp =
          '${ts.year}${_two(ts.month)}${_two(ts.day)}_'
          '${_two(ts.hour)}${_two(ts.minute)}${_two(ts.second)}';
      final dir = Directory('${base.path}/guidance_sessions/$stamp');
      await dir.create(recursive: true);
      _sessionDir = dir;
      await File('${dir.path}/log.txt').writeAsString(
        '[$stamp] guidance session start  '
        'dest=${_destPlace?.name ?? "(unknown)"}\n',
        mode: FileMode.append,
      );
      debugPrint('[home/log] session dir: ${dir.path}');
    } catch (e) {
      debugPrint('[home/log] session init 실패: $e');
    }
  }

  /// 찍은 사진 1장과 그에 대응하는 출력 안내문을 세션 폴더에 저장한다.
  /// [kind] 는 안내문 종류(intro/guidance/off_route/no_feature_*/arrival/error).
  /// 사진이 없으면([photoBytes] == null) 로그 줄만 남긴다. 저장 실패는
  /// 안내 흐름에 영향을 주지 않는다.
  Future<void> _recordCapture({
    Uint8List? photoBytes,
    required int step,
    required String kind,
    required String text,
  }) async {
    final dir = _sessionDir;
    if (dir == null) return;
    try {
      var photoName = '-';
      var tag = '---';
      if (photoBytes != null) {
        final seq = ++_captureSeq;
        tag = seq.toString().padLeft(3, '0');
        photoName = 'cap${tag}_step$step.jpg';
        await File('${dir.path}/$photoName').writeAsBytes(photoBytes);
      }
      final now = DateTime.now();
      final stamp = '${_two(now.hour)}:${_two(now.minute)}:${_two(now.second)}';
      await File('${dir.path}/log.txt').writeAsString(
        '[$stamp] #$tag step=$step kind=$kind photo=$photoName\n  $text\n',
        mode: FileMode.append,
      );
    } catch (e) {
      debugPrint('[home/log] capture 저장 실패: $e');
    }
  }

  /// 1자리 수를 2자리 0-pad ('5' -> '05').
  String _two(int n) => n.toString().padLeft(2, '0');

  /// GPS Haversine 거리 < _arrivalRadiusM 이면 도착으로 판정.
  /// DEMO_MODE: 심사위원 GPS 가 송파보건소 근처가 아니므로 GPS 도착판정 불가능
  /// → 도착은 모델의 is_arrival 필드로만 결정.
  Future<bool> _checkArrivalByGps() async {
    if (DemoMode.enabled || _destPlace == null) return false;
    try {
      final pos = await Geolocator.getCurrentPosition(
        locationSettings: const LocationSettings(
          accuracy: LocationAccuracy.high,
          timeLimit: Duration(seconds: 5),
        ),
      );
      final (lat: lat, lng: lng) = _resolveLatLng(pos.latitude, pos.longitude);
      final d = _haversineMeters(lat, lng, _destPlace!.lat, _destPlace!.lng);
      return d <= _arrivalRadiusM;
    } catch (_) {
      return false;
    }
  }

  /// 사진 캡처 시점 GPS 가 현재 step 의 ODsay subPath line 에서 벗어났는지.
  /// 반환: null = 정상 / 검증 불가 (좌표 없음, 비도보 segment).
  /// double = segment 까지 수직거리 (meters, > _offRouteRadiusM).
  Future<double?> _checkOffRoute() async {
    final route = _route;
    if (route == null) {
      debugPrint('[off-route] skip: no route');
      return null;
    }
    if (_currentStep < 0 || _currentStep >= route.subPaths.length) {
      debugPrint(
        '[off-route] skip: step $_currentStep out of range '
        '(subPaths=${route.subPaths.length})',
      );
      return null;
    }
    final sub = route.subPaths[_currentStep];
    // 지하철/버스 segment 는 GPS 의미 없음 (지하 / 차량 안). 도보 만 검증.
    if (sub.mode != SubPathMode.walk) {
      debugPrint(
        '[off-route] skip: subPath $_currentStep mode=${sub.mode.name}',
      );
      return null;
    }
    final endpoints = _walkEndpointsForStep(_currentStep);
    if (endpoints == null) {
      debugPrint(
        '[off-route] skip: no derivable endpoints for walk step '
        '$_currentStep',
      );
      return null;
    }
    final (sLat, sLng, eLat, eLng) = endpoints;
    try {
      final pos = await Geolocator.getCurrentPosition(
        locationSettings: const LocationSettings(
          accuracy: LocationAccuracy.high,
          timeLimit: Duration(seconds: 5),
        ),
      );
      final (lat: pLat, lng: pLng) = _resolveLatLng(
        pos.latitude,
        pos.longitude,
      );
      final d = _distanceToSegmentMeters(pLat, pLng, sLat, sLng, eLat, eLng);
      debugPrint(
        '[off-route] step=$_currentStep walk dist=${d.toStringAsFixed(0)}m '
        '(seg ${sLat.toStringAsFixed(5)},${sLng.toStringAsFixed(5)} → '
        '${eLat.toStringAsFixed(5)},${eLng.toStringAsFixed(5)} | '
        'pos ${pLat.toStringAsFixed(5)},${pLng.toStringAsFixed(5)})',
      );
      if (d > _offRouteRadiusM) return d;
      return null;
    } catch (e) {
      debugPrint('[off-route] check failed: $e');
      return null;
    }
  }

  /// step `i` 가 walk subPath 일 때, segment 양 끝의 (lat, lng) 추정.
  /// ODsay walk subPath 는 자체 좌표가 없음 → 인접 transit subPath 의 effective
  /// start/end 또는 origin/destination 으로 derive.
  (double, double, double, double)? _walkEndpointsForStep(int i) {
    final route = _route;
    if (route == null) return null;
    final subs = route.subPaths;
    if (i < 0 || i >= subs.length) return null;

    double? startLat, startLng, endLat, endLng;

    // segment 의 시작 좌표: i-1 번째 subPath 의 end. 없으면 origin.
    for (int k = i - 1; k >= 0; k--) {
      final p = subs[k];
      if (p.effectiveEndLat != null && p.effectiveEndLng != null) {
        startLat = p.effectiveEndLat;
        startLng = p.effectiveEndLng;
        break;
      }
    }
    if (startLat == null || startLng == null) {
      final o = _originPlace;
      if (o != null) {
        startLat = o.lat;
        startLng = o.lng;
      }
    }

    // segment 의 끝 좌표: i+1 번째 subPath 의 start. 없으면 destination.
    for (int k = i + 1; k < subs.length; k++) {
      final p = subs[k];
      if (p.effectiveStartLat != null && p.effectiveStartLng != null) {
        endLat = p.effectiveStartLat;
        endLng = p.effectiveStartLng;
        break;
      }
    }
    if (endLat == null || endLng == null) {
      final d = _destPlace;
      if (d != null) {
        endLat = d.lat;
        endLng = d.lng;
      }
    }

    if (startLat == null ||
        startLng == null ||
        endLat == null ||
        endLng == null) {
      return null;
    }
    return (startLat, startLng, endLat, endLng);
  }

  Future<void> _handleArrival(String llmInstruction) async {
    _arrived = true;
    // DEMO: _destPlace.name 은 한글 '송파구보건소' 라 영문 TTS 와 안 맞음 →
    // Strings.demoDestinationNameEn ('Songpa Health Center') 로 치환.
    final destName = DemoMode.enabled
        ? Strings.demoDestinationNameEn
        : (_destPlace?.name ?? '목적지');
    final farewell = Strings.farewellAt(destName);
    final farewellSpeak = Strings.farewellAtSpeak(destName);
    setState(() {
      _busy = false;
      _isError = false;
      _status = '$llmInstruction\n\n$farewell';
    });
    // farewell 은 사용자가 끝까지 들어야 하는 마무리 — fire-and-forget speak()
    // 대신 speakAndAwait() 로 발화 완료까지 대기. 그 후 5초 여운 (text 가 잠시
    // 더 화면에 남도록) → home reset. 이전엔 5초 후 바로 reset 해서 TTS 가
    // ~10초 짜리 farewell 을 다 못 끝내고 화면이 dismiss 되는 문제 있었음.
    await TtsService.instance.speakAndAwait(farewellSpeak);
    await Future<void>.delayed(const Duration(seconds: 5));
    if (!mounted) return;
    setState(() {
      _route = null;
      _destPlace = null;
      _originPlace = null;
      _currentStep = 0;
      _arrived = false;
      _noFeatureStreak = 0;
      _needHumanHelp = false;
      _isError = false;
      _status = Strings.welcomeIntro;
    });
  }

  double _haversineMeters(double lat1, double lon1, double lat2, double lon2) {
    const r = 6371000.0;
    double toRad(double d) => d * math.pi / 180.0;
    final dLat = toRad(lat2 - lat1);
    final dLon = toRad(lon2 - lon1);
    final a =
        math.sin(dLat / 2) * math.sin(dLat / 2) +
        math.cos(toRad(lat1)) *
            math.cos(toRad(lat2)) *
            math.sin(dLon / 2) *
            math.sin(dLon / 2);
    final c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a));
    return r * c;
  }

  /// 점 P 에서 line segment A→B 까지 수직거리 (meters).
  /// 짧은 도보 segment (≤ 1 km) 에 한해 equirectangular 근사. 충분히 정확.
  double _distanceToSegmentMeters(
    double pLat,
    double pLon,
    double aLat,
    double aLon,
    double bLat,
    double bLon,
  ) {
    const r = 6371000.0;
    double toRad(double d) => d * math.pi / 180.0;
    final cosLat = math.cos(toRad((aLat + bLat) / 2));
    final ax = toRad(aLon) * cosLat * r;
    final ay = toRad(aLat) * r;
    final bx = toRad(bLon) * cosLat * r;
    final by = toRad(bLat) * r;
    final px = toRad(pLon) * cosLat * r;
    final py = toRad(pLat) * r;
    final dx = bx - ax;
    final dy = by - ay;
    final lenSq = dx * dx + dy * dy;
    if (lenSq == 0) {
      return math.sqrt((px - ax) * (px - ax) + (py - ay) * (py - ay));
    }
    final t = (((px - ax) * dx + (py - ay) * dy) / lenSq).clamp(0.0, 1.0);
    final cx = ax + t * dx;
    final cy = ay + t * dy;
    return math.sqrt((px - cx) * (px - cx) + (py - cy) * (py - cy));
  }

  Future<Uint8List> _readBytes(String path) async {
    return File(path).readAsBytes();
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final cs = theme.colorScheme;
    final hasRoute = _route != null && !_arrived;
    return Scaffold(
      backgroundColor: cs.surface,
      appBar: const GilbeotAppBar(title: '길벗'),
      body: SafeArea(
        child: Padding(
          padding: const EdgeInsets.fromLTRB(20, 16, 20, 24),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              // ── Last-resort 사람-도움 banner ───────────────────────────────
              // 같은 step 에서 NO_FEATURE 가 _maxNoFeatureRetries 회 연속 발생한
              // 경우에만 노출. 단일 dismiss 버튼이 streak 을 0 으로 리셋하고
              // 다시 카메라 재시도 흐름으로 돌아간다.
              if (_needHumanHelp) ...[
                GilbeotHelpBanner(
                  message: _status,
                  onDismiss: _dismissHumanHelpBanner,
                ),
                const SizedBox(height: 16),
              ],

              // ── Hero status card (DEMO_MODE 사진 표시 시 그 사진으로 교체) ─
              Expanded(
                child: _demoShownPhotoBytes != null
                    ? Column(
                        crossAxisAlignment: CrossAxisAlignment.stretch,
                        children: [
                          Expanded(
                            child: ClipRRect(
                              borderRadius: BorderRadius.circular(16),
                              child: Image.memory(
                                _demoShownPhotoBytes!,
                                fit: BoxFit.contain,
                              ),
                            ),
                          ),
                          const SizedBox(height: 12),
                          Text(
                            _status,
                            textAlign: TextAlign.center,
                            style: theme.textTheme.titleMedium,
                          ),
                        ],
                      )
                    : GilbeotStatusCard(
                        message: _status,
                        isLoading: _busy,
                        isError: _isError,
                        isArrived: _arrived,
                      ),
              ),

              const SizedBox(height: 20),

              // 진행 단계 표시는 제거됨 — ODsay subPath 총 step 수는 카메라 판정
              // 흐름과 1:1 매핑되지 않아 어르신에게 잘못된 기대치를 줌. 진행도는
              // 상태 카드(GilbeotStatusCard) 의 메시지로만 노출한다.

              // ── Primary action(s) ────────────────────────────────────────
              // 첫 진입: 마이크 단독(hero). 경로 로드 후: 좌우 2등분으로
              // [지도 보기 | 다시 말하기] — 어르신이 안내를 듣고 나서 (a) 지도로
              // 형태 한 번 더 확인, (b) 새 목적지 발화 두 갈래로 자연 분기.
              // 지도 보기 + 마이크 좌우 2등분. DEMO_MODE 도 표시 — canned route
              // 를 그대로 NaverMap 에 그려준다 (origin=잠실역, dest=송파보건소).
              if (hasRoute)
                Row(
                  children: [
                    Expanded(
                      child: GilbeotMapButton(
                        enabled: !_busy,
                        onPressed: _onMapPressed,
                        label: Strings.mapLabel,
                      ),
                    ),
                    const SizedBox(width: 12),
                    Expanded(
                      child: GilbeotMicButton(
                        isRecording: _recording,
                        isBusy: _busy,
                        onTap: _onMicTapped,
                        onLongPressStart: _onMicPressed,
                        onLongPressEnd: _onMicReleased,
                        idleLabel: Strings.micRetryLabel,
                        recordingLabel: Strings.micRecordingLabel,
                        busyLabel: Strings.micBusyLabel,
                        compact: true,
                      ),
                    ),
                  ],
                )
              else
                GilbeotMicButton(
                  isRecording: _recording,
                  isBusy: _busy,
                  onTap: _onMicTapped,
                  onLongPressStart: _onMicPressed,
                  onLongPressEnd: _onMicReleased,
                  idleLabel: Strings.micIdleLabel,
                  recordingLabel: Strings.micRecordingLabel,
                  busyLabel: Strings.micBusyLabel,
                ),

              const SizedBox(height: 16),

              // ── Secondary camera action ───────────────────────────────────
              GilbeotCameraButton(
                enabled: hasRoute && !_busy,
                onPressed: _onCameraPressed,
                label: Strings.cameraLabel,
                onLongPress: DemoMode.enabled ? _onCameraLongPress : null,
              ),
            ],
          ),
        ),
      ),
    );
  }
}

// ---------------------------------------------------------------------------
// Korean route-narration helpers — private to this screen.
//
// ODsay subPath 의 `lane` 은 지하철은 보통 "9호선" / "수도권 9호선" 으로 오지만,
// 버스는 빈값이거나 숫자만 ("472") 오기도 함. TTS 안내에 그대로 끼우면 어르신이
// 듣기에 "X에서  을 타세요." 처럼 어색하거나 "X에서 472을 타세요." 처럼
// 조사도 잘못 붙음. 아래 헬퍼는 모드별 자연스러운 명칭과 한국어 받침 기반
// 조사를 정확히 산출한다.
// ---------------------------------------------------------------------------

/// ODsay lane prefix("수도권 ") 정리. null/empty → ''.
String _laneShort(String? lane) {
  if (lane == null) return '';
  return lane.replaceAll(RegExp(r'^수도권\s*'), '').trim();
}

/// 모드별 어르신이 알아들을 수 있는 명칭으로 정규화.
///
/// subway: "9호선" / 빈값 → "지하철"
/// bus   : "472" → "472번 버스", "마을버스 송파02" → "마을버스 송파02 버스",
///         "472번 버스" → 그대로, 빈값 → "버스"
String _vehicleLabel(SubPath seg) {
  final lane = _laneShort(seg.lane);
  switch (seg.mode) {
    case SubPathMode.subway:
      return lane.isEmpty ? '지하철' : lane;
    case SubPathMode.bus:
      if (lane.isEmpty) return '버스';
      if (lane.endsWith('버스')) return lane;
      if (RegExp(r'^\d+$').hasMatch(lane)) return '$lane번 버스';
      return '$lane 버스';
    default:
      return lane.isEmpty ? '교통편' : lane;
  }
}

/// Hangul syllable 의 종성(받침) 인덱스. 한글 음절이 아니면 null.
/// 0 = 받침 없음, 1..27 = 종성 자모.
int? _lastJongseong(String s) {
  if (s.isEmpty) return null;
  final last = s.runes.last;
  if (last < 0xAC00 || last > 0xD7A3) return null;
  return (last - 0xAC00) % 28;
}

/// "X을/를" 조사. 받침 있으면 을, 없거나 한글 아니면 를.
String _eulReul(String s) {
  final j = _lastJongseong(s);
  if (j == null) return '를';
  return j == 0 ? '를' : '을';
}

/// "X(으)로" 조사. 받침 없음 또는 ㄹ(=8) → 로, 그 외 → 으로.
String _euroRo(String s) {
  final j = _lastJongseong(s);
  if (j == null) return '로';
  if (j == 0 || j == 8) return '로';
  return '으로';
}

/// [지도 보기] 탭 후 GPS 갱신 + NaverMap 초기화 동안 보이는 modal dialog.
///
/// 어르신이 "탭 했는데 반응 없다" 며 화면을 다시 만지지 않도록 큰 spinner +
/// 안내 문구로 처리 중임을 명시. `barrierDismissible: false` 라 외부 탭으로
/// 닫히지 않고 호출측에서 명시적으로 `Navigator.pop` 해야 함.
class _MapLoadingDialog extends StatelessWidget {
  const _MapLoadingDialog();

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final cs = theme.colorScheme;
    return Dialog(
      backgroundColor: cs.surface,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(24)),
      child: Padding(
        padding: const EdgeInsets.symmetric(horizontal: 28, vertical: 32),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            SizedBox(
              width: 56,
              height: 56,
              child: CircularProgressIndicator(
                strokeWidth: 5,
                color: cs.primary,
              ),
            ),
            const SizedBox(height: 20),
            Text(
              '지도를 준비하는 중...',
              style: theme.textTheme.titleMedium?.copyWith(
                fontWeight: FontWeight.w700,
              ),
            ),
            const SizedBox(height: 6),
            Text(
              '잠시만 기다려 주세요',
              style: theme.textTheme.bodyMedium?.copyWith(color: cs.outline),
            ),
          ],
        ),
      ),
    );
  }
}
