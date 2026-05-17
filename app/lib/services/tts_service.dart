import 'package:flutter/foundation.dart';
import 'package:flutter_tts/flutter_tts.dart';

import '../config/demo_mode.dart';

/// TTS 출력 (시스템 엔진 사용 — Android: Google TTS, 무료).
/// 어르신 가독성 — 약간 느린 속도 + 안정적 톤.
///
/// 기본 한국어. `DemoMode.enabled` 면 영문(en-US) — 심사위원 데모용.
class TtsService {
  TtsService._();
  static final TtsService instance = TtsService._();

  final FlutterTts _tts = FlutterTts();
  bool _ready = false;

  Future<void> initialize() async {
    if (_ready) return;
    final lang = DemoMode.enabled ? 'en-US' : 'ko-KR';
    // S23 (Samsung TTS 기본) + DEMO_MODE(en-US) 처리 노트:
    //
    // 시도 1: setEngine('com.google.android.tts') 강제 호출 — 실패. 두번째
    // TextToSpeech 인스턴스가 생성되며 첫 인스턴스와 race condition 으로
    // process death/unbind cycle 발생 (logcat: Unbinding TTS engine, client
    // process death). 결과적으로 speak 가 dead instance 로 가서 완전 무음.
    //
    // 시도 2 (현재): setEngine 제거. Android system 이 이미 private engine
    // (Samsung SMT) 거부 후 getHighestRankedPublicEngineName 으로 Google TTS
    // 자동 fallback 함 (logcat 첫 binding 시점에 자동). flutter_tts 의 단일
    // 인스턴스가 자연스럽게 Google TTS bound — 충돌 없음.
    //
    // 만약 isLanguageAvailable 가 false 면 (Google TTS 미설치 + Samsung 도
    // en-US 없음) 사용자에게 OS TTS 설정 안내 로그.
    final setLangResult = await _tts.setLanguage(lang);
    debugPrint('[tts] setLanguage($lang) returned $setLangResult');
    try {
      final available = await _tts.isLanguageAvailable(lang);
      debugPrint('[tts] isLanguageAvailable($lang)=$available');
    } catch (e) {
      debugPrint('[tts] isLanguageAvailable check failed: $e');
    }
    // setLanguage 가 silently fail 하면 default voice (보통 ko-KR) 이 영어
    // 텍스트를 한국어 발음으로 읽는다 (S23 Google TTS 에서 관측). 명시
    // setVoice 로 강제 — voice list 에서 lang prefix 매칭 첫 voice 사용.
    if (DemoMode.enabled) {
      try {
        final voices = await _tts.getVoices;
        if (voices is List) {
          final targetPrefix =
              lang.split('-').first.toLowerCase(); // "en" or "ko"
          final matching = voices
              .whereType<Map>()
              .where((v) {
            final loc = v['locale']?.toString().toLowerCase() ?? '';
            return loc.startsWith(targetPrefix);
          }).toList();
          if (matching.isNotEmpty) {
            final v = matching.first;
            await _tts.setVoice(
                {'name': v['name'].toString(), 'locale': v['locale'].toString()});
            debugPrint('[tts] forced setVoice → ${v['name']} (${v['locale']})');
          } else {
            debugPrint('[tts] no $targetPrefix voice in getVoices list '
                '(count=${voices.length})');
          }
        }
      } catch (e) {
        debugPrint('[tts] setVoice fallback failed: $e');
      }
    }
    // 0.42 는 한국어 어르신 페이스. 영어 음성도 같은 슬로우 톤이 accessibility
    // 시연 의도에 맞아 그대로 사용.
    await _tts.setSpeechRate(0.42);
    await _tts.setPitch(1.0);
    await _tts.setVolume(1.0);
    // awaitSpeakCompletion(false) — speak() 가 음성 완료까지 기다리지 않고
    // 즉시 return. 이전 (true) 동작은 home_screen 의 `await speak()` 가 음성
    // 끝날 때까지 풀리지 않아 `_busy` 가 8~20초 잡혀 음성 도중 다른 버튼이
    // 무시됐다. 어르신/심사위원이 "다음 사진" 또는 "다시 말하기" 를 음성
    // 도중 누르는 게 자연스러운 UX 라 인터럽트 가능해야 한다. 새 동작이
    // 호출하는 `TtsService.stop()` 이 현재 발화를 끊는다. 또 fire-and-forget
    // 이라 S23 Samsung TTS 콜백 미수신 hang 위험도 자동 해소.
    await _tts.awaitSpeakCompletion(false);
    debugPrint('[tts] initialized lang=$lang demoMode=${DemoMode.enabled}');
    _ready = true;
  }

  /// 텍스트를 음성으로 출력. `awaitSpeakCompletion(false)` 라 즉시 return —
  /// 음성은 백그라운드에서 계속 재생된다. 새 발화나 [stop] 으로 인터럽트
  /// 가능. 같은 텍스트로 빠르게 두 번 호출되면 첫 발화가 중단되고 두 번째
  /// 발화만 들린다 (`stop()` → `speak()` 시퀀스).
  ///
  /// **S23 Google TTS lifecycle 대응**: init 후 ~30s idle 동안 시스템이
  /// TTS service binder 를 회수 → 다음 speak 시 `DeadObjectException`. 이를
  /// 감지하면 light re-bind (setLanguage 만 — getVoices/setVoice 는 무거워
  /// S23 메모리 압박에서 OOM 트리거) 후 1회 재시도. S10e Samsung TTS 는
  /// idle 후에도 connection 유지해 이 경로 안 탐.
  Future<void> speak(String text) async {
    if (!_ready) await initialize();
    final ok = await _trySpeak(text);
    if (ok) return;
    // 첫 시도 실패 — light 재바인딩 (setLanguage 만, getVoices/setVoice
    // 호출 안 함 — full init 은 메모리 비용 크고 retry path 에선 voice 가
    // 이미 첫 init 에서 강제 setVoice 됐으므로 다시 안 해도 OK).
    debugPrint('[tts] first speak failed, light re-binding (setLanguage only)');
    try {
      final lang = DemoMode.enabled ? 'en-US' : 'ko-KR';
      await _tts.setLanguage(lang).timeout(const Duration(seconds: 3),
          onTimeout: () => 0);
    } catch (e) {
      debugPrint('[tts] light re-bind setLanguage failed: $e');
    }
    final ok2 = await _trySpeak(text);
    if (!ok2) {
      debugPrint('[tts] speak failed on both attempts — giving up silently');
    }
  }

  /// 한 번의 speak 시도 — stop+speak 시퀀스. 결과: int 1 = success
  /// (queued), 그 외 = fail. flutter_tts 가 throw 하면 false.
  Future<bool> _trySpeak(String text) async {
    try {
      await _tts.stop().timeout(
            const Duration(seconds: 3),
            onTimeout: () {},
          );
      final result = await _tts
          .speak(text)
          .timeout(const Duration(seconds: 2), onTimeout: () => 0);
      return result == 1;
    } catch (e) {
      debugPrint('[tts] _trySpeak threw: $e');
      return false;
    }
  }

  Future<void> stop() async {
    try {
      await _tts.stop().timeout(const Duration(seconds: 3), onTimeout: () {});
    } catch (e) {
      debugPrint('[tts] stop failed/timed out: $e');
    }
  }

  /// 발화 완료까지 await — 사용자가 끝까지 듣고 화면이 자연스럽게 종료되어야
  /// 하는 케이스 (`_handleArrival` 의 farewell) 에만 사용. 일반 안내는 [speak]
  /// 사용 (fire-and-forget 으로 interrupt 가능).
  ///
  /// 동작: `awaitSpeakCompletion(true)` 일회용 활성화 → speak → 결과/예외 무관
  /// finally 에서 `(false)` 복원. 이후 일반 [speak] 은 다시 fire-and-forget.
  /// timeout 20s (farewell 2-3 문장 안전 상한).
  Future<void> speakAndAwait(String text) async {
    if (!_ready) await initialize();
    try {
      await _tts.awaitSpeakCompletion(true);
      await _tts.stop().timeout(
            const Duration(seconds: 3),
            onTimeout: () {},
          );
      await _tts
          .speak(text)
          .timeout(const Duration(seconds: 20), onTimeout: () => 0);
    } catch (e) {
      debugPrint('[tts] speakAndAwait threw: $e');
    } finally {
      try {
        await _tts.awaitSpeakCompletion(false);
      } catch (_) {}
    }
  }
}
