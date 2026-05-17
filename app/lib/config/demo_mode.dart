import 'package:flutter_dotenv/flutter_dotenv.dart';

/// 심사위원용 데모 모드 게이트.
///
/// `assets/env_config`의 `DEMO_MODE=true`로 활성화. 이 모드에서는:
/// - UI/안내문/TTS 모두 영문 (`Strings` 헬퍼가 영문 분기)
/// - 음성 입력: 미리 저장된 한국어 WAV 재생 (judge 가청) + 실 Gemma audio STT
/// - 목적지: 송파구보건소 hardcode (parseDestination 우회)
/// - 출발지: 잠실역 hardcode (Geolocator GPS 우회)
/// - 라우팅: canned `assets/demo/route.json` (ODsay/T-Map skip — 한국 영토 의존성 제거)
/// - 카메라: 첫 탭 미리 저장된 사진 표시 / 두 번째 탭 실 Gemma vision (English systemPrompt)
///
/// `scripts/build_install_demo_apk.py`가 빌드 시 이 플래그를 true로 패치하고
/// 패키지 ID를 `com.psymon.gilbeot.demo` / 라벨 "Gilbeot Demo"로 분리해 한국
/// production 빌드 (`com.psymon.gilbeot` / "길벗")와 같은 디바이스에 공존.
///
/// 기본값 false — 옵트인만 켜진다 (production 안전).
class DemoMode {
  const DemoMode._();

  static bool get enabled {
    final v = dotenv.maybeGet('DEMO_MODE')?.toLowerCase().trim();
    return v == 'true' || v == '1';
  }
}
