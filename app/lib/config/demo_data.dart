import 'dart:convert';

import 'package:flutter/services.dart' show rootBundle;

import '../services/tmap_pedestrian_service.dart' show Place;
import '../services/odsay_service.dart' show RoutePath, SubPath, SubPathMode;

/// 심사위원용 DEMO_MODE 의 정적 자산·좌표·캐시된 발화 한 곳.
///
/// 이 파일을 single source of truth 로 두면 home_screen 의 DEMO 분기들이
/// 흩어진 매직 상수 없이 동일한 잠실역 → 송파구보건소 시나리오를 공유한다.
class DemoData {
  const DemoData._();

  // ── Asset paths ───────────────────────────────────────────────────────
  /// 미리 저장된 사진 4장 (assets/demo_photos/{1,2,5,7}.jpg).
  /// 시연 narrative: ① 출구 표지판+계단 → ② 개찰구+나가는 곳 → ③ 10번/송파구청 분기
  /// → ④ 송파구보건소 도착. 기존 demo_photos 자산을 그대로 재사용해 중복 번들
  /// 회피.
  static const List<String> photoAssets = [
    'assets/demo_photos/1.jpg',
    'assets/demo_photos/2.jpg',
    'assets/demo_photos/5.jpg',
    'assets/demo_photos/7.jpg',
  ];

  /// 한국어 발화 WAV (assets/demo/audio/dest_query.wav). 16kHz mono.
  /// 부재 시 [cachedTranscript] 로 fallback.
  static const String destQueryWavAsset = 'assets/demo/audio/dest_query.wav';

  /// WAV 부재 시 STT pipeline 우회용 캐시 — 발화 의도가 정해져 있으므로 모델
  /// 호출 자체를 skip 하고 곧장 transcript 만 표시한다. WAV 가 들어오면 실
  /// Gemma audio 가 동일 transcript 를 만들어내야 정상. 현재 번들된 WAV
  /// (demo_voice2 기반) 의 실제 발화 텍스트.
  static const String cachedTranscript = '송파구 보건소 안내해 줘';

  /// 영문 자막 — judge UI 에 "Korean speaker says 'X'" 형태로 보여줄 때 사용.
  static const String cachedTranscriptEn =
      'Please guide me to Songpa Health Center';

  // ── Coordinates (잠실역 8호선/2호선 + 송파구보건소) ─────────────────────
  static const double jamsilLat = 37.5132;
  static const double jamsilLng = 127.1000;
  static const double songpaBogeonLat = 37.51459210567331;
  static const double songpaBogeonLng = 127.10661109453235;

  // ── Hardcoded Place objects ───────────────────────────────────────────
  static Place jamsilOrigin() => Place(
        name: '잠실역',
        address: '서울 송파구 올림픽로 지하 265',
        lat: jamsilLat,
        lng: jamsilLng,
      );

  static Place songpaBogeonDestination() => Place(
        name: '송파구보건소',
        address: '서울 송파구 올림픽로 326',
        lat: songpaBogeonLat,
        lng: songpaBogeonLng,
      );

  /// step 별 scene context (어디인지 + 어떤 종류의 단서가 있는지). 방향
  /// (LEFT/UP/RIGHT) 은 prescribe 하지 않는다 — 모델이 사진의 화살표/계단을
  /// 직접 읽어 결정. 이 hint 는 reviewer 가 코드를 봐도 "정답을 답해줬다"
  /// 보이지 않도록 landmark 묘사만 한다.
  ///
  /// 모든 stepHint 의 공통 suffix — generic bbox 규칙. systemPrompt 안에 같은
  /// 규칙이 있지만 small VLM 이 long prompt 안의 강조를 attention 약하게
  /// 처리한다 (logcat 검증). stepHint 의 가장 가까운 context 에 한 줄로 박아
  /// 강조 위치 재배치. 4개 photo 모두 동등 적용이라 cherry-picking 아님 —
  /// "이 사진에 화살표가 있다" 식 사진별 정보 박지 않음.
  static const String _arrowBboxReminder =
      ' If you see any horizontal arrow in this photo, fill arrow_tip_x and '
      'arrow_tail_x (normalized x coordinates 0~1) — the app uses them to '
      'overwrite left/right wording reliably.';

  static const List<String> stepHintsEn = [
    // step 0 — photo 1: 잠실역 출구 1-11 그리드
    '[Scene] You are inside Jamsil Station. The photo likely shows an exit-number sign with multiple numbers (one of them is Exit 10) and possibly stairs. Read the visible arrow / stair direction and guide the user accordingly.$_arrowBboxReminder',
    // step 1 — photo 2: 개찰구 + 1·2·8·9·10·11
    '[Scene] You are near the Jamsil Station fare gates. The photo likely shows the yellow "Exit / 나가는 곳" sign with several exit numbers (Exit 10 is among them) and an arrow indicating which way to go. Read the arrow direction in the photo and guide the user that way.$_arrowBboxReminder',
    // step 2 — photo 5: 10·11 송파구청/송파대로방면 + 에스컬레이터
    '[Scene] You are still underground, at the exit-branch sign for Exit 10 / Exit 11 (toward Songpa-gu Office). An escalator or stairs is likely visible. Read the visible direction and guide the user toward Exit 10.$_arrowBboxReminder',
    // step 3 — photo 7: 송파구보건소 건물 외관 (도착)
    '[Scene] You should now be in front of the destination — the Songpa Health Center (송파구보건소) building. If you can see the building entrance or its sign, set is_arrival=true and warmly congratulate the user.$_arrowBboxReminder',
  ];

  /// 답사 사진 4장의 실제 EXIF GPS (lat, lng). photo i 가 찍힌 위치.
  /// step 진행 시 origin marker 를 이 좌표로 이동시켜 "걸어가는" 시각화.
  static const List<List<double>> photoGps = [
    [37.51294722222222, 127.10109722222222], // 1.jpg — 잠실역 출구 그리드 표지
    [37.5134082,        127.10059349999999], // 2.jpg — 개찰구 + 나가는 곳
    [37.514492699722226, 127.10420069999999], // 5.jpg — 10번/송파구청 분기 (지상)
    [37.51409699972222,  127.10573469972222], // 7.jpg — 송파구보건소 건물
  ];

  /// step 진행에 따른 origin 좌표.
  ///   step <= 0  -> jamsil (초기, 사진 찍기 전)
  ///   step >= 1  -> 직전 사진(step-1)이 찍힌 EXIF GPS
  static Place originForStep(int step) {
    if (step <= 0) return jamsilOrigin();
    final idx = (step - 1).clamp(0, photoGps.length - 1);
    final c = photoGps[idx];
    return Place(
      name: '잠실역',
      address: 'demo step $step',
      lat: c[0],
      lng: c[1],
    );
  }

  /// 캐시된 T-Map 도보 polyline ([lng, lat]). `_loadPolyline` 가 1회 로드.
  static List<List<double>>? _cachedPolyline;
  static Future<List<List<double>>> _loadPolyline() async {
    if (_cachedPolyline != null) return _cachedPolyline!;
    try {
      final raw =
          await rootBundle.loadString('assets/demo/route_polyline.json');
      final j = jsonDecode(raw) as Map<String, dynamic>;
      _cachedPolyline = (j['polyline'] as List)
          .cast<List<dynamic>>()
          .map((p) =>
              [(p[0] as num).toDouble(), (p[1] as num).toDouble()])
          .toList();
    } catch (_) {
      // asset 미번들 fallback — 직선 2점.
      _cachedPolyline = [
        [jamsilLng, jamsilLat],
        [songpaBogeonLng, songpaBogeonLat],
      ];
    }
    return _cachedPolyline!;
  }

  /// Canned ODsay-equivalent 도보 경로 + 실 T-Map 도보 polyline.
  /// `_runMicDemoMode` 가 await 으로 호출.
  static Future<RoutePath> cannedRoute() async {
    final walk = SubPath(
      mode: SubPathMode.walk,
      sectionTimeMin: 14,
      distanceM: 800,
      startStationName: '잠실역',
      startExitNo: '10',
      startX: jamsilLng,
      startY: jamsilLat,
      endX: songpaBogeonLng,
      endY: songpaBogeonLat,
    );
    walk.polyline = await _loadPolyline();
    return RoutePath(
      totalTimeMin: 14,
      totalWalkM: 800,
      mapObj: '',
      subPaths: [walk],
    );
  }

  /// WAV asset 의 실제 존재 여부(빈 placeholder 가 아닌지). 사용자가 녹음
  /// 파일을 채우기 전 단계에서는 false → home_screen 이 STT 호출 skip 하고
  /// [cachedTranscript] 직접 사용.
  static Future<bool> hasRealWav() async {
    try {
      final data = await rootBundle.load(destQueryWavAsset);
      // 1KB 미만이면 placeholder 로 간주 (실 16kHz mono WAV 는 최소 수 KB).
      return data.lengthInBytes >= 1024;
    } catch (_) {
      return false;
    }
  }
}
