import 'dart:math' as math;

import 'package:dio/dio.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter_dotenv/flutter_dotenv.dart';

import 'tmap_pedestrian_service.dart';

/// ODsay 대중교통 + 도보 라우팅 — 한국에서 지하철 출구 번호(`startExitNo`,
/// `endExitNo`)를 무료로 반환하는 사실상 유일한 공개 REST.
///
/// 라이트업/UX에서는 브랜드 노출 안 함 — "지도 라우팅 API"로 표현.
class OdsayService {
  OdsayService._();
  static final OdsayService instance = OdsayService._();

  final Dio _dio = Dio(
    BaseOptions(
      baseUrl: 'https://api.odsay.com/v1/api',
      connectTimeout: const Duration(seconds: 10),
      receiveTimeout: const Duration(seconds: 15),
    ),
  );

  String get _key => dotenv.maybeGet('ODSAY_ANDROID_KEY')?.trim() ?? '';

  /// 출발지(sx, sy) → 목적지(ex, ey) 대중교통 + 도보 통합 경로.
  /// 좌표는 WGS84 (lng=x, lat=y). 700m 미만이면 ODsay가 거부 → 직선거리 fallback.
  Future<RouteResult> searchTransit({
    required double sx,
    required double sy,
    required double ex,
    required double ey,
  }) async {
    final r = await _dio.get(
      '/searchPubTransPathT',
      queryParameters: {'apiKey': _key, 'SX': sx, 'SY': sy, 'EX': ex, 'EY': ey},
    );
    final body = r.data as Map<String, dynamic>;
    // ODsay 가 700m 미만(같은 정류장권) 또는 라우팅 불가 시 error 반환. 어르신
    // 이 "지금 가까운 곳" 으로 가는 경우거나 데모 시연 시 흐름 끊기는 사고를
    // 막기 위해 도보 only fallback 라우트 자동 생성 — T-Map 보행자로 polyline
    // 받아 단일 walk SubPath 로 구성. 흐름은 정상 success 분기 그대로.
    if (body.containsKey('error')) {
      return RouteResult.success(await _walkOnlyRoute(sx, sy, ex, ey));
    }
    final paths = (body['result']?['path'] as List?) ?? const [];
    if (paths.isEmpty) {
      return RouteResult.success(await _walkOnlyRoute(sx, sy, ex, ey));
    }
    final route = RoutePath.fromJson(paths.first as Map<String, dynamic>);
    // ODsay loadLane → graphPos (정류장 + 일부 turn point) 로 1차 polyline
    // 채움. 이건 점이 듬성해 직선 이으면 건물 가로지름. 그래서 다음 단계로
    // NCP Directions 호출해서 *실제 도로* polyline 으로 덮어씀. 둘 다 실패해도
    // 직선 fallback 으로 동작하니 fire-and-fail-soft.
    try {
      await _loadLanePolyline(route);
    } catch (_) {}
    try {
      await _enrichWithDrivingPath(
        route,
        originLng: sx,
        originLat: sy,
        destLng: ex,
        destLat: ey,
      );
    } catch (_) {}
    return RouteResult.success(route);
  }

  /// 각 subPath (transit + walk) 를 NCP Directions 5 driving 경로로 채움.
  ///
  /// Transit (버스/지하철): start→end 정류장 사이 도로 polyline.
  /// Walk: ODsay 가 walk 의 start/end 좌표를 안 줘서 인접 subPath / 라우트
  /// 전체의 origin/dest 좌표로 보완:
  ///   - 첫 walk (전 transit 없음)         : origin → 다음 transit.start
  ///   - 마지막 walk (다음 transit 없음)   : 이전 transit.end → dest
  ///   - 중간 walk (환승 도보)             : 이전 transit.end → 다음 transit.start
  ///
  /// 자동차 경로라 도보 우회 (지하상가/공원 도보길) 와 일부 차이가 있지만,
  /// 도로 따라가는 모양으로 어르신이 위치 인지하기엔 충분.
  Future<void> _enrichWithDrivingPath(
    RoutePath route, {
    required double originLng,
    required double originLat,
    required double destLng,
    required double destLat,
  }) async {
    debugPrint(
      '[GILBEOT_directions] enrich start — '
      'subPathModes=${route.subPaths.map((s) => s.mode.name).toList()}',
    );
    final futures = <Future<void>>[];
    for (var i = 0; i < route.subPaths.length; i++) {
      final sp = route.subPaths[i];
      double? sLat, sLng, eLat, eLng;
      if (sp.mode == SubPathMode.bus || sp.mode == SubPathMode.subway) {
        sLat = sp.effectiveStartLat;
        sLng = sp.effectiveStartLng;
        eLat = sp.effectiveEndLat;
        eLng = sp.effectiveEndLng;
      } else if (sp.mode == SubPathMode.walk) {
        // 인접 transit 의 end / start 로 walk start/end 보완. 라우트 양 끝의
        // walk 면 origin/dest 사용.
        SubPath? prev = i > 0 ? route.subPaths[i - 1] : null;
        SubPath? next = i < route.subPaths.length - 1
            ? route.subPaths[i + 1]
            : null;
        sLat = prev?.effectiveEndLat ?? originLat;
        sLng = prev?.effectiveEndLng ?? originLng;
        eLat = next?.effectiveStartLat ?? destLat;
        eLng = next?.effectiveStartLng ?? destLng;
      }
      if (sLat == null || sLng == null || eLat == null || eLng == null) {
        debugPrint('[GILBEOT_directions]   sp[$i] skip — null coord');
        continue;
      }
      // 도보 구간도 driving 으로 도로 따라가는 경로를 받음. 자동차 일방통행
      // 우회가 가끔 비효율적이긴 하지만, 직선은 건물 가로지르는 더 큰 사고를
      // 야기. 도보는 시각적으로 *점선* 으로 구분해 어르신이 "여기서 잠깐
      // 걸어요" 인지하도록 (overlay 단에서 처리).
      debugPrint(
        '[GILBEOT_directions]   sp[$i] ${sp.mode.name} req '
        '$sLng,$sLat → $eLng,$eLat',
      );
      // 도보는 T-Map 보행자, 대중교통은 T-Map 자동차. 두 API 가 동일 엔진
      // 이라 segment 사이 연결 좌표가 매끄러움 (NCP/T-Map 혼합 시 미세 어긋남
      // 가능성 제거). ODsay 는 라우트 결정(어느 버스/환승/출구) 만 담당.
      final Future<List<List<double>>> req = sp.mode == SubPathMode.walk
          ? TMapPedestrianService.instance.pedestrianPath(
              startLng: sLng,
              startLat: sLat,
              endLng: eLng,
              endLat: eLat,
            )
          : TMapPedestrianService.instance.carPath(
              startLng: sLng,
              startLat: sLat,
              endLng: eLng,
              endLat: eLat,
            );
      futures.add(
        req.then((path) {
          debugPrint(
            '[GILBEOT_directions]   sp[$i] ${sp.mode.name} got '
            '${path.length} points',
          );
          if (path.length >= 2) {
            sp.polyline = path;
          }
        }),
      );
    }
    await Future.wait(futures);
    debugPrint('[GILBEOT_directions] enrich done');
  }

  /// ODsay `loadLane` 으로 mapObj 에 해당하는 lane[].section[].graphPos[] 를
  /// 가져와서 각 transit subPath 의 `polyline` 에 채워넣음.
  ///
  /// 매핑 규칙: `lane[i]` ↔ i번째 transit subPath (도보 제외, 등장 순서대로).
  /// loadLane 응답은 대중교통 노선만 들고 있어서 도보(trafficType=3) subPath
  /// 는 graphPos 가 없음 → 인접 transit 끝/시작점 직선으로 보완 (`polylinePoints`
  /// getter 책임).
  Future<void> _loadLanePolyline(RoutePath route) async {
    if (route.mapObj.isEmpty) return;
    // ODsay search 응답의 info.mapObj 는 `ID:Class:s:e[@ID:Class:s:e...]`
    // 형식이지만, loadLane 호출 시엔 *`BaseX:BaseY@` prefix 가 필수* (없으면
    // error -8). `0:0@` 가 좌표 offset 없이 절대 WGS84 lng/lat 그대로 반환.
    final r = await _dio.get(
      '/loadLane',
      queryParameters: {'apiKey': _key, 'mapObject': '0:0@${route.mapObj}'},
    );
    final body = r.data as Map<String, dynamic>;
    if (body.containsKey('error')) return;
    final lanes = (body['result']?['lane'] as List?) ?? const [];
    final transitIndices = <int>[];
    for (var i = 0; i < route.subPaths.length; i++) {
      final m = route.subPaths[i].mode;
      if (m == SubPathMode.bus || m == SubPathMode.subway) {
        transitIndices.add(i);
      }
    }
    for (var li = 0; li < lanes.length && li < transitIndices.length; li++) {
      final lane = lanes[li] as Map<String, dynamic>;
      final sections = (lane['section'] as List?) ?? const [];
      final pts = <List<double>>[];
      for (final sect in sections) {
        final graph = (sect as Map<String, dynamic>)['graphPos'] as List?;
        if (graph == null) continue;
        for (final g in graph) {
          final gp = g as Map<String, dynamic>;
          final x = (gp['x'] as num?)?.toDouble();
          final y = (gp['y'] as num?)?.toDouble();
          if (x != null && y != null) pts.add([x, y]);
        }
      }
      route.subPaths[transitIndices[li]].polyline = pts;
    }
  }

  /// ODsay 가 라우트 못 잡는 짧은 거리 / 같은 정류장권 fallback. T-Map 보행자
  /// 로 도보 polyline 받아 단일 walk SubPath 로 RoutePath 구성.
  Future<RoutePath> _walkOnlyRoute(
    double sx,
    double sy,
    double ex,
    double ey,
  ) async {
    final path = await TMapPedestrianService.instance.pedestrianPath(
      startLng: sx,
      startLat: sy,
      endLng: ex,
      endLat: ey,
    );
    final dist = _haversine(sy, sx, ey, ex).round();
    final walkMin = (dist / 60).ceil();
    final walk = SubPath(
      mode: SubPathMode.walk,
      sectionTimeMin: walkMin,
      distanceM: dist,
    );
    walk.polyline = path.length >= 2
        ? path
        : [
            [sx, sy],
            [ex, ey],
          ];
    return RoutePath(
      totalTimeMin: walkMin,
      totalWalkM: dist,
      mapObj: '',
      subPaths: [walk],
    );
  }

  /// 두 좌표 사이 거리 (m). Haversine — 한국 짧은 거리에서 ±수 미터 정확.
  double _haversine(double lat1, double lng1, double lat2, double lng2) {
    const r = 6371000.0;
    final dLat = (lat2 - lat1) * math.pi / 180;
    final dLng = (lng2 - lng1) * math.pi / 180;
    final a =
        math.sin(dLat / 2) * math.sin(dLat / 2) +
        math.cos(lat1 * math.pi / 180) *
            math.cos(lat2 * math.pi / 180) *
            math.sin(dLng / 2) *
            math.sin(dLng / 2);
    final c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a));
    return r * c;
  }
}

class RouteResult {
  RouteResult._({this.path, this.errorCode, this.errorMessage});
  factory RouteResult.success(RoutePath p) => RouteResult._(path: p);
  factory RouteResult.failure({
    required String code,
    required String message,
  }) => RouteResult._(errorCode: code, errorMessage: message);

  final RoutePath? path;
  final String? errorCode;
  final String? errorMessage;
  bool get ok => path != null;
}

class RoutePath {
  RoutePath({
    required this.totalTimeMin,
    required this.totalWalkM,
    required this.subPaths,
    required this.mapObj,
  });
  final int totalTimeMin;
  final int totalWalkM;
  final List<SubPath> subPaths;

  /// ODsay `loadLane` API 호출용 식별자. 라우트 전체에 대해 한 값.
  final String mapObj;

  factory RoutePath.fromJson(Map<String, dynamic> j) {
    final info = j['info'] as Map<String, dynamic>? ?? const {};
    final raw = (j['subPath'] as List?) ?? const [];
    return RoutePath(
      totalTimeMin: (info['totalTime'] as num?)?.toInt() ?? 0,
      totalWalkM: (info['totalWalk'] as num?)?.toInt() ?? 0,
      mapObj: info['mapObj']?.toString() ?? '',
      subPaths: raw
          .map((e) => SubPath.fromJson(e as Map<String, dynamic>))
          .toList(),
    );
  }

  /// 출발 → (각 subPath 의 실 경로 or 정류장 직선) → 도착 단일 polyline.
  ///
  /// Transit subPath 는 loadLane 으로 채워진 `polyline` 사용, walk subPath 는
  /// 인접 transit 의 끝/시작점만 통과시켜 짧은 직선으로 보완. 어차피 도보
  /// 구간은 정류장 ↔ 정류장 단거리라 어르신이 "이쯤이구나" 인지하는 데 충분.
  ///
  /// 형식: `[lng, lat]` 순. CustomPaint 와 Mercator 변환 모두 (x, y) 순서로
  /// 처리하므로 lng 가 먼저.
  List<List<double>> get polylinePoints {
    final pts = <List<double>>[];
    for (final sp in subPaths) {
      if (sp.polyline.isNotEmpty) {
        pts.addAll(sp.polyline);
      } else {
        final sLat = sp.effectiveStartLat;
        final sLng = sp.effectiveStartLng;
        final eLat = sp.effectiveEndLat;
        final eLng = sp.effectiveEndLng;
        if (sLat != null && sLng != null) pts.add([sLng, sLat]);
        if (eLat != null && eLng != null) pts.add([eLng, eLat]);
      }
    }
    // 인접 중복 제거 (subPath 경계에서 흔히 발생).
    final dedup = <List<double>>[];
    for (final p in pts) {
      if (dedup.isEmpty || dedup.last[0] != p[0] || dedup.last[1] != p[1]) {
        dedup.add(p);
      }
    }
    return dedup;
  }
}

enum SubPathMode { subway, bus, walk, unknown }

class SubPath {
  SubPath({
    required this.mode,
    required this.sectionTimeMin,
    this.distanceM,
    this.startStationName,
    this.endStationName,
    this.startExitNo,
    this.endExitNo,
    this.startExitX,
    this.startExitY,
    this.endExitX,
    this.endExitY,
    this.startX,
    this.startY,
    this.endX,
    this.endY,
    this.lane,
  });

  final SubPathMode mode;
  final int sectionTimeMin;
  final int? distanceM;
  final String? startStationName;
  final String? endStationName;
  final String? startExitNo;
  final String? endExitNo;
  final double? startExitX;
  final double? startExitY;
  final double? endExitX;
  final double? endExitY;
  // Generic 좌표 — ODsay 가 bus/subway subPath 의 출발/도착 정류장에 채움.
  // walk subPath 에는 채워지지 않음 → 인접 transit subPath 에서 derive.
  final double? startX;
  final double? startY;
  final double? endX;
  final double? endY;
  final String? lane; // 호선/노선

  /// ODsay `loadLane` 으로 채워지는 실 도로/철도 좌표. `[lng, lat]` 순.
  /// Transit subPath (bus/subway) 만 채워지고 walk 는 빈 리스트로 남음 —
  /// `RoutePath.polylinePoints` getter 가 인접 정류장 직선으로 보완.
  List<List<double>> polyline = [];

  /// 이 subPath 의 effective start lat/lng (출구 좌표 우선, 없으면 generic).
  double? get effectiveStartLat => startExitY ?? startY;
  double? get effectiveStartLng => startExitX ?? startX;
  double? get effectiveEndLat => endExitY ?? endY;
  double? get effectiveEndLng => endExitX ?? endX;

  factory SubPath.fromJson(Map<String, dynamic> j) {
    final t = j['trafficType'];
    final mode = switch (t) {
      1 => SubPathMode.subway,
      2 => SubPathMode.bus,
      3 => SubPathMode.walk,
      _ => SubPathMode.unknown,
    };
    // ODsay `lane[]` 의 식별 키가 mode 별로 다름:
    //   - 지하철 (trafficType=1): `name` (예: "수도권 2호선")
    //   - 버스   (trafficType=2): `busNo` (예: "341", "마을06") — `name` 필드 자체가 없음
    // 둘을 같은 `SubPath.lane` 문자열로 평탄화해서 downstream (`_vehicleLabel`,
    // `toPromptContext`) 가 분기 없이 다룬다. 이전 구현은 두 mode 모두 `name`
    // 만 읽어 버스 노선번호가 항상 null 로 떨어졌고, journey overview/LLM 컨텍스트
    // 가 "버스를 타세요" 같은 일반화된 문장으로 fallback 됐다.
    final lanes = (j['lane'] as List?)?.cast<Map<String, dynamic>>();
    final laneKey = mode == SubPathMode.bus ? 'busNo' : 'name';
    final laneName = lanes?.isNotEmpty == true
        ? lanes!.first[laneKey]?.toString()
        : null;
    return SubPath(
      mode: mode,
      sectionTimeMin: (j['sectionTime'] as num?)?.toInt() ?? 0,
      distanceM: (j['distance'] as num?)?.toInt(),
      startStationName: j['startName']?.toString(),
      endStationName: j['endName']?.toString(),
      startExitNo: j['startExitNo']?.toString(),
      endExitNo: j['endExitNo']?.toString(),
      startExitX: (j['startExitX'] as num?)?.toDouble(),
      startExitY: (j['startExitY'] as num?)?.toDouble(),
      endExitX: (j['endExitX'] as num?)?.toDouble(),
      endExitY: (j['endExitY'] as num?)?.toDouble(),
      startX: (j['startX'] as num?)?.toDouble(),
      startY: (j['startY'] as num?)?.toDouble(),
      endX: (j['endX'] as num?)?.toDouble(),
      endY: (j['endY'] as num?)?.toDouble(),
      lane: laneName,
    );
  }

  /// Gemma prompt에 주입할 한국어 컨텍스트 한 줄.
  String toPromptContext() {
    return switch (mode) {
      SubPathMode.subway =>
        '지하철 ${lane ?? ""} ${startStationName ?? "?"}역에서 ${endStationName ?? "?"}역까지. ${endStationName ?? "?"}역의 ${endExitNo ?? "?"}번 출구로 나와야 합니다.',
      // 지하철 템플릿과 평행한 풀 문장 형태로 LLM 에 전달 — 모델이 노선번호/
      // 승차/하차 정류장명을 그대로 사용자 안내에 반영하도록.
      SubPathMode.bus =>
        '버스 ${lane ?? "?"}번을 ${startStationName ?? "?"} 정류장에서 타고 '
            '${endStationName ?? "?"} 정류장에서 내려야 합니다.',
      SubPathMode.walk => () {
        // walk subPath 에 startStationName + startExitNo 가 채워진 경우
        // (walk-only fallback 의 출구 hint), "{역} {번호}번 출구로 나가셔서
        // ..." 형태로 안내 보강. 어르신이 지하철에서 어떻게 나가야 할지
        // 즉시 인지.
        final station = startStationName;
        final exit = startExitNo;
        if (station != null &&
            station.isNotEmpty &&
            exit != null &&
            exit.isNotEmpty) {
          return '$station $exit번 출구로 나오셔서 ${distanceM ?? 0}m, '
              '약 $sectionTimeMin분 걸어가셔야 합니다.';
        }
        return '도보 ${distanceM ?? 0}m, 약 $sectionTimeMin분';
      }(),
      SubPathMode.unknown => '알 수 없는 구간',
    };
  }
}
