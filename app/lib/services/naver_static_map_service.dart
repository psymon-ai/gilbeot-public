import 'dart:math' as math;

import 'package:flutter_dotenv/flutter_dotenv.dart';

import 'odsay_service.dart';

/// Naver Cloud Platform — Maps Static API (raster).
///
/// Static Map v2 raster API 가 *공식적으로 `path` 를 지원하지 않음* — 어떤
/// 형식으로 보내도 silently ignored. 그래서 두 단계로 합성:
///   1. 서버: 이 service 가 `markers` 만 박힌 PNG 를 받음
///   2. 클라이언트: `MapPreviewScreen` 이 `MapViewport` 메타데이터로 같은
///      Mercator projection 을 재현해 PNG 위에 폴리라인을 CustomPaint 로
///      overlay. 마커 위치와 1px 단위로 정렬됨.
///
/// 인증: NCP `X-NCP-APIGW-API-KEY-ID` / `X-NCP-APIGW-API-KEY` 헤더 +
/// 신 endpoint `maps.apigw.ntruss.com` (구 `naveropenapi` 는 deprecated 되어
/// 같은 키로도 401).
class NaverStaticMapService {
  NaverStaticMapService._();
  static final NaverStaticMapService instance = NaverStaticMapService._();

  // 2024 마이그레이션 후 신 host. 구 host 는 같은 키로 401 반환.
  static const String _baseUrl =
      'https://maps.apigw.ntruss.com/map-static/v2/raster';

  String get _clientId => dotenv.maybeGet('NAVER_CLIENT_ID')?.trim() ?? '';
  String get _clientSecret => dotenv.maybeGet('NAVER_CLIENT_SECRET')?.trim() ?? '';

  Map<String, String> get authHeaders => {
        'X-NCP-APIGW-API-KEY-ID': _clientId,
        'X-NCP-APIGW-API-KEY': _clientSecret,
      };

  bool get hasCredentials =>
      _clientId.isNotEmpty && _clientSecret.isNotEmpty;

  /// 경로의 환승 정류장/역. NCP `type:n` 가 0-9 만 지원해서 cap 9.
  List<TransitStop> transitStops(RoutePath route) {
    final stops = <TransitStop>[];
    int idx = 1;
    for (final sp in route.subPaths) {
      if (sp.mode != SubPathMode.subway && sp.mode != SubPathMode.bus) {
        continue;
      }
      final lat = sp.effectiveStartLat;
      final lng = sp.effectiveStartLng;
      if (lat == null || lng == null || idx > 9) continue;
      stops.add(
        TransitStop(
          index: idx,
          lat: lat,
          lng: lng,
          stationName: sp.startStationName ?? '?',
          endStationName: sp.endStationName ?? '?',
          lane: sp.lane ?? '',
          mode: sp.mode,
        ),
      );
      idx++;
    }
    return stops;
  }

  /// 한 번 계산하면 URL · overlay painter 둘 다 같은 projection 으로 일치.
  MapViewport computeViewport({
    required RoutePath route,
    required double originLat,
    required double originLng,
    required double destLat,
    required double destLng,
  }) {
    final stops = transitStops(route);

    // ── 폴리라인: 출발 → ODsay loadLane 실 경로 → 도착 ─────────────────
    // RoutePath.polylinePoints 가 transit subPath 의 graphPos + walk 보완을
    // 합쳐서 단일 좌표 리스트로 반환. 앞뒤로 사용자 GPS 출발지/도착지
    // 좌표를 끼워넣어 마커 위치와 polyline 끝점이 정확히 만나도록.
    final polyline = <List<double>>[
      [originLng, originLat],
      ...route.polylinePoints,
      [destLng, destLat],
    ];
    final dedup = <List<double>>[];
    for (final p in polyline) {
      if (dedup.isEmpty ||
          dedup.last[0] != p[0] ||
          dedup.last[1] != p[1]) {
        dedup.add(p);
      }
    }

    // ── Bounding box: polyline 전체 + 마커 ─────────────────────────────
    // 마커 + polyline 모두 viewport 안에 들어와야 어르신이 경로 어디로 흘러
    // 가는지 처음부터 끝까지 한 화면에 인지 가능.
    double minLng = originLng, maxLng = originLng;
    double minLat = originLat, maxLat = originLat;
    void include(double lng, double lat) {
      if (lng < minLng) minLng = lng;
      if (lng > maxLng) maxLng = lng;
      if (lat < minLat) minLat = lat;
      if (lat > maxLat) maxLat = lat;
    }
    include(destLng, destLat);
    for (final s in stops) {
      include(s.lng, s.lat);
    }
    for (final p in dedup) {
      include(p[0], p[1]);
    }

    final centerLng = (minLng + maxLng) / 2;
    final centerLat = (minLat + maxLat) / 2;

    // 가시 영역 30% 여유 → "도착" 라벨이 가장자리에서 잘리는 사고 방지.
    final spanLng = (maxLng - minLng).abs() * 1.3;
    final spanLat = (maxLat - minLat).abs() * 1.3;
    final span = spanLng > spanLat ? spanLng : spanLat;
    int level;
    if (span < 0.003) {
      level = 16;
    } else if (span < 0.007) {
      level = 15;
    } else if (span < 0.015) {
      level = 14;
    } else if (span < 0.03) {
      level = 13;
    } else if (span < 0.06) {
      level = 12;
    } else {
      level = 11;
    }

    return MapViewport(
      centerLng: centerLng,
      centerLat: centerLat,
      level: level,
      scale: 2,
      polylinePoints: dedup,
      originLng: originLng,
      originLat: originLat,
      destLng: destLng,
      destLat: destLat,
      transitStops: stops,
    );
  }

  /// NCP Static Map URL. `viewport` 가 center/level/scale 을 들고 있어서
  /// CustomPaint overlay 와 *완전히 동일한* projection 으로 마커 그려짐.
  String buildUrl(MapViewport v, {required int width, required int height}) {
    final markers = <String>[];
    markers.add(_encodeMarker(
      type: 't',
      color: '0x2E7D32',
      label: '출발',
      lng: v.originLng,
      lat: v.originLat,
    ));
    for (final s in v.transitStops) {
      markers.add(_encodeMarker(
        type: 'n',
        color: '0x1565C0',
        label: '${s.index}',
        lng: s.lng,
        lat: s.lat,
      ));
    }
    markers.add(_encodeMarker(
      type: 't',
      color: '0xC62828',
      label: '도착',
      lng: v.destLng,
      lat: v.destLat,
    ));

    final markerParams = markers.map((m) => 'markers=$m').join('&');

    return '$_baseUrl'
        '?w=$width&h=$height'
        '&center=${_fmt(v.centerLng)},${_fmt(v.centerLat)}'
        '&level=${v.level}'
        '&scale=${v.scale}'
        '&$markerParams';
  }

  /// NCP 변태 사양:
  ///   - `pos:` lng/lat 사이는 *공백* (쉼표는 403)
  ///   - 공백은 `+` 아닌 `%20` 으로 인코딩 (`+` 거부)
  ///   - `|`, `:` 는 raw 유지
  ///   - 한글 라벨은 UTF-8 percent-encode
  String _encodeMarker({
    required String type,
    required String color,
    required String label,
    required double lng,
    required double lat,
  }) {
    final encodedLabel = Uri.encodeQueryComponent(label).replaceAll('+', '%20');
    return 'type:$type|size:mid|color:$color|label:$encodedLabel'
        '|pos:${_fmt(lng)}%20${_fmt(lat)}';
  }

  String _fmt(double v) => v.toStringAsFixed(6);
}

/// NCP Static Map 한 장을 완전히 기술하는 메타데이터. 서버 URL 빌드와
/// 클라이언트 CustomPaint overlay 가 *동일한* projection 으로 동작하도록
/// 단일 source of truth 역할.
class MapViewport {
  MapViewport({
    required this.centerLng,
    required this.centerLat,
    required this.level,
    required this.scale,
    required this.polylinePoints,
    required this.originLng,
    required this.originLat,
    required this.destLng,
    required this.destLat,
    required this.transitStops,
  });

  final double centerLng;
  final double centerLat;
  final int level;
  final int scale;
  /// `[lng, lat]` 순서 — Web Mercator 변환 시 (x, y) 순서와 일관.
  final List<List<double>> polylinePoints;
  final double originLng;
  final double originLat;
  final double destLng;
  final double destLat;
  final List<TransitStop> transitStops;
}

class TransitStop {
  TransitStop({
    required this.index,
    required this.lat,
    required this.lng,
    required this.stationName,
    required this.endStationName,
    required this.lane,
    required this.mode,
  });

  final int index;
  final double lat;
  final double lng;
  final String stationName;
  final String endStationName;
  final String lane;
  final SubPathMode mode;

  /// 화면 카드용 한 줄. 예: "333번 버스 · 방이중학교 → 호수임광아파트"
  String describe() {
    final vehicle = switch (mode) {
      SubPathMode.bus => lane.isEmpty ? '버스' : '$lane번 버스',
      SubPathMode.subway => lane.isEmpty ? '지하철' : '$lane 지하철',
      _ => '',
    };
    return '$vehicle · $stationName → $endStationName';
  }
}

/// Web Mercator (EPSG:3857) projection 헬퍼.
///
/// NCP Static Map v2 는 `crs` 기본값이 EPSG:4326 (WGS84 lng/lat 입력) 이지만
/// 내부 렌더 좌표계는 EPSG:3857 — 표준 web tile map 과 동일. 우리도 같은
/// projection 으로 좌표를 계산하면 NCP 가 그린 마커 위에 1px 단위로 정확히
/// 정렬되는 폴리라인을 그릴 수 있음.
class MercatorProjection {
  MercatorProjection._();

  static const double _earthRadius = 6378137.0;
  static const double _originShift = math.pi * _earthRadius;

  /// (lng_deg, lat_deg) → (mx_m, my_m) in EPSG:3857.
  static List<double> lngLatToMeters(double lng, double lat) {
    final mx = lng * _originShift / 180.0;
    final clamped = lat.clamp(-85.05112878, 85.05112878);
    final radians = clamped * math.pi / 180.0;
    final my = math.log(math.tan(math.pi / 4 + radians / 2)) * _earthRadius;
    return [mx, my];
  }

  /// scale=1 기준 pixels per Mercator-meter at given zoom level.
  /// 표준 web tile: world = 256 * 2^level px.
  static double pixelsPerMeter(int level) {
    final worldPx = 256.0 * math.pow(2, level);
    return worldPx / (2 * _originShift);
  }
}
