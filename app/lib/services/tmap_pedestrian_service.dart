import 'package:dio/dio.dart';
import 'package:flutter_dotenv/flutter_dotenv.dart';

/// T-Map (SK텔레콤) routing + POI 검색 API — 도보 / 자동차 / 장소 검색을
/// 단일 vendor 로 통합.
///
/// 한국 routing 의 사실상 표준. 무료 60만 호출/월. POI 검색까지 같은 키 + 같은
/// 도메인으로 처리되므로 Kakao Local API 의존성을 제거할 수 있다 (FU107).
///
/// 메서드:
///   - `pedestrianPath`: 도보 segment (인도/횡단보도/공원길 반영)
///   - `carPath`:        버스/지하철 segment 의 도로 polyline (버스가 도로
///                       따라가니 자동차 routing 으로 보완)
///   - `searchPlace`:    POI 키워드 검색 (목적지명 → 좌표). 이전 Kakao Local
///                       의 `KakaoService.searchPlace` 와 같은 signature 라
///                       home_screen 의 호출처는 임포트만 바꾸면 된다.
///
/// ODsay 가 대중교통 *라우트 결정* (어느 버스/지하철, 환승, 출구번호) 만
/// 담당하고, polyline 좌표는 모두 T-Map 으로 통일 — 동일 routing 엔진의
/// 좌표라 segment 사이 연결이 매끄럽고 정확도 일관성 확보.
///
/// 인증: `appKey` HTTP header. 가입은 SK Open API 포털(openapi.sk.com) 에서.
class TMapPedestrianService {
  TMapPedestrianService._();
  static final TMapPedestrianService instance = TMapPedestrianService._();

  final Dio _dio = Dio(BaseOptions(
    baseUrl: 'https://apis.openapi.sk.com/tmap',
    connectTimeout: const Duration(seconds: 5),
    receiveTimeout: const Duration(seconds: 8),
  ));

  String get _key => dotenv.maybeGet('TMAP_APP_KEY')?.trim() ?? '';
  bool get hasKey => _key.isNotEmpty;

  /// POI 키워드 검색. 첫 번째 결과를 [Place] 로 반환. 이전 Kakao Local
  /// `keyword.json` 호출 대체 — T-Map POI 통합 검색 (한국 시설/지하철역/
  /// 상호 모두 cover).
  Future<Place?> searchPlace(String query, {int count = 5}) async {
    if (!hasKey) return null;
    try {
      final r = await _dio.get(
        '/pois',
        queryParameters: {
          'version': '1',
          'searchKeyword': query,
          'searchType': 'all',
          'count': count,
          'resCoordType': 'WGS84GEO',
          'reqCoordType': 'WGS84GEO',
        },
        options: Options(headers: {'appKey': _key}),
      );
      final pois =
          ((r.data as Map?)?['searchPoiInfo'] as Map?)?['pois'] as Map?;
      final poiList = pois?['poi'] as List?;
      if (poiList == null || poiList.isEmpty) return null;
      final first = poiList.first as Map<String, dynamic>;
      final lat = double.tryParse(
          '${first['frontLat'] ?? first['noorLat'] ?? ''}');
      final lng = double.tryParse(
          '${first['frontLon'] ?? first['noorLon'] ?? ''}');
      if (lat == null || lng == null) return null;
      return Place(
        name: first['name']?.toString() ?? query,
        address: _addrFromTmap(first),
        lat: lat,
        lng: lng,
      );
    } catch (_) {
      return null;
    }
  }

  /// T-Map POI 응답에서 사람이 읽을 수 있는 주소 문자열 추출. 신주소
  /// (newAddressList) 우선, fallback 으로 4단(법정/행정/도로/상세) 결합.
  String _addrFromTmap(Map<String, dynamic> poi) {
    final newAddr = poi['newAddressList'] as Map?;
    if (newAddr != null) {
      final list = newAddr['newAddress'] as List?;
      if (list != null && list.isNotEmpty) {
        final full =
            (list.first as Map?)?['fullAddressRoad']?.toString() ?? '';
        if (full.isNotEmpty) return full;
      }
    }
    return [
      poi['upperAddrName'],
      poi['middleAddrName'],
      poi['lowerAddrName'],
      poi['detailAddrName'],
    ]
        .where((s) => s != null && s.toString().isNotEmpty)
        .join(' ')
        .trim();
  }

  /// 도보 routing.
  Future<List<List<double>>> pedestrianPath({
    required double startLng,
    required double startLat,
    required double endLng,
    required double endLat,
  }) {
    return _route(
      path: '/routes/pedestrian?version=1',
      startLng: startLng,
      startLat: startLat,
      endLng: endLng,
      endLat: endLat,
    );
  }

  /// 자동차 routing — 버스/지하철 segment 의 도로 polyline 보완용.
  Future<List<List<double>>> carPath({
    required double startLng,
    required double startLat,
    required double endLng,
    required double endLat,
  }) {
    return _route(
      path: '/routes?version=1',
      startLng: startLng,
      startLat: startLat,
      endLng: endLng,
      endLat: endLat,
    );
  }

  /// 두 routing 모두 같은 응답 형식 (GeoJSON FeatureCollection) 이라 단일
  /// 헬퍼로 LineString features 좌표를 합쳐 반환.
  Future<List<List<double>>> _route({
    required String path,
    required double startLng,
    required double startLat,
    required double endLng,
    required double endLat,
  }) async {
    if (!hasKey) return const [];
    try {
      final r = await _dio.post(
        path,
        data: {
          'startX': startLng,
          'startY': startLat,
          'endX': endLng,
          'endY': endLat,
          'startName': '출발',
          'endName': '도착',
        },
        options: Options(headers: {
          'appKey': _key,
          'Content-Type': 'application/json',
        }),
      );
      final body = r.data as Map<String, dynamic>;
      final features = body['features'] as List?;
      if (features == null) return const [];
      final pts = <List<double>>[];
      for (final f in features) {
        final feature = f as Map<String, dynamic>;
        final geom = feature['geometry'] as Map<String, dynamic>?;
        if (geom == null) continue;
        if (geom['type'] != 'LineString') continue;
        final coords = geom['coordinates'] as List?;
        if (coords == null) continue;
        for (final c in coords) {
          final x = (c[0] as num).toDouble();
          final y = (c[1] as num).toDouble();
          pts.add([x, y]);
        }
      }
      return pts;
    } catch (_) {
      return const [];
    }
  }
}

/// 장소 정보 — 이름 + 주소 + WGS84 좌표. T-Map POI 검색 결과 + 데모 캔드
/// origin/dest 표현용. (FU107 이전엔 kakao_service.dart 에 정의돼 있었으나
/// Kakao 의존성 제거하면서 T-Map service 파일로 이동.)
class Place {
  Place({
    required this.name,
    required this.address,
    required this.lat,
    required this.lng,
    this.category,
  });

  final String name;
  final String address;
  final double lat;
  final double lng;
  final String? category;
}
