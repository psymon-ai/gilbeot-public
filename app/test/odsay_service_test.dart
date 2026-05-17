// Regression tests for the ODsay `lane[]` field parsing in SubPath.
//
// 배경: ODsay `searchPubTransPathT` 응답에서 `lane` 배열의 식별 키가 mode 별로 다름.
//   - 지하철 (trafficType=1): `name` (예: "수도권 2호선")
//   - 버스   (trafficType=2): `busNo` (예: "341") — `name` 필드 자체가 없음
// 이전 구현은 두 mode 모두 `name` 만 읽어 버스 lane 이 항상 null 이 되었고,
// downstream 의 "X번 버스" 안내가 그냥 "버스" 로 fallback 되는 버그가 있었음.
//
// 픽스처는 `scripts/verify_out/odsay_long.json` 에서 실제 캡쳐된 응답을 발췌.

import 'package:flutter_test/flutter_test.dart';
import 'package:gilbeot/services/odsay_service.dart';

void main() {
  group('SubPath.fromJson', () {
    test('버스 (trafficType=2) — busNo 필드에서 노선번호를 읽는다', () {
      // 실제 ODsay 응답에서 발췌 (강남 → 시청 경로의 341번 버스 segment).
      final j = {
        'trafficType': 2,
        'distance': 6809,
        'sectionTime': 25,
        'stationCount': 13,
        'lane': [
          {
            'busNo': '341',
            'type': 11,
            'busID': 1435,
            'busLocalBlID': '100100056',
            'busCityCode': 1000,
            'busProviderCode': 4,
          },
        ],
        'startName': '강남역1번출구.역삼세무서',
        'endName': '강남파이낸스센터역',
        'startX': 127.0285,
        'startY': 37.4979,
        'endX': 126.9784,
        'endY': 37.5663,
      };

      final sp = SubPath.fromJson(j);

      expect(sp.mode, SubPathMode.bus);
      expect(sp.lane, '341',
          reason: 'busNo 가 SubPath.lane 으로 들어와야 _vehicleLabel 이 '
              '"341번 버스" 를 생성할 수 있음');
      expect(sp.startStationName, '강남역1번출구.역삼세무서');
      expect(sp.endStationName, '강남파이낸스센터역');

      final ctx = sp.toPromptContext();
      expect(ctx, contains('341'),
          reason: 'LLM 프롬프트 컨텍스트에 노선번호가 반드시 포함되어야 함');
      expect(ctx, contains('강남역1번출구.역삼세무서'));
      expect(ctx, contains('강남파이낸스센터역'));
      expect(ctx, isNot(contains('?')),
          reason: '모든 필드가 채워지면 fallback "?" 가 없어야 함');
    });

    test('지하철 (trafficType=1) — name 필드에서 호선명을 읽는다', () {
      // 실제 ODsay 응답에서 발췌 (강남 → 시청의 2호선 segment).
      final j = {
        'trafficType': 1,
        'distance': 6700,
        'sectionTime': 11,
        'stationCount': 6,
        'lane': [
          {
            'name': '수도권 2호선',
            'subwayCode': 2,
            'subwayCityCode': 1000,
          },
        ],
        'startName': '강남',
        'endName': '시청',
        'endExitNo': '4',
      };

      final sp = SubPath.fromJson(j);

      expect(sp.mode, SubPathMode.subway);
      expect(sp.lane, '수도권 2호선',
          reason: '버스 fix 가 지하철 파싱을 망가뜨리지 않아야 함 (regression guard)');
      expect(sp.startStationName, '강남');
      expect(sp.endStationName, '시청');
      expect(sp.endExitNo, '4');
    });

    test('버스 — lane 배열이 비면 lane=null (graceful)', () {
      final j = {
        'trafficType': 2,
        'lane': [],
        'startName': '미지의정류장',
        'endName': '다른정류장',
      };
      final sp = SubPath.fromJson(j);
      expect(sp.mode, SubPathMode.bus);
      expect(sp.lane, isNull);
      // 프롬프트 컨텍스트는 "?" 로 fallback 하되 정류장명은 살아있어야 함.
      final ctx = sp.toPromptContext();
      expect(ctx, contains('미지의정류장'));
      expect(ctx, contains('다른정류장'));
    });

    test('도보 (trafficType=3) — lane 무관, 거리/시간만', () {
      final j = {'trafficType': 3, 'distance': 157, 'sectionTime': 2};
      final sp = SubPath.fromJson(j);
      expect(sp.mode, SubPathMode.walk);
      expect(sp.distanceM, 157);
      expect(sp.sectionTimeMin, 2);
      expect(sp.toPromptContext(), contains('157m'));
    });
  });
}
