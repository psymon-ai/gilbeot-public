import 'dart:ui' as ui;

import 'package:flutter/material.dart';
import 'package:flutter_map/flutter_map.dart';
import 'package:flutter_naver_map/flutter_naver_map.dart';
import 'package:latlong2/latlong.dart' as ll2;

import '../config/demo_mode.dart';
import '../services/naver_static_map_service.dart';
import '../services/odsay_service.dart';

/// 어르신 친화 인터랙티브 경로 미리보기.
///
/// NCP Mobile Dynamic Map SDK (Flutter wrapper) 사용. Native rendering 이라
/// Static Map raster (path 미지원/줌 불가) 와 WebView+JS (hybrid origin 문제)
/// 의 한계를 모두 해결:
///   - 확대/축소·드래그·핀치 모두 지원
///   - Polyline, Marker 가 네이티브 overlay
///   - 인증은 Android applicationId (`com.psymon.gilbeot`) 화이트리스트 →
///     Web URL 등록 불필요, production-safe
///   - 짤림 방지는 `NCameraUpdate.fitBounds` 로 모든 좌표 + padding 자동 fit
class MapPreviewScreen extends StatefulWidget {
  const MapPreviewScreen({
    super.key,
    required this.route,
    required this.originLat,
    required this.originLng,
    required this.destLat,
    required this.destLng,
    required this.destName,
  });

  final RoutePath route;
  final double originLat;
  final double originLng;
  final double destLat;
  final double destLng;
  final String destName;

  @override
  State<MapPreviewScreen> createState() => _MapPreviewScreenState();
}

class _MapPreviewScreenState extends State<MapPreviewScreen> {
  // 마커·overlay 색 토큰 (Static Map 시절 디자인 그대로 유지 — 어르신 인지 일관성)
  static const Color _startColor = Color(0xFF2E7D32);
  static const Color _busColor = Color(0xFF1565C0);
  static const Color _endColor = Color(0xFFC62828);
  static const Color _walkColor = Color(0xFF00897B); // teal — 도보 dot
  static const Color _defaultTransitColor = Color(0xFF00897B);

  /// 수도권 지하철 노선 공식 색상표. ODsay 의 `lane` 필드 값 (예: "수도권 2호선",
  /// "신분당선", "수인분당선") 에 substring 매치. 어르신이 평소 지하철 노선도
  /// 에서 보던 색감 그대로라 호선 인지가 즉각적.
  static Color _subwayColor(String? lane) {
    if (lane == null || lane.isEmpty) return _defaultTransitColor;
    // 더 긴 이름 먼저 매치 (예: "신분당" 이 "분당" 보다 먼저, "수인분당" 도)
    if (lane.contains('GTX-A')) return const Color(0xFF9E4510);
    if (lane.contains('우이신설')) return const Color(0xFFB7C452);
    if (lane.contains('수인분당')) return const Color(0xFFF5A200);
    if (lane.contains('신분당')) return const Color(0xFFD31145);
    if (lane.contains('경의중앙')) return const Color(0xFF77C4A3);
    if (lane.contains('공항')) return const Color(0xFF0090D2);
    if (lane.contains('경춘')) return const Color(0xFF178C72);
    if (lane.contains('서해')) return const Color(0xFF81A914);
    if (lane.contains('신림')) return const Color(0xFF6789CA);
    if (lane.contains('분당')) return const Color(0xFFF5A200);
    if (lane.contains('1호선')) return const Color(0xFF0052A4);
    if (lane.contains('2호선')) return const Color(0xFF00A84D);
    if (lane.contains('3호선')) return const Color(0xFFEF7C1C);
    if (lane.contains('4호선')) return const Color(0xFF00A5DE);
    if (lane.contains('5호선')) return const Color(0xFF996CAC);
    if (lane.contains('6호선')) return const Color(0xFFCD7C2F);
    if (lane.contains('7호선')) return const Color(0xFF747F00);
    if (lane.contains('8호선')) return const Color(0xFFE6186C);
    if (lane.contains('9호선')) return const Color(0xFFBDB092);
    return _defaultTransitColor;
  }

  /// subPath 의 시각화 색 — 지하철은 호선별, 버스/도보는 토큰.
  static Color _subPathColor(SubPath sp) {
    return switch (sp.mode) {
      SubPathMode.subway => _subwayColor(sp.lane),
      SubPathMode.bus => _busColor,
      SubPathMode.walk => _walkColor,
      _ => _defaultTransitColor,
    };
  }

  /// chip strip 의 환승 stop badge 색 — TransitStop 에는 SubPath 가 없어
  /// stop 의 mode + lane 으로 직접 결정.
  static Color _transitStopColor(TransitStop s) {
    return switch (s.mode) {
      SubPathMode.subway => _subwayColor(s.lane),
      SubPathMode.bus => _busColor,
      _ => _defaultTransitColor,
    };
  }

  late final MapViewport _viewport;
  NaverMapController? _controller;

  @override
  void initState() {
    super.initState();
    _viewport = NaverStaticMapService.instance.computeViewport(
      route: widget.route,
      originLat: widget.originLat,
      originLng: widget.originLng,
      destLat: widget.destLat,
      destLng: widget.destLng,
    );
  }

  @override
  void dispose() {
    // NaverMap controller / overlay GPU memory 명시 release. plugin 이 widget
    // dispose 시 자동 cleanup 한다고 명시하지만, S23 + Android 16 처럼 메모리
    // 압박 marginal 한 디바이스에선 explicit clearOverlays 가 다음 BYO /
    // 사진 분석 진입 직전 GPU memory peak 를 더 빨리 풀어준다. 그 다음 LMK
    // target 확률 ↓.
    try {
      _controller?.clearOverlays();
    } catch (_) {}
    _controller = null;
    super.dispose();
  }

  /// 지도 준비 완료 콜백 — 오버레이 추가 + 카메라 fit.
  ///
  /// 모든 폴리라인 점 + 출발/도착/환승 마커가 한 화면에 들어오도록 `fitBounds`.
  /// 어르신이 시작과 끝을 한 번에 인지하도록 padding 80px (라벨 여유).
  Future<void> _onMapReady(NaverMapController controller) async {
    _controller = controller;
    // ── Bounding box ──────────────────────────────────────────────────
    final bounds = _computeBounds();

    // ── 폴리라인: subPath 별로 분리 ─────────────────────────────────
    // 네이버 지도 스타일 참고 — 색은 teal 통일, 모양으로 구분:
    //   - 도보       = 작은 dot 점선 (width 7, pattern [1, 22], lineCap.round)
    //   - 대중교통   = 굵은 실선 (width 10) + 흰 outline (width 14) 두 겹.
    //                  도로 위에서 도드라져 어르신 가시성 확보.
    final segments = <NAddableOverlay>{};
    for (var i = 0; i < widget.route.subPaths.length; i++) {
      final sp = widget.route.subPaths[i];
      if (sp.polyline.length < 2) continue;
      final coords = [for (final p in sp.polyline) NLatLng(p[1], p[0])];
      final color = _subPathColor(sp);
      if (sp.mode == SubPathMode.walk) {
        segments.add(
          NPolylineOverlay(
            id: 'seg-$i-walk',
            coords: coords,
            color: color,
            width: 8,
            lineCap: NLineCap.round,
            lineJoin: NLineJoin.round,
            pattern: const [4, 8],
          ),
        );
      } else {
        // 대중교통 = 흰 outline + 모드/호선별 본체. 지하철은 호선 공식 색상
        // (어르신이 노선도에서 보던 색과 동일).
        segments.add(
          NPolylineOverlay(
            id: 'seg-$i-outline',
            coords: coords,
            color: Colors.white,
            width: 14,
            lineCap: NLineCap.round,
            lineJoin: NLineJoin.round,
          ),
        );
        segments.add(
          NPolylineOverlay(
            id: 'seg-$i-main',
            coords: coords,
            color: color,
            width: 10,
            lineCap: NLineCap.round,
            lineJoin: NLineJoin.round,
          ),
        );
      }
    }
    if (segments.isNotEmpty) {
      await controller.addOverlayAll(segments);
    }

    // ── 마커: 출발 / 환승(번호) / 도착 ─────────────────────────────
    // NaverMap 기본 pin 의 iconTintColor blend 는 어두운 회색 위에 살짝
    // 톤만 입히는 정도라 chip 의 vivid green/red 와 시각 일치가 약하다.
    // Canvas 로 직접 그린 colored roundel 을 fromByteArray 로 주입해 화면에
    // 그대로 초록/빨강이 보이게 한다.
    final originIcon = await _solidPinIcon(_startColor);
    final destIcon = await _solidPinIcon(_endColor);
    final transitIcons = <int, NOverlayImage>{};
    for (final s in _viewport.transitStops) {
      transitIcons[s.index] = await _solidPinIcon(_transitStopColor(s));
    }

    final markers = <NAddableOverlay>{};
    markers.add(
      _chipMarker(
        id: 'origin',
        lat: _viewport.originLat,
        lng: _viewport.originLng,
        text: '출발',
        color: _startColor,
        icon: originIcon,
      ),
    );
    for (final s in _viewport.transitStops) {
      markers.add(
        _chipMarker(
          id: 'transit-${s.index}',
          lat: s.lat,
          lng: s.lng,
          text: '${s.index}',
          color: _transitStopColor(s),
          icon: transitIcons[s.index],
        ),
      );
    }
    markers.add(
      _chipMarker(
        id: 'dest',
        lat: _viewport.destLat,
        lng: _viewport.destLng,
        text: '도착',
        color: _endColor,
        icon: destIcon,
      ),
    );
    await controller.addOverlayAll(markers);

    // ── 카메라 fit (한 화면에 전체 경로 + 80px padding) ────────────
    await controller.updateCamera(
      NCameraUpdate.fitBounds(bounds, padding: const EdgeInsets.all(80)),
    );
  }

  /// 마커는 [icon] 으로 받은 colored roundel + caption 으로 표시. caption 이
  /// 한글 텍스트 라벨 ("출발", "도착", "1") + 색 배경. [icon] 이 null 이면
  /// NaverMap 기본 pin 을 [color] 로 tint (호환 fallback).
  NMarker _chipMarker({
    required String id,
    required double lat,
    required double lng,
    required String text,
    required Color color,
    NOverlayImage? icon,
  }) {
    return NMarker(
      id: id,
      position: NLatLng(lat, lng),
      icon: icon,
      caption: NOverlayCaption(
        text: text,
        textSize: 14,
        color: Colors.white,
        haloColor: color,
        minZoom: 0,
      ),
      // icon 이 직접 색을 가지면 tint 적용 X (이중 곱해져서 탁해짐).
      iconTintColor: icon != null ? Colors.transparent : color,
    );
  }

  /// Canvas → PNG → [NOverlayImage] 변환으로 vivid colored marker icon 을
  /// 생성. NaverMap 기본 pin 의 iconTintColor blend 가 약해서 (회색 pin 위에
  /// 살짝 색만 묻음) chip 의 강한 green/red 와 시각 일치가 안 되는 문제 해결.
  ///
  /// 디자인: 흰색 ring + colored disc. 그림자로 입체감. 픽셀 사이즈 96.
  /// 한 번 생성한 PNG byteArray 는 NaverMap 내부 cache 에 [cacheKey] 로 저장.
  Future<NOverlayImage> _solidPinIcon(Color color) async {
    const size = 96.0;
    final recorder = ui.PictureRecorder();
    final canvas = Canvas(recorder, const Rect.fromLTWH(0, 0, size, size));

    const cx = size / 2;
    const cy = size / 2;

    // Drop shadow
    canvas.drawCircle(
      const Offset(cx, cy + 3),
      size / 2 - 10,
      Paint()
        ..color = Colors.black38
        ..maskFilter = const MaskFilter.blur(BlurStyle.normal, 5),
    );

    // White outer ring
    canvas.drawCircle(
      const Offset(cx, cy),
      size / 2 - 6,
      Paint()..color = Colors.white,
    );

    // Colored inner disc
    canvas.drawCircle(
      const Offset(cx, cy),
      size / 2 - 14,
      Paint()..color = color,
    );

    final picture = recorder.endRecording();
    final image = await picture.toImage(size.toInt(), size.toInt());
    final byteData = await image.toByteData(format: ui.ImageByteFormat.png);
    final bytes = byteData!.buffer.asUint8List();
    // Keep Flutter 3.22 compatibility; Color.toARGB32 is newer.
    // ignore: deprecated_member_use
    final colorKey = color.value.toRadixString(16);
    return NOverlayImage.fromByteArray(bytes, cacheKey: 'pin-$colorKey');
  }

  /// 지도 위에 떠 있는 가로 chip strip. 출발 / 환승 N / 도착 의 *시각 토큰*
  /// (색 + 번호/이름) 이 지도 마커와 1:1 매칭되어 어르신이 "지도의 1번이
  /// 333번 버스 정류장" 임을 한눈에 인지.
  ///
  /// 가로 스크롤 가능 — 환승이 많은 라우트도 한 줄 유지.
  Widget _buildRouteChips(ThemeData theme, List<TransitStop> stops) {
    return SizedBox(
      height: 52,
      child: ListView(
        scrollDirection: Axis.horizontal,
        physics: const BouncingScrollPhysics(),
        children: [
          _chip(theme, color: _startColor, badge: '출발', label: '지금'),
          for (final s in stops)
            _chip(
              theme,
              color: _transitStopColor(s),
              badge: '${s.index}',
              label: _shortLane(s),
            ),
          _chip(theme, color: _endColor, badge: '도착', label: widget.destName),
        ],
      ),
    );
  }

  String _shortLane(TransitStop s) {
    return switch (s.mode) {
      SubPathMode.bus => s.lane.isEmpty ? '버스' : '${s.lane}번 버스',
      SubPathMode.subway => s.lane.isEmpty ? '지하철' : s.lane,
      _ => '',
    };
  }

  Widget _chip(
    ThemeData theme, {
    required Color color,
    required String badge,
    required String label,
  }) {
    final cs = theme.colorScheme;
    return Container(
      margin: const EdgeInsets.only(right: 8),
      padding: const EdgeInsets.fromLTRB(4, 4, 14, 4),
      decoration: BoxDecoration(
        color: cs.surfaceContainerHigh,
        borderRadius: BorderRadius.circular(28),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Container(
            width: 44,
            height: 44,
            alignment: Alignment.center,
            decoration: BoxDecoration(color: color, shape: BoxShape.circle),
            child: Text(
              badge,
              style: const TextStyle(
                color: Colors.white,
                fontSize: 14,
                fontWeight: FontWeight.w800,
              ),
            ),
          ),
          const SizedBox(width: 10),
          Text(
            label,
            style: theme.textTheme.titleMedium?.copyWith(
              fontWeight: FontWeight.w700,
            ),
          ),
        ],
      ),
    );
  }

  NLatLngBounds _computeBounds() {
    double minLat = _viewport.originLat;
    double maxLat = _viewport.originLat;
    double minLng = _viewport.originLng;
    double maxLng = _viewport.originLng;
    void inc(double lat, double lng) {
      if (lat < minLat) minLat = lat;
      if (lat > maxLat) maxLat = lat;
      if (lng < minLng) minLng = lng;
      if (lng > maxLng) maxLng = lng;
    }

    inc(_viewport.destLat, _viewport.destLng);
    for (final s in _viewport.transitStops) {
      inc(s.lat, s.lng);
    }
    for (final p in _viewport.polylinePoints) {
      inc(p[1], p[0]);
    }
    return NLatLngBounds(
      southWest: NLatLng(minLat, minLng),
      northEast: NLatLng(maxLat, maxLng),
    );
  }

  /// Demo 빌드용 지도 — NaverMap NCP 인증 없이 OSM 타일 + 커스텀 polyline /
  /// marker layer. realuse 빌드는 그대로 [NaverMap] 사용.
  ///
  /// `_viewport` (NaverStaticMapService 가 계산한 bounds + transit stops) 와
  /// `widget.route.subPaths` polyline 좌표를 latlong2 좌표로 변환. polyline
  /// 색은 _subPathColor 그대로 (walk teal, subway 호선 공식색, bus 파랑). walk
  /// 는 얇은 단일선 / 대중교통은 white outline + colored body — NaverMap 분기
  /// 와 같은 시각 hierarchy.
  Widget _buildDemoMap(ThemeData theme) {
    final naverBounds = _computeBounds();
    // LatLngBounds 는 flutter_map 의 클래스 (latlong2 에는 없음); LatLng 만
    // latlong2 의 것을 ll2 prefix 로 사용해 flutter_naver_map 의 NLatLng 와 분리.
    final bounds = LatLngBounds(
      ll2.LatLng(
        naverBounds.southWest.latitude,
        naverBounds.southWest.longitude,
      ),
      ll2.LatLng(
        naverBounds.northEast.latitude,
        naverBounds.northEast.longitude,
      ),
    );

    final polylines = <Polyline>[];
    for (final sp in widget.route.subPaths) {
      if (sp.polyline.length < 2) continue;
      final points = [
        for (final p in sp.polyline) ll2.LatLng(p[1], p[0]),
      ];
      final color = _subPathColor(sp);
      if (sp.mode == SubPathMode.walk) {
        polylines.add(Polyline(
          points: points,
          color: color,
          strokeWidth: 5,
        ));
      } else {
        polylines.add(Polyline(
          points: points,
          color: Colors.white,
          strokeWidth: 9,
        ));
        polylines.add(Polyline(
          points: points,
          color: color,
          strokeWidth: 6,
        ));
      }
    }

    final markers = <Marker>[
      Marker(
        point: ll2.LatLng(_viewport.originLat, _viewport.originLng),
        width: 44,
        height: 44,
        child: _flutterMapMarker('출발', _startColor),
      ),
      for (final s in _viewport.transitStops)
        Marker(
          point: ll2.LatLng(s.lat, s.lng),
          width: 44,
          height: 44,
          child: _flutterMapMarker('${s.index}', _transitStopColor(s)),
        ),
      Marker(
        point: ll2.LatLng(_viewport.destLat, _viewport.destLng),
        width: 44,
        height: 44,
        child: _flutterMapMarker('도착', _endColor),
      ),
    ];

    return FlutterMap(
      options: MapOptions(
        initialCameraFit: CameraFit.bounds(
          bounds: bounds,
          padding: const EdgeInsets.all(48),
        ),
        interactionOptions: const InteractionOptions(
          flags: InteractiveFlag.pinchZoom |
              InteractiveFlag.drag |
              InteractiveFlag.doubleTapZoom,
        ),
      ),
      children: [
        TileLayer(
          urlTemplate: 'https://tile.openstreetmap.org/{z}/{x}/{y}.png',
          userAgentPackageName: 'com.psymon.gilbeot.demo',
          maxZoom: 19,
        ),
        PolylineLayer(polylines: polylines),
        MarkerLayer(markers: markers),
        const RichAttributionWidget(
          attributions: [
            TextSourceAttribution('© OpenStreetMap contributors'),
          ],
        ),
      ],
    );
  }

  /// flutter_map Marker child — colored circle + 텍스트. NaverMap 측의
  /// `_solidPinIcon` (Canvas→PNG) 시각 토큰과 동일 (출발=초록, 도착=빨강,
  /// 환승=호선 공식색).
  Widget _flutterMapMarker(String label, Color color) {
    return Container(
      decoration: BoxDecoration(
        color: color,
        shape: BoxShape.circle,
        boxShadow: const [
          BoxShadow(color: Colors.black26, blurRadius: 4, offset: Offset(0, 2)),
        ],
      ),
      alignment: Alignment.center,
      child: Text(
        label,
        style: const TextStyle(
          color: Colors.white,
          fontSize: 13,
          fontWeight: FontWeight.w800,
        ),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final cs = theme.colorScheme;

    return Scaffold(
      backgroundColor: cs.surface,
      appBar: AppBar(
        backgroundColor: cs.surface,
        foregroundColor: cs.onSurface,
        title: Text(
          widget.destName,
          maxLines: 1,
          overflow: TextOverflow.ellipsis,
          style: theme.textTheme.titleMedium?.copyWith(
            fontWeight: FontWeight.w800,
            fontSize: 22,
          ),
        ),
        actions: [
          IconButton(
            icon: const Icon(Icons.close_rounded, size: 32),
            tooltip: '닫기',
            onPressed: () => Navigator.of(context).pop(),
          ),
          const SizedBox(width: 8),
        ],
      ),
      body: SafeArea(
        child: Padding(
          padding: const EdgeInsets.fromLTRB(8, 4, 8, 8),
          child: Column(
            children: [
              _buildRouteChips(theme, _viewport.transitStops),
              const SizedBox(height: 8),
              Expanded(
                child: ClipRRect(
                  borderRadius: BorderRadius.circular(16),
                  child: DemoMode.enabled
                      ? _buildDemoMap(theme)
                      : NaverMap(
                          options: NaverMapViewOptions(
                            initialCameraPosition: NCameraPosition(
                              target: NLatLng(
                                _viewport.centerLat,
                                _viewport.centerLng,
                              ),
                              zoom: (_viewport.level + 1).toDouble(),
                            ),
                            // 어르신 가시성: zoom 컨트롤 켜고, 잡요소(축척/실내) 끔
                            mapType: NMapType.basic,
                            activeLayerGroups: const [
                              NLayerGroup.building,
                              NLayerGroup.transit,
                            ],
                            logoAlign: NLogoAlign.leftBottom,
                            scaleBarEnable: false,
                            indoorEnable: false,
                            rotationGesturesEnable: false,
                            tiltGesturesEnable: false,
                          ),
                          onMapReady: _onMapReady,
                        ),
                ),
              ),
              const SizedBox(height: 8),
              SizedBox(
                width: double.infinity,
                // titleLarge fontSize 32 + 위아래 padding 고려해 72dp. 56dp 면
                // 글자 ascender/descender 가 잘림.
                height: 72,
                child: FilledButton(
                  style: FilledButton.styleFrom(
                    backgroundColor: cs.primary,
                    foregroundColor: cs.onPrimary,
                    padding: const EdgeInsets.symmetric(horizontal: 24),
                    shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(20),
                    ),
                  ),
                  onPressed: () => Navigator.of(context).pop(),
                  child: Text(
                    '닫기',
                    style: theme.textTheme.titleLarge?.copyWith(
                      color: cs.onPrimary,
                      fontWeight: FontWeight.w800,
                      height: 1.0,
                    ),
                  ),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}
