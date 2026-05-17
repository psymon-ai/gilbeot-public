import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';

/// Android `ActivityManager.MemoryInfo` 의 Dart 표현. native channel
/// `gilbeot/memory` → `getMemoryInfo` 호출 결과를 wrapping.
class SystemMemoryInfo {
  const SystemMemoryInfo({
    required this.availBytes,
    required this.totalBytes,
    required this.thresholdBytes,
    required this.lowMemory,
  });

  final int availBytes;
  final int totalBytes;
  final int thresholdBytes;
  final bool lowMemory;

  int get availMB => availBytes ~/ (1024 * 1024);
  int get totalMB => totalBytes ~/ (1024 * 1024);
  int get thresholdMB => thresholdBytes ~/ (1024 * 1024);

  /// `availBytes < requiredBytes` 면 부족.
  ///
  /// **임계 = 400MB** (FU109 조정). Galaxy S23 / One UI 6+ 디바이스는
  /// baseline 으로 system + Samsung services + Galaxy AI 가 ~6GB 잡고 시작해
  /// 평상시 user-available 가 800MB 정도밖에 안 된다. 800MB 임계 잡으면
  /// "다른 앱 안 깔려 있어도 항상 차단" 의 false positive. 400MB 미만일 때만
  /// 진짜 critical 로 보고 차단.
  bool isInsufficient({int requiredBytes = 400 * 1024 * 1024}) =>
      availBytes < requiredBytes || lowMemory;

  @override
  String toString() =>
      'SystemMemoryInfo(avail=${availMB}MB / total=${totalMB}MB, '
      'threshold=${thresholdMB}MB, lowMemory=$lowMemory)';
}

/// 시스템 메모리 상태 query 및 onTrimMemory 콜백 수신.
///
/// **왜 필요한가**: Gemma 4 E2B 가 ~2.4GB GPU memory + Dart/Flutter overhead +
/// background app churn 으로 S23 (8GB RAM) 의 marginal 한계에 자주 도달. 우리
/// 앱이 silent OOM kill 되면 시연이 깨지고 judge 가 불안해한다. BYO 갤러리
/// 진입 / 카메라 분석 진입 같은 메모리 peak 직전에 시스템 상태를 query 해
/// 위험하면 사용자에게 미리 경고 → graceful retry path.
class SystemMemory {
  SystemMemory._();

  static const _channel = MethodChannel('gilbeot/memory');
  static int? _lastTrimLevel;
  static DateTime? _lastTrimAt;
  static bool _listenerAttached = false;

  /// 시스템 메모리 상태 조회. native channel 실패 시 null.
  static Future<SystemMemoryInfo?> query() async {
    try {
      final raw = await _channel.invokeMapMethod<String, dynamic>('getMemoryInfo');
      if (raw == null) return null;
      return SystemMemoryInfo(
        availBytes: (raw['availMemBytes'] as num).toInt(),
        totalBytes: (raw['totalMemBytes'] as num).toInt(),
        thresholdBytes: (raw['thresholdBytes'] as num).toInt(),
        lowMemory: raw['lowMemory'] == true,
      );
    } catch (e) {
      debugPrint('[memory] query failed: $e');
      return null;
    }
  }

  /// MainActivity.onTrimMemory 콜백 수신 등록. 한 번만 호출.
  /// 콜백은 native side 에서 `invokeMethod('onTrimMemory', level)` 로 들어옴.
  /// level 은 Android `ComponentCallbacks2.TRIM_MEMORY_*` 상수값.
  static void attachTrimListener() {
    if (_listenerAttached) return;
    _listenerAttached = true;
    _channel.setMethodCallHandler((call) async {
      if (call.method == 'onTrimMemory') {
        _lastTrimLevel = (call.arguments as num?)?.toInt();
        _lastTrimAt = DateTime.now();
        debugPrint('[memory] onTrimMemory level=$_lastTrimLevel');
      }
    });
  }

  /// 최근 (지난 30초) trim 콜백 받았는지 — 시스템 메모리 압박 신호.
  static bool get recentlyTrimmed {
    final t = _lastTrimAt;
    if (t == null) return false;
    return DateTime.now().difference(t).inSeconds < 30;
  }

  static int? get lastTrimLevel => _lastTrimLevel;
}
