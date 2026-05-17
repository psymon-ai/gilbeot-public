import 'dart:io';

import 'package:flutter/foundation.dart';
import 'package:flutter_gemma/pigeon.g.dart';

/// 디바이스 메모리/성능 프로파일.
///
/// Gemma 4 E2B vision 모델은 디바이스에 따라 최적 설정이 크게 다름:
///   - low   (≤ 6 GB RAM): CPU backend, image 768px,  maxTokens 2048 — S10e (5.5GB Mali-G76)
///   - mid   (6~6.5 GB):   CPU backend, image 1280px, maxTokens 2048
///   - high  (≥ 6.5 GB):   GPU backend, image 1280px, maxTokens 2048 — S23 (8GB Adreno 740)
///
/// high 의 image 1280px (Follow-up 102): prefill 단축용으로 768px 까지 내려봤으나
/// (ⓐ 출구번호·화살표 같은 작은 표지판 글자가 다운샘플로 뭉개져 안내문 grounding
/// 붕괴, ⓑ logcat 실측상 768/896/1280 prefill 차이가 미미해 속도 이득도 없음)
/// 검증된 1280px 로 복원. 속도는 decode(~20chars/s)·prefill(~5~6s) 하드 한계.
///
/// GPU backend 는 OpenCL 커널 컴파일 + GPU weight 변환에 ~1 GB 추가 — 8 GB 미만 폰에서
/// lmkd OOM. CPU 는 XNNPACK 으로 동작하며 추론 약 3 배 느리지만 메모리 안정적.
enum DeviceProfile { low, mid, high }

class DeviceCaps {
  DeviceCaps._({
    required this.memTotalGb,
    required this.profile,
    required this.preferredBackend,
    required this.imageTargetWidth,
    required this.maxTokens,
  });

  final double memTotalGb;
  final DeviceProfile profile;
  final PreferredBackend preferredBackend;

  /// vision LLM 호출 전에 image 다운샘플 폭. patch 수 + RAM 영향.
  final int imageTargetWidth;

  /// LiteRT-LM KV cache 크기 결정.
  final int maxTokens;

  static DeviceCaps? _cached;
  static Future<DeviceCaps> probe() async {
    if (_cached != null) return _cached!;

    double memGb = 0;
    if (Platform.isAndroid || Platform.isLinux) {
      try {
        final txt = await File('/proc/meminfo').readAsString();
        final m = RegExp(r'MemTotal:\s+(\d+)\s*kB').firstMatch(txt);
        if (m != null) {
          memGb = int.parse(m.group(1)!) / (1024 * 1024);
        }
      } catch (e) {
        debugPrint('[DeviceCaps] /proc/meminfo read failed: $e');
      }
    }

    DeviceProfile profile;
    PreferredBackend backend;
    int targetWidth;
    int maxTokens;

    if (memGb < 6.0) {
      profile = DeviceProfile.low;
      backend = PreferredBackend.cpu;
      // 768 — vision prefill 의 대부분이 image patches. 1024 면 S10e CPU 에서
      // prefill 23초+. 768 로 줄이면 patches ~55% 로 감소 → prefill 크게 단축.
      // 어르신용 안내는 간판/색/큰 사물 인식이라 768 해상도로 충분.
      targetWidth = 768;
      maxTokens = 2048;
    } else if (memGb < 6.5) {
      // 6.0~6.5GB 는 GPU delegate 의 ~1GB 추가 메모리 부담이 risky.
      profile = DeviceProfile.mid;
      backend = PreferredBackend.cpu;
      targetWidth = 1280;
      maxTokens = 2048;
    } else {
      // ≥ 6.5GB — S23 spec 8GB 인데 시스템 점유 후 가용 6.9GB. high 분기
      // 진입해 GPU 사용 (Adreno 740 + OpenCL delegate).
      profile = DeviceProfile.high;
      backend = PreferredBackend.gpu;
      targetWidth = 1280;
      maxTokens = 2048;
    }

    _cached = DeviceCaps._(
      memTotalGb: memGb,
      profile: profile,
      preferredBackend: backend,
      imageTargetWidth: targetWidth,
      maxTokens: maxTokens,
    );
    debugPrint('[DeviceCaps] memTotal=${memGb.toStringAsFixed(1)}GB '
        'profile=${profile.name} backend=${backend.name} '
        'imgWidth=$targetWidth maxTokens=$maxTokens');
    return _cached!;
  }
}
