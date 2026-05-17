import 'dart:typed_data';

import 'package:flutter/material.dart';

/// 심사위원용 데모 — 카메라 버튼 첫 탭 시 표시되는 전체화면 사진 preview.
///
/// 실 카메라 화면처럼 ① 사진 풀스크린 ② 하단 원형 카메라 버튼 ③ 상단 안내 캡션.
/// 하단 카메라 버튼 탭 → `Navigator.pop` 으로 photoBytes 반환 → 호출측이 그대로
/// `GemmaService.generateGuidance` 에 넘긴다.
class DemoPhotoPreviewScreen extends StatelessWidget {
  const DemoPhotoPreviewScreen({
    super.key,
    required this.photoBytes,
    required this.caption,
  });

  final Uint8List photoBytes;
  final String caption;

  @override
  Widget build(BuildContext context) {
    final cs = Theme.of(context).colorScheme;
    return Scaffold(
      backgroundColor: Colors.black,
      body: SafeArea(
        child: Column(
          children: [
            Padding(
              padding: const EdgeInsets.fromLTRB(16, 16, 16, 8),
              child: Text(
                caption,
                textAlign: TextAlign.center,
                style: const TextStyle(
                  color: Colors.white,
                  fontSize: 16,
                  fontWeight: FontWeight.w500,
                  height: 1.35,
                ),
              ),
            ),
            Expanded(
              child: Center(
                child: Image.memory(
                  photoBytes,
                  fit: BoxFit.contain,
                ),
              ),
            ),
            Padding(
              padding: const EdgeInsets.only(top: 16, bottom: 36),
              child: Material(
                color: Colors.transparent,
                child: InkWell(
                  onTap: () => Navigator.pop(context, photoBytes),
                  customBorder: const CircleBorder(),
                  child: Container(
                    width: 88,
                    height: 88,
                    decoration: BoxDecoration(
                      shape: BoxShape.circle,
                      color: cs.tertiary,
                      boxShadow: const [
                        BoxShadow(
                          color: Colors.black54,
                          blurRadius: 14,
                          offset: Offset(0, 4),
                        ),
                      ],
                    ),
                    child: Icon(
                      Icons.photo_camera_rounded,
                      size: 46,
                      color: cs.onTertiary,
                    ),
                  ),
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}
