import 'dart:io';

import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

/// 어르신용 단순 카메라 화면 — 큰 셔터 버튼 1개.
/// 캡처 시 photo의 Uint8List를 pop으로 반환.
///
/// Design decisions:
/// - Full-bleed camera preview, no competing chrome.
/// - Top instruction banner: translucent dark scrim, respects status bar safe area.
/// - Bottom control strip: dark frosted panel with generous padding, respects nav bar.
/// - Shutter button 110dp — matches the mic CTA footprint.
/// - Close button 56dp tap target in top-left, inside the safe area.
/// - Loading state: centred spinner on black with descriptive label.
/// - Capture state: shutter fades to a muted grey and shows a spinner so the
///   user knows a photo is being processed.
class CameraScreen extends StatefulWidget {
  const CameraScreen({super.key});

  @override
  State<CameraScreen> createState() => _CameraScreenState();
}

class _CameraScreenState extends State<CameraScreen>
    with WidgetsBindingObserver {
  CameraController? _controller;
  bool _busy = false;
  String? _errorMessage;

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
    // Full-screen immersive while camera is active.
    SystemChrome.setEnabledSystemUIMode(SystemUiMode.immersiveSticky);
    _setup();
  }

  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
    // Restore normal system UI when leaving the camera.
    SystemChrome.setEnabledSystemUIMode(SystemUiMode.edgeToEdge);
    _controller?.dispose();
    super.dispose();
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    final ctrl = _controller;
    if (ctrl == null || !ctrl.value.isInitialized) return;
    if (state == AppLifecycleState.inactive) {
      ctrl.dispose();
    } else if (state == AppLifecycleState.resumed) {
      _setup();
    }
  }

  Future<void> _setup() async {
    setState(() {
      _errorMessage = null;
    });
    try {
      final cameras = await availableCameras();
      if (cameras.isEmpty) {
        if (mounted) {
          setState(() => _errorMessage = '카메라를 찾을 수 없어요.');
        }
        return;
      }
      final back = cameras.firstWhere(
        (c) => c.lensDirection == CameraLensDirection.back,
        orElse: () => cameras.first,
      );
      final c = CameraController(
        back,
        ResolutionPreset
            .medium, // 1280x720 정도 — Gemma vision encoder는 medium으로 충분
        enableAudio: false,
        imageFormatGroup: ImageFormatGroup.jpeg,
      );
      await c.initialize();
      if (!mounted) return;
      setState(() => _controller = c);
    } catch (e) {
      if (mounted) {
        setState(() => _errorMessage = '카메라 초기화 실패: $e');
      }
    }
  }

  Future<void> _capture() async {
    if (_busy || _controller == null) return;
    setState(() => _busy = true);
    try {
      final shot = await _controller!.takePicture();
      final raw = await File(shot.path).readAsBytes();
      // takePicture() on the S10e returns a fixed ~773KB buffer for most
      // captures: a valid JPEG at the front, then stale bytes from previous
      // frames, never truncated to the real length. Trim to the real EOI so
      // the vision encoder and session log only ever see the actual photo.
      final bytes = _trimToJpegEnd(raw);
      if (bytes.length != raw.length) {
        debugPrint(
          '[camera] trimmed JPEG buffer ${raw.length} -> '
          '${bytes.length} bytes (stale tail dropped)',
        );
      }
      if (!mounted) return;
      Navigator.pop(context, bytes);
    } catch (e) {
      if (!mounted) return;
      setState(() => _busy = false);
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text('촬영 실패: $e', style: const TextStyle(fontSize: 18)),
        ),
      );
    }
  }

  @override
  Widget build(BuildContext context) {
    // Error state — camera unavailable.
    if (_errorMessage != null) {
      return _ErrorView(
        message: _errorMessage!,
        onRetry: _setup,
        onClose: () => Navigator.pop(context, null),
      );
    }

    // Loading state — camera initialising.
    if (_controller == null || !_controller!.value.isInitialized) {
      return const _LoadingView();
    }

    // Active camera view.
    return Scaffold(
      backgroundColor: Colors.black,
      body: Stack(
        fit: StackFit.expand,
        children: [
          // ── Full-bleed camera preview ─────────────────────────────────────
          CameraPreview(_controller!),

          // ── Top instruction banner ────────────────────────────────────────
          Positioned(
            top: 0,
            left: 0,
            right: 0,
            child: _InstructionBanner(busy: _busy),
          ),

          // ── Bottom control strip ──────────────────────────────────────────
          Positioned(
            bottom: 0,
            left: 0,
            right: 0,
            child: _BottomControls(
              busy: _busy,
              onCapture: _capture,
              onClose: () => Navigator.pop(context, null),
            ),
          ),
        ],
      ),
    );
  }
}

// ---------------------------------------------------------------------------
// Sub-widgets — private to this file
// ---------------------------------------------------------------------------

/// Translucent instruction banner at the top of the camera view.
/// Respects status bar inset via SafeArea.
class _InstructionBanner extends StatelessWidget {
  const _InstructionBanner({required this.busy});

  final bool busy;

  @override
  Widget build(BuildContext context) {
    return SafeArea(
      bottom: false,
      child: Container(
        margin: const EdgeInsets.fromLTRB(16, 12, 16, 0),
        padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 14),
        decoration: BoxDecoration(
          color: Colors.black.withAlpha(153), // 60% opacity scrim
          borderRadius: BorderRadius.circular(16),
        ),
        child: Text(
          busy ? '사진 처리 중...' : '주변을 한 장 찍어주세요',
          textAlign: TextAlign.center,
          style: const TextStyle(
            color: Colors.white,
            fontSize: 26,
            fontWeight: FontWeight.w700,
            height: 1.3,
            // Letterpress shadow for legibility over any background.
            shadows: [Shadow(color: Colors.black87, blurRadius: 6)],
          ),
        ),
      ),
    );
  }
}

/// Dark frosted bottom strip containing the shutter and close buttons.
/// Respects nav bar inset via SafeArea.
class _BottomControls extends StatelessWidget {
  const _BottomControls({
    required this.busy,
    required this.onCapture,
    required this.onClose,
  });

  final bool busy;
  final VoidCallback onCapture;
  final VoidCallback onClose;

  @override
  Widget build(BuildContext context) {
    return Container(
      decoration: const BoxDecoration(
        gradient: LinearGradient(
          begin: Alignment.bottomCenter,
          end: Alignment.topCenter,
          colors: [Colors.black87, Colors.transparent],
          stops: [0.55, 1.0],
        ),
      ),
      child: SafeArea(
        top: false,
        child: Padding(
          padding: const EdgeInsets.fromLTRB(24, 20, 24, 32),
          child: Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            crossAxisAlignment: CrossAxisAlignment.center,
            children: [
              // Close button — 56dp tap target
              _CloseButton(onClose: onClose),

              // Shutter button — 110dp hero
              _ShutterButton(busy: busy, onCapture: onCapture),

              // Spacer to balance the close button width
              const SizedBox(width: 56),
            ],
          ),
        ),
      ),
    );
  }
}

class _ShutterButton extends StatelessWidget {
  const _ShutterButton({required this.busy, required this.onCapture});

  final bool busy;
  final VoidCallback onCapture;

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: onCapture,
      child: AnimatedContainer(
        duration: const Duration(milliseconds: 160),
        width: 110,
        height: 110,
        decoration: BoxDecoration(
          shape: BoxShape.circle,
          color: busy ? Colors.grey.shade600 : Colors.white,
          border: Border.all(
            color: busy ? Colors.grey.shade400 : Colors.white,
            width: 5,
          ),
          boxShadow: const [
            BoxShadow(color: Colors.black54, blurRadius: 16, spreadRadius: 2),
          ],
        ),
        child: busy
            ? const Padding(
                padding: EdgeInsets.all(28),
                child: CircularProgressIndicator(
                  strokeWidth: 3,
                  color: Colors.white,
                ),
              )
            : const Icon(
                Icons.camera_alt_rounded,
                size: 56,
                color: Colors.black87,
              ),
      ),
    );
  }
}

class _CloseButton extends StatelessWidget {
  const _CloseButton({required this.onClose});

  final VoidCallback onClose;

  @override
  Widget build(BuildContext context) {
    return SizedBox(
      width: 56,
      height: 56,
      child: Material(
        color: Colors.black45,
        shape: const CircleBorder(),
        child: InkWell(
          customBorder: const CircleBorder(),
          onTap: onClose,
          child: const Icon(Icons.close_rounded, color: Colors.white, size: 32),
        ),
      ),
    );
  }
}

/// Full-screen loading view while camera initialises.
class _LoadingView extends StatelessWidget {
  const _LoadingView();

  @override
  Widget build(BuildContext context) {
    return const Scaffold(
      backgroundColor: Colors.black,
      body: Center(
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            CircularProgressIndicator(color: Colors.white, strokeWidth: 3),
            SizedBox(height: 24),
            Text(
              '카메라 준비 중...',
              style: TextStyle(
                color: Colors.white,
                fontSize: 22,
                fontWeight: FontWeight.w600,
              ),
            ),
          ],
        ),
      ),
    );
  }
}

/// Full-screen error view when camera is unavailable.
class _ErrorView extends StatelessWidget {
  const _ErrorView({
    required this.message,
    required this.onRetry,
    required this.onClose,
  });

  final String message;
  final VoidCallback onRetry;
  final VoidCallback onClose;

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.black,
      body: SafeArea(
        child: Padding(
          padding: const EdgeInsets.all(32),
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              const Icon(
                Icons.no_photography_rounded,
                color: Colors.white54,
                size: 72,
              ),
              const SizedBox(height: 24),
              Text(
                message,
                textAlign: TextAlign.center,
                style: const TextStyle(
                  color: Colors.white,
                  fontSize: 22,
                  fontWeight: FontWeight.w600,
                  height: 1.5,
                ),
              ),
              const SizedBox(height: 40),
              SizedBox(
                width: double.infinity,
                height: 64,
                child: FilledButton.icon(
                  onPressed: onRetry,
                  icon: const Icon(Icons.refresh_rounded, size: 28),
                  label: const Text(
                    '다시 시도',
                    style: TextStyle(fontSize: 22, fontWeight: FontWeight.w700),
                  ),
                  style: FilledButton.styleFrom(
                    backgroundColor: Colors.white,
                    foregroundColor: Colors.black87,
                    shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(20),
                    ),
                  ),
                ),
              ),
              const SizedBox(height: 16),
              SizedBox(
                width: double.infinity,
                height: 64,
                child: OutlinedButton.icon(
                  onPressed: onClose,
                  icon: const Icon(Icons.arrow_back_rounded, size: 28),
                  label: const Text(
                    '돌아가기',
                    style: TextStyle(fontSize: 22, fontWeight: FontWeight.w700),
                  ),
                  style: OutlinedButton.styleFrom(
                    foregroundColor: Colors.white,
                    side: const BorderSide(color: Colors.white54, width: 1.5),
                    shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(20),
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

// ---------------------------------------------------------------------------
// JPEG buffer trimming
// ---------------------------------------------------------------------------

/// Returns [b] trimmed to exactly [SOI .. EOI].
///
/// `takePicture()` on the S10e hands back a fixed-size (~773KB) buffer for most
/// captures: a complete, valid JPEG at the front, followed by hundreds of KB of
/// stale bytes from earlier frames, never truncated to the real JPEG length.
/// That bloated buffer would otherwise flow into the vision encoder and the
/// session log.
///
/// This walks the JPEG marker structure from SOI past the APP segments (the
/// EXIF thumbnail lives inside APP1, so it is skipped wholesale by length) to
/// the Start-of-Scan, then scans the entropy stream for the terminating EOI.
/// If the bytes do not parse as a JPEG we recognise, the original buffer is
/// returned unchanged — trimming must never corrupt a good capture.
Uint8List _trimToJpegEnd(Uint8List b) {
  final n = b.length;
  if (n < 4 || b[0] != 0xFF || b[1] != 0xD8) return b; // not a JPEG we know
  if (b[n - 2] == 0xFF && b[n - 1] == 0xD9) return b; // already clean

  var i = 2;
  while (i + 1 < n) {
    if (b[i] != 0xFF) return b; // expected a marker here — bail safely
    while (i < n && b[i] == 0xFF) {
      i++; // skip fill bytes
    }
    if (i >= n) return b;
    final marker = b[i];
    i++;
    if (marker == 0xD9) return b.sublist(0, i); // stray top-level EOI
    if (marker == 0xDA) {
      // Start of Scan: skip its header, then scan entropy data for the EOI.
      if (i + 1 >= n) return b;
      i += (b[i] << 8) | b[i + 1];
      while (i + 1 < n) {
        if (b[i] != 0xFF) {
          i++;
          continue;
        }
        final m = b[i + 1];
        if (m == 0xD9) return b.sublist(0, i + 2);
        // 0x00 = stuffed 0xFF, 0xD0..0xD7 = restart markers: part of the
        // stream. Anything else: skip and keep scanning.
        i += 2;
      }
      return b; // no EOI found — leave the capture untouched
    }
    // Any other segment (APPn, DQT, DHT, SOFn, COM, ...) carries a length.
    if (i + 1 >= n) return b;
    i += (b[i] << 8) | b[i + 1];
  }
  return b;
}
