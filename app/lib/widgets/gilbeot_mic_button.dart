import 'package:flutter/material.dart';

/// The primary voice-trigger CTA for [HomeScreen].
///
/// 110dp tall (≥ 56dp hard requirement, hero element so larger).
/// Amber when idle, red when recording — both colours meet ≥ 4.5:1 against
/// white icon/text. Subtle scale animation on press (≤ 200ms).
class GilbeotMicButton extends StatefulWidget {
  const GilbeotMicButton({
    super.key,
    required this.onTap,
    required this.onLongPressStart,
    required this.onLongPressEnd,
    required this.isRecording,
    required this.isBusy,
    this.idleLabel = '말하기',
    this.recordingLabel = '끝내기',
    this.busyLabel = '처리 중...',
    this.compact = false,
  });

  final VoidCallback onTap;
  final VoidCallback onLongPressStart;
  final VoidCallback onLongPressEnd;
  final bool isRecording;
  final bool isBusy;

  /// 정적 상태(녹음/처리 중 아님)에서 표시할 라벨.
  /// 첫 진입 시 "말하기" / "Speak", 경로 로드 후 "다시 말하기" / "Speak again".
  final String idleLabel;

  /// 녹음 중 상태 라벨 (기본 "끝내기" / 영문 "Stop").
  final String recordingLabel;

  /// busy 상태 라벨 (기본 "처리 중..." / 영문 "Processing...").
  final String busyLabel;

  /// `true` 이면 좌우 2등분 배치용 축소 변형 (아이콘 40, 라벨 24).
  /// 좌우 두 버튼 사이의 시각 균형을 위해 GilbeotMapButton 의 compact 와 동일 비율.
  final bool compact;

  @override
  State<GilbeotMicButton> createState() => _GilbeotMicButtonState();
}

class _GilbeotMicButtonState extends State<GilbeotMicButton>
    with SingleTickerProviderStateMixin {
  late final AnimationController _scale;
  late final Animation<double> _scaleAnim;

  @override
  void initState() {
    super.initState();
    _scale = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 120),
      reverseDuration: const Duration(milliseconds: 160),
      lowerBound: 0.96,
      upperBound: 1.0,
      value: 1.0,
    );
    _scaleAnim = _scale;
  }

  @override
  void dispose() {
    _scale.dispose();
    super.dispose();
  }

  void _onTapDown(TapDownDetails _) => _scale.reverse();
  void _onTapUp(TapUpDetails _) => _scale.forward();
  void _onTapCancel() => _scale.forward();

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final cs = theme.colorScheme;

    // Idle: warm amber. Recording: signal red. Busy: muted surface.
    final Color bgColor;
    final Color fgColor;
    final String label;
    final IconData icon;

    if (widget.isBusy && !widget.isRecording) {
      bgColor = cs.surfaceContainerHighest;
      fgColor = cs.onSurfaceVariant;
      label = widget.busyLabel;
      icon = Icons.hourglass_top_rounded;
    } else if (widget.isRecording) {
      bgColor = const Color(0xFFD32F2F); // red-700 — ≥ 4.5:1 on white
      fgColor = Colors.white;
      label = widget.recordingLabel;
      icon = Icons.stop_circle_rounded;
    } else {
      bgColor = cs.tertiary; // amber-700
      fgColor = cs.onTertiary; // black — high contrast on amber
      label = widget.idleLabel;
      icon = Icons.mic_rounded;
    }

    // Compact 모드(좌우 2등분 배치) 에서는 아이콘/라벨 축소 + FittedBox 로
    // overflow 자동 회피. 단독 모드는 기존 비율 유지(어르신 초기 시야 hero).
    final double iconSize = widget.compact ? 40 : 52;
    final double fontSize = widget.compact ? 24 : 32;

    return ScaleTransition(
      scale: _scaleAnim,
      child: GestureDetector(
        onTap: widget.onTap,
        onTapDown: _onTapDown,
        onTapUp: _onTapUp,
        onTapCancel: _onTapCancel,
        onLongPressStart: (_) => widget.onLongPressStart(),
        onLongPressEnd: (_) => widget.onLongPressEnd(),
        child: AnimatedContainer(
          duration: const Duration(milliseconds: 160),
          curve: Curves.easeOut,
          width: double.infinity,
          height: 110,
          decoration: BoxDecoration(
            color: bgColor,
            borderRadius: BorderRadius.circular(28),
            boxShadow: [
              BoxShadow(
                color: bgColor.withAlpha(100),
                blurRadius: 16,
                offset: const Offset(0, 6),
                spreadRadius: 2,
              ),
              const BoxShadow(
                color: Colors.black26,
                blurRadius: 8,
                offset: Offset(0, 3),
              ),
            ],
          ),
          child: Row(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              AnimatedSwitcher(
                duration: const Duration(milliseconds: 150),
                child: Icon(
                  icon,
                  key: ValueKey(icon),
                  size: iconSize,
                  color: fgColor,
                ),
              ),
              SizedBox(width: widget.compact ? 12 : 16),
              Flexible(
                child: FittedBox(
                  fit: BoxFit.scaleDown,
                  child: AnimatedSwitcher(
                    duration: const Duration(milliseconds: 150),
                    child: Text(
                      label,
                      key: ValueKey(label),
                      textAlign: TextAlign.center,
                      style: theme.textTheme.labelLarge?.copyWith(
                        color: fgColor,
                        fontSize: fontSize,
                        fontWeight: FontWeight.w800,
                        height: 1.1,
                      ),
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
