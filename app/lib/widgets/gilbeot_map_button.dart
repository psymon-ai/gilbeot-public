import 'package:flutter/material.dart';

/// 경로 미리보기(정적 지도) 트리거 버튼.
///
/// 마이크 버튼과 같은 행에 좌우 2등분으로 배치되는 시나리오를 가정해 [compact]
/// 모드를 기본값으로 둠. 본문 정렬·아이콘·라벨 비율은 [GilbeotMicButton] 의
/// compact variant 와 시각적으로 동일하게 맞춤(어르신이 "두 개의 큰 버튼" 으로
/// 인지하도록).
class GilbeotMapButton extends StatelessWidget {
  const GilbeotMapButton({
    super.key,
    required this.onPressed,
    this.enabled = true,
    this.compact = true,
    this.label = '지도\n보기',
  });

  final VoidCallback? onPressed;
  final bool enabled;
  final String label;

  /// `true` 이면 좌우 분할 시 사용하는 작은 사이즈(아이콘 40, 라벨 24).
  /// `false` 이면 full-width 단독 배치용 (마이크 버튼 단독 모드와 동일 비율).
  final bool compact;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final cs = theme.colorScheme;

    const double height = 110;
    final double iconSize = compact ? 40 : 52;
    final double fontSize = compact ? 24 : 32;
    const double radius = 28;

    // primary (teal-800) 톤. 마이크가 amber 라 시각적으로 색상 충돌 없음.
    final bg = enabled ? cs.primary : cs.surfaceContainerHighest;
    final fg = enabled ? cs.onPrimary : cs.onSurfaceVariant;

    return GestureDetector(
      onTap: enabled ? onPressed : null,
      child: AnimatedContainer(
        duration: const Duration(milliseconds: 160),
        curve: Curves.easeOut,
        width: double.infinity,
        height: height,
        decoration: BoxDecoration(
          color: bg,
          borderRadius: BorderRadius.circular(radius),
          boxShadow: enabled
              ? [
                  BoxShadow(
                    color: bg.withAlpha(90),
                    blurRadius: 16,
                    offset: const Offset(0, 6),
                    spreadRadius: 2,
                  ),
                  const BoxShadow(
                    color: Colors.black26,
                    blurRadius: 8,
                    offset: Offset(0, 3),
                  ),
                ]
              : null,
        ),
        child: Row(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Icon(Icons.map_rounded, size: iconSize, color: fg),
            const SizedBox(width: 12),
            Flexible(
              child: FittedBox(
                fit: BoxFit.scaleDown,
                child: Text(
                  label,
                  textAlign: TextAlign.center,
                  style: theme.textTheme.labelLarge?.copyWith(
                    color: fg,
                    fontSize: fontSize,
                    fontWeight: FontWeight.w800,
                    height: 1.1,
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
