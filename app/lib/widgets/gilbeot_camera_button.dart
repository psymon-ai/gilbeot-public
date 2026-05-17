import 'package:flutter/material.dart';

/// Secondary action button for triggering the camera on [HomeScreen].
///
/// Disabled when no route is loaded or the app is busy.
/// 64dp height (≥ 56dp requirement). Uses FilledButton.tonal for visual
/// hierarchy — prominent but secondary to the mic CTA.
class GilbeotCameraButton extends StatelessWidget {
  const GilbeotCameraButton({
    super.key,
    required this.onPressed,
    this.enabled = true,
    this.label = '사진 찍기',
    this.onLongPress,
  });

  final VoidCallback? onPressed;
  final bool enabled;
  final String label;

  /// Optional long-press handler. DEMO_MODE uses this to let judges pick a
  /// photo from their gallery (BYO photo) — proves the on-device model
  /// isn't tuned to the 4 bundled demo photos.
  final VoidCallback? onLongPress;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final cs = theme.colorScheme;

    return SizedBox(
      width: double.infinity,
      height: 72,
      child: GestureDetector(
        onLongPress: (enabled && onLongPress != null) ? onLongPress : null,
        child: FilledButton.tonal(
          onPressed: enabled ? onPressed : null,
          style: FilledButton.styleFrom(
          backgroundColor: enabled
              ? cs.secondaryContainer
              : cs.surfaceContainerHighest,
          foregroundColor: enabled
              ? cs.onSecondaryContainer
              : cs.onSurfaceVariant,
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(20),
          ),
          padding: const EdgeInsets.symmetric(horizontal: 24),
        ),
        child: Row(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Icon(
              Icons.photo_camera_rounded,
              size: 40,
              color: enabled ? cs.onSecondaryContainer : cs.onSurfaceVariant,
            ),
            const SizedBox(width: 16),
            Text(
              label,
              style: theme.textTheme.labelLarge?.copyWith(
                color: enabled ? cs.onSecondaryContainer : cs.onSurfaceVariant,
              ),
            ),
          ],
        ),
        ),
      ),
    );
  }
}
