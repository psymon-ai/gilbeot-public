import 'package:flutter/material.dart';

/// Last-resort help banner shown when the camera-based guidance pipeline has
/// failed to extract usable features for the same step several times in a row
/// (default: 3 consecutive NO_FEATURE results).
///
/// Surfaces a clear "ask someone nearby" message + a single dismiss button.
/// The intent is to break the user out of an infinite retry loop without
/// silently advancing the route step (which would risk wrong directions).
class GilbeotHelpBanner extends StatelessWidget {
  const GilbeotHelpBanner({
    super.key,
    required this.message,
    required this.onDismiss,
  });

  /// One-sentence message read aloud and shown on the banner.
  /// Example: "주변 분께 잠실역 위치를 여쭤보세요."
  final String message;

  /// Called when the user taps the "확인" button.
  /// The host screen should reset the consecutive-failure counter so the next
  /// photo gets a fresh retry budget.
  final VoidCallback onDismiss;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final cs = theme.colorScheme;

    return Container(
      width: double.infinity,
      decoration: BoxDecoration(
        color: cs.tertiaryContainer,
        borderRadius: BorderRadius.circular(20),
        border: Border.all(color: cs.tertiary.withAlpha(120), width: 1.5),
        boxShadow: [
          BoxShadow(
            color: cs.shadow.withAlpha(20),
            blurRadius: 12,
            offset: const Offset(0, 4),
          ),
        ],
      ),
      child: ClipRRect(
        borderRadius: BorderRadius.circular(20),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            // Top accent stripe — warm tertiary to read as "help" not error.
            Container(
              height: 4,
              decoration: BoxDecoration(
                gradient: LinearGradient(
                  colors: [cs.tertiary, cs.tertiaryContainer],
                ),
              ),
            ),
            Padding(
              padding: const EdgeInsets.fromLTRB(20, 16, 20, 16),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.stretch,
                children: [
                  Row(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Icon(
                        Icons.people_alt_rounded,
                        size: 36,
                        color: cs.tertiary,
                      ),
                      const SizedBox(width: 12),
                      Expanded(
                        child: Text(
                          message,
                          style: theme.textTheme.bodyLarge?.copyWith(
                            color: cs.onTertiaryContainer,
                            height: 1.4,
                            fontWeight: FontWeight.w500,
                          ),
                        ),
                      ),
                    ],
                  ),
                  const SizedBox(height: 12),
                  Align(
                    alignment: Alignment.centerRight,
                    child: FilledButton.tonal(
                      onPressed: onDismiss,
                      style: FilledButton.styleFrom(
                        minimumSize: const Size(120, 48),
                        shape: RoundedRectangleBorder(
                          borderRadius: BorderRadius.circular(16),
                        ),
                      ),
                      child: const Text('확인'),
                    ),
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }
}
