import 'package:flutter/material.dart';

/// The hero status card shown on [HomeScreen].
///
/// Displays the current navigation guidance or status message.
/// Scrollable for long Korean guidance sentences.
/// Shows a subtle loading indicator when [isLoading] is true.
class GilbeotStatusCard extends StatelessWidget {
  const GilbeotStatusCard({
    super.key,
    required this.message,
    this.isLoading = false,
    this.isError = false,
    this.isArrived = false,
  });

  final String message;
  final bool isLoading;
  final bool isError;
  final bool isArrived;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final cs = theme.colorScheme;

    // Pick surface colour based on state.
    final Color cardColor;
    final Color textColor;
    final Color? borderColor;

    if (isArrived) {
      cardColor = cs.primaryContainer;
      textColor = cs.onPrimaryContainer;
      borderColor = cs.primary.withAlpha(102);
    } else if (isError) {
      cardColor = cs.errorContainer;
      textColor = cs.onErrorContainer;
      borderColor = cs.error.withAlpha(102);
    } else {
      cardColor = cs.surfaceContainerHigh;
      textColor = cs.onSurface;
      borderColor = null;
    }

    return AnimatedContainer(
      duration: const Duration(milliseconds: 180),
      curve: Curves.easeOut,
      width: double.infinity,
      decoration: BoxDecoration(
        color: cardColor,
        borderRadius: BorderRadius.circular(20),
        border: borderColor != null
            ? Border.all(color: borderColor, width: 1.5)
            : null,
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
            // Thin accent stripe at top.
            Container(
              height: 4,
              decoration: BoxDecoration(
                gradient: LinearGradient(
                  colors: isArrived
                      ? [cs.primary, cs.secondary]
                      : isError
                          ? [cs.error, cs.errorContainer]
                          : [cs.primary.withAlpha(180), cs.primaryContainer],
                ),
              ),
            ),
            Expanded(
              child: Stack(
                children: [
                  // Main scrollable text.
                  SingleChildScrollView(
                    padding: const EdgeInsets.fromLTRB(24, 20, 24, 20),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.center,
                      children: [
                        if (isArrived)
                          Padding(
                            padding: const EdgeInsets.only(bottom: 12),
                            child: Icon(
                              Icons.check_circle_rounded,
                              color: cs.primary,
                              size: 48,
                            ),
                          ),
                        Text(
                          message,
                          style: theme.textTheme.headlineMedium?.copyWith(
                            color: textColor,
                            height: 1.5,
                          ),
                          textAlign: TextAlign.center,
                        ),
                      ],
                    ),
                  ),
                  // Loading overlay — subtle, non-blocking.
                  if (isLoading)
                    Positioned(
                      top: 12,
                      right: 16,
                      child: SizedBox(
                        width: 24,
                        height: 24,
                        child: CircularProgressIndicator(
                          strokeWidth: 2.5,
                          color: cs.primary,
                        ),
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
