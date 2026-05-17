import 'package:flutter/material.dart';

/// Standardised AppBar for all Gilbeot screens.
///
/// Uses the theme's AppBar colour (teal primary) with a centred title.
/// [trailingBadge] — optional small badge text shown after the title
/// (e.g. " [TEST]") in a muted style so it doesn't compete with the title.
class GilbeotAppBar extends StatelessWidget implements PreferredSizeWidget {
  const GilbeotAppBar({
    super.key,
    required this.title,
    this.trailingBadge,
    this.leading,
  });

  final String title;
  final String? trailingBadge;
  final Widget? leading;

  @override
  Size get preferredSize => const Size.fromHeight(72);

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final badge = trailingBadge;
    return AppBar(
      leading: leading,
      title: Row(
        mainAxisSize: MainAxisSize.min,
        crossAxisAlignment: CrossAxisAlignment.baseline,
        textBaseline: TextBaseline.alphabetic,
        children: [
          Text(
            title,
            style: theme.appBarTheme.titleTextStyle,
          ),
          if (badge != null && badge.isNotEmpty) ...[
            const SizedBox(width: 6),
            Text(
              badge,
              style: theme.textTheme.labelSmall?.copyWith(
                color: theme.colorScheme.onPrimary.withAlpha(178),
                fontWeight: FontWeight.w500,
              ),
            ),
          ],
        ],
      ),
    );
  }
}
