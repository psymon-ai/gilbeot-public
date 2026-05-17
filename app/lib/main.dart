import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_dotenv/flutter_dotenv.dart';
import 'package:flutter_gemma/flutter_gemma.dart';
import 'package:flutter_naver_map/flutter_naver_map.dart';
import 'package:google_fonts/google_fonts.dart';

import 'screens/eval_screen.dart';
import 'screens/home_screen.dart';

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  // dot prefix 파일은 Flutter 3.41+ asset bundler 가 자동 제외 → `.env` 대신 `env_config`.
  await dotenv.load(fileName: 'assets/env_config');
  // 0.14.x 부터 plugin 사용 전 명시적 initialize 필수 (LiteRT-LM FFI registry 셋업).
  await FlutterGemma.initialize();

  // NCP Mobile Dynamic Map SDK — Application 의 Android applicationId
  // (`com.psymon.gilbeot`) 화이트리스트로 인증. Web URL 검증 없음.
  final naverId = dotenv.maybeGet('NAVER_CLIENT_ID')?.trim() ?? '';
  if (naverId.isNotEmpty) {
    await FlutterNaverMap().init(
      clientId: naverId,
      onAuthFailed: (ex) {
        debugPrint('[naver_map/auth_failed] $ex');
      },
    );
  }

  // Lock to portrait — landscape breaks the home/camera layouts (carded hero
  // CTA + bottom action stack assumes a tall safe area).
  await SystemChrome.setPreferredOrientations(const [
    DeviceOrientation.portraitUp,
  ]);

  // Immersive: status bar light icons on dark app bar.
  SystemChrome.setSystemUIOverlayStyle(const SystemUiOverlayStyle(
    statusBarColor: Colors.transparent,
    statusBarIconBrightness: Brightness.light,
  ));

  runApp(const GilbeotApp());
}

bool get _evalMode =>
    (dotenv.maybeGet('EVAL_MODE')?.toLowerCase().trim() == 'true');

// ---------------------------------------------------------------------------
// Design tokens
// ---------------------------------------------------------------------------

/// Deep teal primary — calm, trustworthy, readable against white/light surfaces.
const Color _seedColor = Color(0xFF00695C); // teal-800

/// Warm amber — voice CTA accent. High-energy, familiar for "press to speak".
const Color _accentColor = Color(0xFFFFB300); // amber-700

/// Teal-900 used as dark surface primary.
const Color _darkSeedColor = Color(0xFF004D40);

/// 4dp base grid unit.
const double _grid = 4.0;

// ---------------------------------------------------------------------------

class GilbeotApp extends StatelessWidget {
  const GilbeotApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: '길벗',
      theme: _buildTheme(Brightness.light),
      darkTheme: _buildTheme(Brightness.dark),
      themeMode: ThemeMode.system,
      debugShowCheckedModeBanner: false,
      home: _evalMode ? const EvalScreen() : const HomeScreen(),
    );
  }
}

ThemeData _buildTheme(Brightness brightness) {
  final isDark = brightness == Brightness.dark;

  final colorScheme = ColorScheme.fromSeed(
    seedColor: isDark ? _darkSeedColor : _seedColor,
    brightness: brightness,
  ).copyWith(
    // Override so voice CTA uses warm amber regardless of generated scheme.
    tertiary: _accentColor,
    onTertiary: Colors.black,
    tertiaryContainer: const Color(0xFFFFF8E1),
    onTertiaryContainer: const Color(0xFF3E2000),
  );

  // Noto Sans KR — excellent 한글 support, shipped via google_fonts.
  // We override the entire text theme to enforce the Korean-optimised family.
  final notoBase = GoogleFonts.notoSansKrTextTheme(
    ThemeData(brightness: brightness).textTheme,
  );

  final textTheme = notoBase.copyWith(
    // Screen title (AppBar, large headings)
    displayLarge: notoBase.displayLarge?.copyWith(
      fontSize: 40,
      fontWeight: FontWeight.w800,
      height: 1.2,
      letterSpacing: -0.5,
    ),
    displayMedium: notoBase.displayMedium?.copyWith(
      fontSize: 36,
      fontWeight: FontWeight.w700,
      height: 1.25,
    ),
    // Section headings / status card primary text
    headlineLarge: notoBase.headlineLarge?.copyWith(
      fontSize: 32,
      fontWeight: FontWeight.w700,
      height: 1.3,
    ),
    headlineMedium: notoBase.headlineMedium?.copyWith(
      fontSize: 28,
      fontWeight: FontWeight.w600,
      height: 1.35,
    ),
    headlineSmall: notoBase.headlineSmall?.copyWith(
      fontSize: 24,
      fontWeight: FontWeight.w600,
      height: 1.35,
    ),
    // Body — ≥ 18sp hard requirement
    bodyLarge: notoBase.bodyLarge?.copyWith(
      fontSize: 22,
      fontWeight: FontWeight.w400,
      height: 1.5,
    ),
    bodyMedium: notoBase.bodyMedium?.copyWith(
      fontSize: 18,
      fontWeight: FontWeight.w400,
      height: 1.5,
    ),
    bodySmall: notoBase.bodySmall?.copyWith(
      fontSize: 16,
      fontWeight: FontWeight.w400,
      height: 1.5,
    ),
    // Button / label
    labelLarge: notoBase.labelLarge?.copyWith(
      fontSize: 24,
      fontWeight: FontWeight.w700,
      letterSpacing: 0.1,
    ),
    labelMedium: notoBase.labelMedium?.copyWith(
      fontSize: 20,
      fontWeight: FontWeight.w600,
    ),
    labelSmall: notoBase.labelSmall?.copyWith(
      fontSize: 16,
      fontWeight: FontWeight.w500,
    ),
    // Title (AppBar widget title)
    titleLarge: notoBase.titleLarge?.copyWith(
      fontSize: 32,
      fontWeight: FontWeight.w800,
      letterSpacing: -0.3,
    ),
    titleMedium: notoBase.titleMedium?.copyWith(
      fontSize: 22,
      fontWeight: FontWeight.w600,
    ),
    titleSmall: notoBase.titleSmall?.copyWith(
      fontSize: 18,
      fontWeight: FontWeight.w600,
    ),
  );

  return ThemeData(
    useMaterial3: true,
    colorScheme: colorScheme,
    textTheme: textTheme,

    // AppBar — filled with primary, large toolbar for legibility.
    appBarTheme: AppBarTheme(
      backgroundColor: colorScheme.primary,
      foregroundColor: colorScheme.onPrimary,
      elevation: 0,
      centerTitle: true,
      toolbarHeight: 72,
      titleTextStyle: textTheme.titleLarge?.copyWith(
        color: colorScheme.onPrimary,
      ),
      systemOverlayStyle: SystemUiOverlayStyle(
        statusBarColor: Colors.transparent,
        statusBarIconBrightness:
            isDark ? Brightness.light : Brightness.light,
      ),
    ),

    // FilledButton — used for secondary actions.
    filledButtonTheme: FilledButtonThemeData(
      style: FilledButton.styleFrom(
        minimumSize: const Size(double.infinity, 64),
        textStyle: textTheme.labelLarge,
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(_grid * 7), // 28dp
        ),
        padding: const EdgeInsets.symmetric(
          horizontal: _grid * 6,
          vertical: _grid * 4,
        ),
      ),
    ),

    // ElevatedButton — kept for legacy callers; matches FilledButton scale.
    elevatedButtonTheme: ElevatedButtonThemeData(
      style: ElevatedButton.styleFrom(
        minimumSize: const Size(double.infinity, 64),
        textStyle: textTheme.labelLarge,
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(_grid * 5), // 20dp
        ),
        padding: const EdgeInsets.symmetric(
          horizontal: _grid * 6,
          vertical: _grid * 4,
        ),
      ),
    ),

    // OutlinedButton.
    outlinedButtonTheme: OutlinedButtonThemeData(
      style: OutlinedButton.styleFrom(
        minimumSize: const Size(double.infinity, 64),
        textStyle: textTheme.labelLarge,
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(_grid * 5),
        ),
      ),
    ),

    // Card — gentle elevation, generous radius.
    cardTheme: CardThemeData(
      elevation: 2,
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(_grid * 5), // 20dp
      ),
      margin: EdgeInsets.zero,
    ),

    // SnackBar — larger text for readability.
    snackBarTheme: SnackBarThemeData(
      contentTextStyle: textTheme.bodyLarge,
      behavior: SnackBarBehavior.floating,
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(_grid * 4),
      ),
    ),

    // Icon size default — filled style throughout.
    iconTheme: const IconThemeData(size: 32),

    // Divider
    dividerTheme: const DividerThemeData(space: _grid * 4),

    // Visual density — comfortable for touch targets.
    visualDensity: VisualDensity.comfortable,
  );
}
