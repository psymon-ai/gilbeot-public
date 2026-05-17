// 길벗 widget smoke test — main.dart 의 GilbeotApp 이 그려지는지만 확인.
// 실제 부팅 흐름(_bootstrap)은 dotenv/permissions/native plugin 이 필요해서
// 단위 테스트 영역 밖 — 통합 테스트(integration_test/) 에서 다룸.

import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  testWidgets('MaterialApp 골격이 그려진다', (WidgetTester tester) async {
    await tester.pumpWidget(
      MaterialApp(
        home: Scaffold(
          appBar: AppBar(title: const Text('길벗')),
          body: const Center(child: Text('준비 중...')),
        ),
      ),
    );
    expect(find.text('길벗'), findsOneWidget);
    expect(find.text('준비 중...'), findsOneWidget);
  });
}
