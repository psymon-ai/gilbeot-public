import 'dart:convert';
import 'dart:io';

import 'package:flutter/material.dart';
import 'package:path_provider/path_provider.dart';

import '../services/gemma_service.dart';

/// 배치 평가 화면 — 134-sample HF eval 을 device .litertlm 으로 처리해
/// transcribe 결과 + CER 을 results.json 으로 저장.
///
/// 사용 흐름:
///   1. ADB push 134 WAVs + refs.json → app external dir (eval/audio, eval/refs.json)
///   2. env_config 의 EVAL_MODE=true 로 빌드
///   3. 앱 실행 → 자동 batch eval
///   4. 완료 후 ADB pull results.json
class EvalScreen extends StatefulWidget {
  const EvalScreen({super.key});

  @override
  State<EvalScreen> createState() => _EvalScreenState();
}

class _EvalScreenState extends State<EvalScreen> {
  String _status = '준비 중...';
  int _done = 0;
  int _total = 0;
  int _ok = 0;
  int _errors = 0;
  double _runningCer = 0.0;
  bool _busy = false;

  @override
  void initState() {
    super.initState();
    _start();
  }

  Future<void> _start() async {
    setState(() => _status = '모델 초기화 중...');
    try {
      await GemmaService.instance.initialize();
    } catch (e) {
      setState(() => _status = '모델 로딩 실패: $e');
      return;
    }
    setState(() => _status = '평가 데이터 로딩 중...');
    final dir = await getExternalStorageDirectory();
    if (dir == null) {
      setState(() => _status = 'external storage 접근 실패');
      return;
    }
    final evalDir = Directory('${dir.path}/eval');
    final refsFile = File('${evalDir.path}/refs.json');
    if (!refsFile.existsSync()) {
      setState(() => _status = 'refs.json 없음: ${refsFile.path}');
      return;
    }
    final refsJson =
        jsonDecode(await refsFile.readAsString()) as Map<String, dynamic>;
    final items = (refsJson['items'] as List).cast<Map<String, dynamic>>();
    setState(() {
      _total = items.length;
      _busy = true;
    });

    final results = <Map<String, dynamic>>[];
    final cers = <double>[];
    final start = DateTime.now();

    for (var i = 0; i < items.length; i++) {
      final item = items[i];
      final name = item['name'] as String;
      final ref = item['ref'] as String;
      final wavFile = File('${evalDir.path}/audio/$name');
      if (!wavFile.existsSync()) {
        results.add({
          'i': i,
          'name': name,
          'ref': ref,
          'error': 'wav 없음',
        });
        setState(() {
          _errors++;
          _done = i + 1;
        });
        continue;
      }
      final wav = await wavFile.readAsBytes();
      final t0 = DateTime.now();
      String? hyp;
      String? error;
      try {
        hyp = await GemmaService.instance.transcribeAudio(wav);
      } catch (e) {
        error = e.toString();
      }
      final rawHyp = GemmaService.instance.lastRawAudioTranscript ?? hyp ?? '';
      final dt = DateTime.now().difference(t0).inMilliseconds;
      final rawCer = error == null ? _cer(ref, rawHyp) : null;
      if (rawCer != null) cers.add(rawCer);
      results.add({
        'i': i,
        'name': name,
        'ref': ref,
        'raw_hyp': rawHyp,
        'hyp': hyp ?? '',
        'cer': rawCer,
        'raw_cer': rawCer,
        'duration_sec': item['duration_sec'],
        'elapsed_ms': dt,
        'error': error,
      });
      setState(() {
        _done = i + 1;
        if (error == null) {
          _ok++;
        } else {
          _errors++;
        }
        if (cers.isNotEmpty) {
          _runningCer = cers.reduce((a, b) => a + b) / cers.length;
        }
        _status =
            '$name | hyp="${(hyp ?? '').replaceAll(RegExp(r"\s+"), " ")}"';
      });
      // 중간 저장 — crash 시에도 진행분 보존.
      if (i % 5 == 0 || i + 1 == items.length) {
        await _writeResults(
          evalDir,
          items.length,
          results,
          cers,
          start,
        );
      }
    }
    await _writeResults(evalDir, items.length, results, cers, start);
    setState(() {
      _busy = false;
      _status = '완료. results.json 저장됨.';
    });
  }

  Future<void> _writeResults(
    Directory evalDir,
    int total,
    List<Map<String, dynamic>> rows,
    List<double> cers,
    DateTime start,
  ) async {
    final out = File('${evalDir.path}/results.json');
    final dur = DateTime.now().difference(start).inSeconds;
    final summary = {
      'total': total,
      'done': rows.length,
      'ok': rows.where((r) => r['error'] == null).length,
      'errors': rows.where((r) => r['error'] != null).length,
      'mean_cer':
          cers.isEmpty ? null : cers.reduce((a, b) => a + b) / cers.length,
      // raw_mean_cer 필드는 raw vs cleaned CER 분리 측정 의도였으나 실제로는
      // 같은 cers 평균을 두 번 계산하던 dead duplicate 였음 — 제거. 추후 echo
      // strip 전/후 분리 측정이 필요하면 rows 에 raw_hyp 보존 후 별도 계산.
      'elapsed_sec': dur,
    };
    await out.writeAsString(
      const JsonEncoder.withIndent('  ').convert({
        'summary': summary,
        'rows': rows,
      }),
    );
  }

  double _cer(String ref, String hyp) {
    final r = _normalize(ref);
    final h = _normalize(hyp);
    if (r.isEmpty) return h.isEmpty ? 0.0 : 1.0;
    return _levenshtein(r, h) / r.length;
  }

  String _normalize(String s) {
    return s.replaceAll(RegExp(r'[^\w가-힣]'), '').replaceAll(RegExp(r'\s+'), '');
  }

  int _levenshtein(String a, String b) {
    if (a.isEmpty) return b.length;
    if (b.isEmpty) return a.length;
    var prev = List<int>.generate(b.length + 1, (i) => i);
    final cur = List<int>.filled(b.length + 1, 0);
    for (var i = 1; i <= a.length; i++) {
      cur[0] = i;
      for (var j = 1; j <= b.length; j++) {
        final cost = a.codeUnitAt(i - 1) == b.codeUnitAt(j - 1) ? 0 : 1;
        cur[j] = [
          cur[j - 1] + 1,
          prev[j] + 1,
          prev[j - 1] + cost,
        ].reduce((x, y) => x < y ? x : y);
      }
      prev = List<int>.from(cur);
    }
    return prev[b.length];
  }

  @override
  Widget build(BuildContext context) {
    final pct = _total == 0 ? 0.0 : _done / _total;
    return Scaffold(
      appBar: AppBar(title: const Text('배치 평가')),
      body: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            const SizedBox(height: 8),
            LinearProgressIndicator(value: _total == 0 ? null : pct),
            const SizedBox(height: 16),
            Text(
              'progress: $_done / $_total  (ok=$_ok err=$_errors)',
              style: const TextStyle(fontSize: 18, fontWeight: FontWeight.w600),
            ),
            const SizedBox(height: 8),
            Text(
              'mean CER: ${_runningCer.toStringAsFixed(4)}',
              style: const TextStyle(fontSize: 18, fontWeight: FontWeight.w600),
            ),
            const Divider(height: 32),
            Expanded(
              child: SingleChildScrollView(
                child: Text(
                  _status,
                  style: const TextStyle(fontSize: 14, fontFamily: 'monospace'),
                ),
              ),
            ),
            if (!_busy)
              const Padding(
                padding: EdgeInsets.symmetric(vertical: 8),
                child: Text('완료. results.json 저장됨.\n'
                    'adb pull /sdcard/Android/data/com.psymon.gilbeot/files/eval/results.json'),
              ),
          ],
        ),
      ),
    );
  }
}
