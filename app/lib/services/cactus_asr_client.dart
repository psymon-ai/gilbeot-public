import 'dart:convert';
import 'dart:io';

import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';
import 'package:path_provider/path_provider.dart';

class CactusAsrClient {
  CactusAsrClient({
    required this.binPath,
    required this.modelPath,
    required this.prompt,
    this.language = 'ko',
    this.timeout = const Duration(minutes: 5),
  });

  static const MethodChannel _channel = MethodChannel('gilbeot/cactus_asr');

  final String binPath;
  final String modelPath;
  final String prompt;
  final String language;
  final Duration timeout;

  Future<String> transcribeAudio(Uint8List wavBytes) async {
    final dir = await getTemporaryDirectory();
    final wavFile = File(
      '${dir.path}/cactus_${DateTime.now().microsecondsSinceEpoch}.wav',
    );
    await wavFile.writeAsBytes(wavBytes, flush: true);

    final sw = Stopwatch()..start();
    try {
      final response = await _channel
          .invokeMapMethod<String, dynamic>('transcribe', {
            'binPath': binPath,
            'modelPath': modelPath,
            'wavPath': wavFile.path,
            'prompt': prompt,
            'language': language,
            'timeoutMs': timeout.inMilliseconds,
          });
      sw.stop();
      final exitCode = (response?['exitCode'] as num?)?.toInt() ?? -1;
      final elapsedMs = (response?['elapsedMs'] as num?)?.toInt();
      final stdout = response?['stdout']?.toString() ?? '';
      debugPrint(
        '[cactus/asr] exit=$exitCode wall=${sw.elapsedMilliseconds}ms '
        'native=${elapsedMs ?? -1}ms',
      );
      if (exitCode != 0) {
        throw StateError(
          'Cactus ASR failed (exit=$exitCode): ${_tail(stdout)}',
        );
      }
      final transcript = _reassembleByteFallback(_extractTranscript(stdout));
      if (transcript.isEmpty) {
        throw StateError('Cactus ASR returned no transcript: ${_tail(stdout)}');
      }
      debugPrint('[cactus/asr] transcript="$transcript"');
      return transcript;
    } finally {
      try {
        await wavFile.delete();
      } catch (_) {
        // Best-effort cleanup only.
      }
    }
  }

  static String _extractTranscript(String stdout) {
    final candidates = <String>[];
    for (final line in stdout.split(RegExp(r'\r?\n'))) {
      final trimmed = line.trim();
      if (trimmed.isEmpty) continue;
      if (trimmed.startsWith('[')) continue;
      if (trimmed.startsWith('Loading model')) continue;
      if (trimmed.startsWith('Model loaded')) continue;
      if (trimmed.startsWith('Transcribing:')) continue;
      if (trimmed.startsWith('Error:')) continue;
      if (trimmed.contains('Goodbye')) continue;
      if (trimmed.startsWith('Usage:')) continue;
      candidates.add(trimmed);
    }
    return candidates.isEmpty ? '' : candidates.last.trim();
  }

  static String _reassembleByteFallback(String raw) {
    final spanPattern = RegExp(r'(?:<0x[0-9A-Fa-f]{2}>)+');
    final bytePattern = RegExp(r'<0x([0-9A-Fa-f]{2})>');
    return raw.replaceAllMapped(spanPattern, (span) {
      final bytes = bytePattern
          .allMatches(span.group(0)!)
          .map((match) => int.parse(match.group(1)!, radix: 16))
          .toList(growable: false);
      try {
        return utf8.decode(bytes);
      } on FormatException {
        return span.group(0)!;
      }
    });
  }

  static String _tail(String text) {
    final trimmed = text.trim();
    if (trimmed.length <= 600) return trimmed;
    return trimmed.substring(trimmed.length - 600);
  }
}
