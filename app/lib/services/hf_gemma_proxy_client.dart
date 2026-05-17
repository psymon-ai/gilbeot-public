import 'package:dio/dio.dart';
import 'package:flutter/foundation.dart';

class HfGemmaProxyClient {
  HfGemmaProxyClient(String baseUrl)
      : _dio = Dio(
          BaseOptions(
            baseUrl: _normalizeBaseUrl(baseUrl),
            connectTimeout: const Duration(seconds: 8),
            sendTimeout: const Duration(seconds: 30),
            receiveTimeout: const Duration(minutes: 5),
          ),
        );

  final Dio _dio;

  static String _normalizeBaseUrl(String baseUrl) {
    final trimmed = baseUrl.trim();
    if (trimmed.endsWith('/')) {
      return trimmed.substring(0, trimmed.length - 1);
    }
    return trimmed;
  }

  Future<void> health() async {
    final response = await _dio.get<Map<String, dynamic>>('/health');
    final data = response.data ?? const <String, dynamic>{};
    if (data['ok'] != true) {
      throw StateError('HF Gemma proxy is not ready: $data');
    }
    debugPrint('[gemma/hf-proxy] health ok: $data');
  }

  Future<String> transcribeAudio(Uint8List wavBytes) async {
    final response = await _postAudio('/v1/transcribe', wavBytes);
    final transcript = response['transcript']?.toString().trim() ?? '';
    if (transcript.isEmpty) {
      throw StateError('HF Gemma proxy returned an empty transcript: $response');
    }
    return transcript;
  }

  Future<Map<String, String?>> parseDestinationFromAudio(
    Uint8List wavBytes,
  ) async {
    final response = await _postAudio('/v1/intent', wavBytes);
    return {
      'transcript': _stringOrNull(response['transcript']),
      'destination': _stringOrNull(response['destination']),
      'origin': _stringOrNull(response['origin']),
    };
  }

  Future<Map<String, String?>> parseDestination(String userText) async {
    final response = await _dio.post<Map<String, dynamic>>(
      '/v1/parse-destination',
      data: {'text': userText},
    );
    final data = response.data ?? const <String, dynamic>{};
    return {
      'destination': _stringOrNull(data['destination']),
      'origin': _stringOrNull(data['origin']),
    };
  }

  Future<Map<String, dynamic>> _postAudio(String path, Uint8List wavBytes) async {
    final form = FormData.fromMap({
      'audio': MultipartFile.fromBytes(wavBytes, filename: 'audio.wav'),
    });
    final response = await _dio.post<Map<String, dynamic>>(path, data: form);
    return response.data ?? const <String, dynamic>{};
  }

  String? _stringOrNull(Object? value) {
    final text = value?.toString().trim();
    if (text == null || text.isEmpty || text.toLowerCase() == 'null') {
      return null;
    }
    return text;
  }
}
