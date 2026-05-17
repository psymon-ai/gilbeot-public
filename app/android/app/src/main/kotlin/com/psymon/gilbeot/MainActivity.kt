package com.psymon.gilbeot

import android.app.ActivityManager
import android.content.ComponentCallbacks2
import android.content.Context
import io.flutter.embedding.android.FlutterActivity
import io.flutter.embedding.engine.FlutterEngine
import io.flutter.plugin.common.MethodChannel
import java.io.File
import java.util.concurrent.TimeUnit
import kotlin.concurrent.thread

class MainActivity: FlutterActivity() {
    private var memoryWarningChannel: MethodChannel? = null

    override fun onTrimMemory(level: Int) {
        super.onTrimMemory(level)
        // Android system 메모리 압박 콜백. level 분류 (낮은 → 높은 압박):
        //   TRIM_MEMORY_RUNNING_MODERATE/LOW/CRITICAL — foreground 인데 메모리 부족
        //   TRIM_MEMORY_UI_HIDDEN — UI 가려짐 (BYO 갤러리 진입 등)
        //   TRIM_MEMORY_BACKGROUND/MODERATE/COMPLETE — background 상태에서 LMK 임박
        // 우리 앱은 critical / background 시그널 받으면 Dart 쪽에 알려 cache
        // 정리 + 사용자 경고. complete 는 imminent kill 신호라 "지금 모델을
        // 잃을 수 있다" 통지에 사용.
        memoryWarningChannel?.invokeMethod("onTrimMemory", level)
    }

    override fun configureFlutterEngine(flutterEngine: FlutterEngine) {
        super.configureFlutterEngine(flutterEngine)
        // ── Memory query channel ──────────────────────────────────────────
        memoryWarningChannel = MethodChannel(
            flutterEngine.dartExecutor.binaryMessenger,
            "gilbeot/memory"
        ).apply {
            setMethodCallHandler { call, result ->
                when (call.method) {
                    "getMemoryInfo" -> {
                        val am = getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager
                        val mi = ActivityManager.MemoryInfo()
                        am.getMemoryInfo(mi)
                        result.success(mapOf(
                            "availMemBytes" to mi.availMem,
                            "totalMemBytes" to mi.totalMem,
                            "thresholdBytes" to mi.threshold,
                            "lowMemory" to mi.lowMemory,
                        ))
                    }
                    else -> result.notImplemented()
                }
            }
        }

        // ── Cactus ASR channel ────────────────────────────────────────────
        MethodChannel(
            flutterEngine.dartExecutor.binaryMessenger,
            "gilbeot/cactus_asr"
        ).setMethodCallHandler { call, result ->
            if (call.method != "transcribe") {
                result.notImplemented()
                return@setMethodCallHandler
            }
            val requestedBinPath = call.argument<String>("binPath") ?: ""
            val binPath = if (requestedBinPath.isBlank() || requestedBinPath == "native") {
                "${applicationInfo.nativeLibraryDir}/libcactus_asr.so"
            } else {
                requestedBinPath
            }
            val modelPath = call.argument<String>("modelPath") ?: ""
            val wavPath = call.argument<String>("wavPath") ?: ""
            val prompt = call.argument<String>("prompt") ?: "Transcribe the audio."
            val language = call.argument<String>("language") ?: "ko"
            val timeoutMs = call.argument<Number>("timeoutMs")?.toLong() ?: 300_000L

            thread(name = "cactus-asr") {
                val started = System.nanoTime()
                try {
                    if (binPath.isBlank() || modelPath.isBlank() || wavPath.isBlank()) {
                        throw IllegalArgumentException("Missing Cactus ASR argument")
                    }
                    val process = ProcessBuilder(
                        binPath,
                        modelPath,
                        wavPath,
                        "--language",
                        language,
                    )
                        .redirectErrorStream(true)
                        .directory(File(filesDir, "cactus"))
                        .apply {
                            environment()["CACTUS_NO_CLOUD_TELE"] = "1"
                            environment()["CACTUS_TRANSCRIBE_PROMPT"] = prompt
                        }
                        .start()

                    val completed = process.waitFor(timeoutMs, TimeUnit.MILLISECONDS)
                    val stdout = process.inputStream.bufferedReader(Charsets.UTF_8).readText()
                    val exitCode = if (completed) {
                        process.exitValue()
                    } else {
                        process.destroyForcibly()
                        -124
                    }
                    val elapsedMs = (System.nanoTime() - started) / 1_000_000
                    runOnUiThread {
                        result.success(
                            mapOf(
                                "exitCode" to exitCode,
                                "elapsedMs" to elapsedMs,
                                "stdout" to stdout,
                            )
                        )
                    }
                } catch (t: Throwable) {
                    val elapsedMs = (System.nanoTime() - started) / 1_000_000
                    runOnUiThread {
                        result.success(
                            mapOf(
                                "exitCode" to -1,
                                "elapsedMs" to elapsedMs,
                                "stdout" to (t.stackTraceToString()),
                            )
                        )
                    }
                }
            }
        }
    }
}
