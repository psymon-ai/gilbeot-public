# Build from source

## Prerequisites

- Flutter SDK ≥ 3.38.4 ([install](https://docs.flutter.dev/get-started/install))
- Android SDK + adb ([Android Studio][as] handles both)
- An arm64 Android device (emulator works for UI but the on-device
  Gemma 4 model needs a real ARM CPU/GPU for sane performance)

[as]: https://developer.android.com/studio

> **Windows.** The build scripts default to `C:\flutter\bin\flutter.bat`
> and `C:\android\platform-tools\adb.exe`, and set `ANDROID_HOME=C:\android`
> when that SDK directory exists. Override with the `FLUTTER`, `ADB`, and
> `ANDROID_HOME` environment variables.

## Step 1 — Clone the repo

```bash
git clone https://github.com/<your>/gilbeot-public.git
cd gilbeot-public
```

The patched `flutter_gemma` fork is **already vendored** at
`third_party/flutter_gemma_repo/` — clone gives you both the app and the
fork in one step. The app's `pubspec.yaml` keeps a local
`dependency_overrides` entry pointing at it, and the build scripts abort
early if the fork is missing or if the MTP / LoRA sidecar patches are
absent, because the verified S23/S10e runtime behavior depends on it.

> **Why vendored, not pinned to upstream:** the on-device behavior
> requires three Dart-side additions to `flutter_gemma` 0.14.5 (MTP
> wiring + LoRA sidecar path through the FFI client) plus the matching
> native `libLiteRtLm.so` prebuilts. Vendoring guarantees reproducibility
> if `DenisovAV/flutter_gemma` is reorganized or release artifacts move.
> See [`../patches/README.md`](../patches/README.md) for the exact
> additions and how to re-derive the fork from a fresh upstream clone.

## Step 2 — Configure

```bash
cd gilbeot-public/app/assets
cp env_config.example env_config
# Edit env_config and fill in the keys you need (see header comments).
```

The build scripts also create `app/assets/env_config` from
`env_config.example` automatically if it is missing. That generated file is
ignored by git because it may contain API keys.

For the **judge demo build** you need almost nothing — only
`MODEL_DOWNLOAD_URL` (already pre-filled in the template). The Korean
Map API keys can stay as placeholders.

For the **Korean production build** you need real keys for:
`ODSAY_ANDROID_KEY`, `NAVER_CLIENT_ID`, `NAVER_CLIENT_SECRET`, and
`TMAP_APP_KEY`. The real-use build script refuses placeholder values so a
release APK is not accidentally produced with non-working route/search config.

## Step 3 — Build the judge demo APK

```bash
cd gilbeot-public
python scripts/build_install_demo_apk.py --serial <YOUR_DEVICE_SERIAL>
```

The script:

1. Patches `app/android/app/build.gradle` →
   `applicationId = "com.psymon.gilbeot.demo"`.
2. Patches `AndroidManifest.xml` → label `Gilbeot Demo`.
3. Patches `env_config` → `DEMO_MODE=true`.
4. Runs `flutter build apk --release`.
5. **Restores all three patches** in a `finally` block (working tree
   stays in resting state regardless of build success/failure).
6. `adb install -r` the resulting APK.

Output: release APK at `app/build/app/outputs/flutter-apk/app-release.apk`.
First launch downloads the ~2.4 GB Gemma 4 model from Hugging Face.

## Step 4 — Build the Korean production APK

```bash
python scripts/build_install_realuse_apk.py --serial <YOUR_DEVICE_SERIAL>
```

Same script pattern but patches:

- `applicationId = "com.psymon.gilbeot.real"`
- label `Gilbeot`
- `env_config` left at resting (`DEMO_MODE=false`, real Map APIs used).

Both APKs install side-by-side because they have distinct `applicationId`s.

## Useful commands

```bash
# inspect logcat from a specific build
adb -s <serial> logcat -d -v time -s flutter:I | head -200

# pull the saved guidance session log (intro + per-photo instruction)
adb -s <serial> exec-out \
  cat "/storage/emulated/0/Android/data/com.psymon.gilbeot.demo/files/guidance_sessions/$(adb shell ls -t /storage/emulated/0/Android/data/com.psymon.gilbeot.demo/files/guidance_sessions | head -1)/log.txt"

# uninstall a build
adb -s <serial> uninstall com.psymon.gilbeot.demo
```

## Verifying the on-device timing

Look for these log lines in `adb logcat -s flutter:I`:

```
[LiteRtLmFfi] speculative decoding (Gemma 4 MTP): true (backend=gpu, ...)
[FfiInferenceModelSession/perf] generation total: 12877ms (prefill 5182ms + decode 7695ms over 59 chunks, ~7.5 chunks/sec)
```

On a S23 GPU the per-photo total should land in the 10–15 s range; on
S10e CPU 25–35 s. See [`architecture.md`](architecture.md) for the
verified numbers and what to expect on other devices.

## Troubleshooting

- **"Could not generate guidance."** — the model emitted invalid JSON.
  This used to happen when the schema example contained vague placeholder
  text; we now ship a concrete-example schema in the prompt. If you see
  it, capture the raw model output from logcat and check for unquoted
  array elements.
- **First launch hangs at "Loading model... 0%"** — the Hugging Face
  download is in progress. Wait 3–5 minutes on Wi-Fi. Watch
  `[gemma] download progress: N%` in logcat.
- **TTS reads "Welcome to Gilbeot" oddly** — the brand name is written
  as `Gil-but` in the welcome string specifically because en-US TTS
  hallucinates the spelling otherwise. Sounds approximately like the
  Korean original [kil.bʌt].
- **Map preview shows only origin/destination markers, no path** — the
  cached T-Map polyline (`assets/demo/route_polyline.json`) didn't
  bundle. Verify `pubspec.yaml` lists `assets/demo/route_polyline.json`.
