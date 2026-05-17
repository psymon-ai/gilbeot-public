# 길벗 Flutter App

어르신을 위한 온디바이스 멀티모달 길안내 앱입니다. 사용자가 목적지를 말하면 Gemma 4가 음성을 해석하고, 한국 지도 API로 경로를 만든 뒤, 길이 헷갈릴 때 찍은 사진을 Gemma 4 vision으로 읽어 음성 안내를 제공합니다.

## Requirements

- Flutter SDK 3.38.4+
- Android SDK + adb
- JDK 17
- arm64 Android device for realistic Gemma 4 inference performance

## Local Setup

```powershell
cd app
Copy-Item assets\env_config.example assets\env_config
flutter pub get
flutter run -d <device-id>
```

`assets/env_config` is ignored by git because it may contain API keys. For the judge demo build, the default template values are enough except for any private model/token settings you choose to use. For the Korean production build, fill in the ODsay, Naver, and T-Map keys.

## Required Patched flutter_gemma Fork

Gilbeot release builds use a patched `flutter_gemma` 0.14.5 fork for LiteRT-LM speculative decoding and the LoRA sidecar path. The app expects the fork at `../third_party/flutter_gemma_repo` relative to this `app/` directory:

```powershell
cd ..
git clone https://github.com/DenisovAV/flutter_gemma third_party\flutter_gemma_repo
cd third_party\flutter_gemma_repo
git checkout v0.14.5
# Apply the Gilbeot patches from ..\..\patches\README.md.
cd ..\..\app
flutter pub get
```

The build scripts check for the required fork patches before building.

## Main Structure

```text
lib/
  main.dart                 app bootstrap, dotenv, Gemma/Naver init
  screens/home_screen.dart  main guidance state machine
  services/gemma_service.dart
  services/odsay_service.dart
  services/tmap_pedestrian_service.dart
  services/tts_service.dart
  config/demo_mode.dart
  config/demo_data.dart
```

## Build Targets

From the repository root:

```powershell
python scripts\build_install_demo_apk.py --no-install
python scripts\build_install_realuse_apk.py --no-install
```

The scripts create `app/assets/env_config` from the public template if it is missing, temporarily patch package id / app label, run `flutter build apk --release`, and restore the working tree afterward. The real-use script also validates that the production map/search API keys are not placeholders.
