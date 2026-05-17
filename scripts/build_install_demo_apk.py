#!/usr/bin/env python3
r"""Build/install the *judge demo* Gilbeot APK alongside the Korean builds.

Hackathon (Kaggle Gemma 4 Good Hackathon) submission build for international
judges who don't have Korean-territory Map APIs, Korean speech, or Korean
signage. Runs a controlled scenario:

  - applicationId  com.psymon.gilbeot.demo   (coexists with com.psymon.gilbeot
                                              and com.psymon.gilbeot.real)
  - android:label  Gilbeot Demo              (distinct app drawer name)
  - env_config     DEMO_MODE=true            (lib/config/demo_mode.dart gate)

The on-device Gemma 4 model still runs for real on every photo and audio
input — only routing and STT input are pre-baked from local assets so the
APK works anywhere on Earth without the Korean Map APIs.

Usage:
  python scripts/build_install_demo_apk.py            # build + install on S23
  python scripts/build_install_demo_apk.py --no-install
  python scripts/build_install_demo_apk.py --serial R39M500MR1J  # S10e

Env overrides:
  FLUTTER       flutter path  (default C:\flutter\bin\flutter.bat)
  ADB           adb path      (default C:\android\platform-tools\adb.exe)
  ANDROID_HOME  Android SDK   (default C:\android when present)
  DEMO_SERIAL   device serial (default R3CW209MZKW = S23)
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
APP_DIR = ROOT / "app"
BUILD_GRADLE = APP_DIR / "android" / "app" / "build.gradle"
MANIFEST = APP_DIR / "android" / "app" / "src" / "main" / "AndroidManifest.xml"
ENV_CONFIG = APP_DIR / "assets" / "env_config"
ENV_CONFIG_EXAMPLE = APP_DIR / "assets" / "env_config.example"
DEFAULT_ANDROID_HOME = Path(r"C:\android")
FLUTTER_GEMMA_FORK = ROOT / "third_party" / "flutter_gemma_repo"

DEMO_PKG = "com.psymon.gilbeot.demo"
DEFAULT_SERIAL = os.environ.get("DEMO_SERIAL", "R3CW209MZKW")

# (file, old, new) patch triples. Each `old` must appear verbatim exactly once
# in the resting configuration; the script aborts and rolls back otherwise.
PATCHES = [
    (BUILD_GRADLE,
     'applicationId = "com.psymon.gilbeot"',
     f'applicationId = "{DEMO_PKG}"'),
    (MANIFEST,
     'android:label="길벗"',
     'android:label="Gilbeot Demo"'),
    (ENV_CONFIG,
     'DEMO_MODE=false',
     'DEMO_MODE=true'),
]


def find_flutter() -> str:
    env = os.environ.get("FLUTTER")
    if env:
        return env
    for c in (shutil.which("flutter"), r"C:\flutter\bin\flutter.bat"):
        if c and Path(c).exists():
            return c
    raise SystemExit("[abort] flutter not found. Set FLUTTER.")


def find_adb() -> str:
    env = os.environ.get("ADB")
    if env:
        return env
    for c in (shutil.which("adb"), r"C:\android\platform-tools\adb.exe"):
        if c and Path(c).exists():
            return c
    raise SystemExit("[abort] adb not found. Set ADB.")


def ensure_env_config() -> None:
    """Create the ignored local env_config from the public template if needed."""
    if ENV_CONFIG.exists():
        return
    if not ENV_CONFIG_EXAMPLE.exists():
        raise SystemExit(f"[abort] missing config template: {ENV_CONFIG_EXAMPLE}")
    ENV_CONFIG.write_bytes(ENV_CONFIG_EXAMPLE.read_bytes())
    print("[config] created app/assets/env_config from env_config.example")


def ensure_android_home() -> None:
    if os.environ.get("ANDROID_HOME") or os.environ.get("ANDROID_SDK_ROOT"):
        return
    if DEFAULT_ANDROID_HOME.exists():
        os.environ["ANDROID_HOME"] = str(DEFAULT_ANDROID_HOME)
        os.environ["ANDROID_SDK_ROOT"] = str(DEFAULT_ANDROID_HOME)
        print(f"[config] ANDROID_HOME={DEFAULT_ANDROID_HOME}")


def ensure_flutter_gemma_fork() -> None:
    """Fail early if the release-critical patched fork is not installed."""
    required = [
        FLUTTER_GEMMA_FORK / "pubspec.yaml",
        FLUTTER_GEMMA_FORK / "native" / "litert_lm" / "include" / "engine.h",
        FLUTTER_GEMMA_FORK / "lib" / "core" / "ffi" / "litert_lm_client.dart",
        FLUTTER_GEMMA_FORK / "lib" / "core" / "ffi" / "litert_lm_bindings.dart",
    ]
    missing = [p for p in required if not p.exists()]
    if missing:
        raise SystemExit(
            "[abort] patched flutter_gemma fork is required at "
            f"{FLUTTER_GEMMA_FORK}. See patches/README.md.\n"
            + "\n".join(f"  missing: {p}" for p in missing)
        )

    engine = (FLUTTER_GEMMA_FORK / "native" / "litert_lm" / "include"
              / "engine.h").read_text(encoding="utf-8")
    client = (FLUTTER_GEMMA_FORK / "lib" / "core" / "ffi"
              / "litert_lm_client.dart").read_text(encoding="utf-8")
    bindings = (FLUTTER_GEMMA_FORK / "lib" / "core" / "ffi"
                / "litert_lm_bindings.dart").read_text(encoding="utf-8")
    checks = {
        "MTP engine header":
            "litert_lm_engine_settings_set_enable_speculative_decoding"
            in engine,
        "MTP Dart binding":
            "litert_lm_engine_settings_set_enable_speculative_decoding"
            in bindings,
        "MTP client switch": "enableSpeculativeDecoding" in client,
        "LoRA sidecar shim": "SetScopedLoraFile" in client,
    }
    failed = [name for name, ok in checks.items() if not ok]
    if failed:
        raise SystemExit(
            "[abort] flutter_gemma fork is present but missing required "
            f"Gilbeot patches: {', '.join(failed)}. See patches/README.md."
        )
    print(f"[config] flutter_gemma fork={FLUTTER_GEMMA_FORK}")


def run(cmd: list[str], cwd: Path) -> None:
    print("[run] " + " ".join(str(c) for c in cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd), check=True)


def apply_patches() -> dict[Path, bytes]:
    """Apply every patch; return {path: true_original_bytes} for restore.

    Snapshots each file's pristine bytes exactly once so the multiple
    env_config patches restore cleanly instead of leaving env_config
    partially patched.
    """
    originals: dict[Path, bytes] = {}
    for path, _old, _new in PATCHES:
        if path not in originals:
            originals[path] = path.read_bytes()
    for path, old, new in PATCHES:
        text = path.read_bytes().decode("utf-8")
        if text.count(old) != 1:
            for p, data in originals.items():  # roll back before aborting
                p.write_bytes(data)
            raise SystemExit(
                f"[abort] expected exactly one {old!r} in {path}, "
                f"found {text.count(old)}"
            )
        path.write_bytes(text.replace(old, new, 1).encode("utf-8"))
        print(f"[patch] {path.name}: {old!r} -> {new!r}")
    return originals


def restore(originals: dict[Path, bytes]) -> None:
    for path, data in originals.items():
        path.write_bytes(data)
    print("[restore] build.gradle / AndroidManifest.xml / env_config restored")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-install", action="store_true",
                        help="build only, do not adb install")
    parser.add_argument("--serial", default=DEFAULT_SERIAL)
    args = parser.parse_args()

    flutter = find_flutter()
    adb = None if args.no_install else find_adb()
    ensure_android_home()
    ensure_env_config()
    ensure_flutter_gemma_fork()
    built_apk = (APP_DIR / "build" / "app" / "outputs" / "flutter-apk"
                 / "app-release.apk")

    originals = apply_patches()
    try:
        run([flutter, "build", "apk", "--release"], APP_DIR)
        if not built_apk.exists():
            raise SystemExit(f"[abort] APK missing after build: {built_apk}")
        size_mb = built_apk.stat().st_size / 1024 / 1024
        print(f"[apk] {built_apk} ({size_mb:.1f} MB)")
    finally:
        restore(originals)

    if args.no_install:
        print("[done] judge-demo APK built (not installed). Working tree is "
              "back in resting configuration.")
        return

    run([adb, "-s", args.serial, "install", "-r", str(built_apk)], ROOT)
    run([adb, "-s", args.serial, "shell", "am", "force-stop", DEMO_PKG], ROOT)
    print(f"[done] judge-demo APK installed: {DEMO_PKG} / label "
          f"'Gilbeot Demo' on {args.serial}, side by side with the Korean "
          "production builds (com.psymon.gilbeot / com.psymon.gilbeot.real).")
    print("[note] guidance sessions for this build land under "
          f"/sdcard/Android/data/{DEMO_PKG}/files/guidance_sessions/")


if __name__ == "__main__":
    if sys.platform != "win32":
        print("[warn] tuned for this Windows workspace, but will continue.")
    main()
