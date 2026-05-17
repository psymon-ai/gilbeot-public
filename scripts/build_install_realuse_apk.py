#!/usr/bin/env python3
r"""Build/install the *real-use* Gilbeot APK alongside the demo build.

The repository resting app is com.psymon.gilbeot, label "길벗". This script
produces the real-use side-by-side app:
  - applicationId  com.psymon.gilbeot.real   (coexists with the demo builds)
  - android:label  Gilbeot                   (distinct app drawer name)

It temporarily patches build.gradle / AndroidManifest.xml, builds, installs,
then restores both files in a finally block so the working tree stays in the
resting configuration.

Usage:
  python scripts/build_install_realuse_apk.py            # build + install
  python scripts/build_install_realuse_apk.py --no-install

Env overrides:
  FLUTTER       flutter path  (default C:\flutter\bin\flutter.bat)
  ADB           adb path      (default C:\android\platform-tools\adb.exe)
  ANDROID_HOME  Android SDK   (default C:\android when present)
  S10E_SERIAL   device serial (default R39M500MR1J)
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

REALUSE_PKG = "com.psymon.gilbeot.real"
DEFAULT_SERIAL = os.environ.get("S10E_SERIAL", "R39M500MR1J")
REQUIRED_REALUSE_KEYS = [
    "ODSAY_ANDROID_KEY",
    "NAVER_CLIENT_ID",
    "NAVER_CLIENT_SECRET",
    "TMAP_APP_KEY",
]

# (file, old, new) patch triples. Each `old` must appear verbatim exactly once
# in the resting configuration; the script aborts (and rolls back) otherwise.
PATCHES = [
    (BUILD_GRADLE,
     'applicationId = "com.psymon.gilbeot"',
     f'applicationId = "{REALUSE_PKG}"'),
    (MANIFEST,
     'android:label="길벗"',
     'android:label="Gilbeot"'),
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


def read_env_config() -> dict[str, str]:
    values: dict[str, str] = {}
    for raw_line in ENV_CONFIG.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip()
    return values


def is_placeholder(value: str | None) -> bool:
    if value is None:
        return True
    stripped = value.strip()
    return (
        not stripped
        or stripped.startswith("<")
        or stripped.endswith(">")
        or "YOUR_" in stripped
    )


def validate_realuse_env_config() -> None:
    values = read_env_config()
    missing = [key for key in REQUIRED_REALUSE_KEYS
               if is_placeholder(values.get(key))]
    if values.get("DEMO_MODE", "").lower() == "true":
        missing.append("DEMO_MODE must be false")
    if missing:
        raise SystemExit(
            "[abort] real-use APK requires real production config in "
            f"{ENV_CONFIG}.\n"
            "Fill these before building com.psymon.gilbeot.real:\n"
            + "\n".join(f"  - {key}" for key in missing)
        )


def run(cmd: list[str], cwd: Path) -> None:
    print("[run] " + " ".join(str(c) for c in cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd), check=True)


def apply_patches() -> dict[Path, bytes]:
    """Apply every patch; return {path: true_original_bytes} for restore."""
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
    print("[restore] build.gradle / AndroidManifest.xml restored")


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
    validate_realuse_env_config()
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
        print("[done] real-use APK built (not installed). Working tree is "
              "back in resting configuration.")
        return

    run([adb, "-s", args.serial, "install", "-r", str(built_apk)], ROOT)
    run([adb, "-s", args.serial, "shell", "am", "force-stop", REALUSE_PKG],
        ROOT)
    print(f"[done] real-use APK installed: {REALUSE_PKG} / label 'Gilbeot', "
          "side by side with the judge demo "
          "(com.psymon.gilbeot.demo / 'Gilbeot Demo').")
    print("[note] guidance sessions for this build land under "
          f"/sdcard/Android/data/{REALUSE_PKG}/files/guidance_sessions/ - "
          f"pull with GILBEOT_ANDROID_PKG={REALUSE_PKG}")


if __name__ == "__main__":
    if sys.platform != "win32":
        print("[warn] tuned for this Windows workspace, but will continue.")
    main()
