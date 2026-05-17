# flutter_gemma fork patches

Gilbeot uses a **forked `flutter_gemma` 0.14.5**, not the upstream pub.dev
package, because the upstream layer never wires LiteRT-LM's
`enable_speculative_decoding` C API even though the underlying
`libLiteRtLm.so` already exports it. Without the wiring, on-device
Gemma 4 MTP (multi-token prediction) cannot be turned on from Flutter.
The active fork also threads Gilbeot's LiteRT-LM LoRA sidecar path through
the FFI client; upstream 0.14.5 throws on `loraPath` in the FFI path.

## What's modified

Four inline edits to the upstream `DenisovAV/flutter_gemma` 0.14.5 source.
The first three enable Gemma 4 MTP (multi-token prediction); the fourth
threads a LiteRT-LM LoRA sidecar through the FFI client (upstream rejects
`loraPath` on the `.litertlm` FFI path).

1. **`native/litert_lm/include/engine.h`** — declare the MTP enable C
   function (the symbol is already exported by the shipped `.so`).

2. **`lib/core/ffi/litert_lm_bindings.dart`** — add the FFI binding for
   the MTP enable symbol.

3. **`lib/core/ffi/litert_lm_client.dart`** — add a `bool?
   enableSpeculativeDecoding` param to `initialize()`. Default behavior
   is `null → auto`: `enable` on GPU/NPU, `disable` on CPU. The CPU
   default is empirically chosen (FU105 in the dev_log) — on S10e CPU
   the drafter forward-pass overhead exceeds the acceptance gain
   because the LoRA sidecar shifts the target distribution away from
   the un-LoRA'd drafter.

4. **`lib/core/ffi/litert_lm_client.dart`** (separate region) — accept a
   `String? loraPath` on the chat / session start path, `dlopen` the
   Gilbeot-side native shim `libgilbeot_litertlm_lora.so`, and call its
   exported `gilbeot_litertlm_session_config_set_lora_path` to attach
   the LoRA sidecar to LiteRT-LM's opaque `SessionConfig` before the
   `ConversationConfig` snapshots it. Upstream's public C API has no
   setter for this, so the shim resolves LiteRT-LM's internal C++
   method `litert::lm::SessionConfig::SetScopedLoraFile` via `dlsym`
   into `libLiteRtLm.so`.

The patches are documented inline below as the "diff" — each block shows
the upstream context plus the lines to add. They're trivial to apply by
hand against `DenisovAV/flutter_gemma` at tag `v0.14.5` (or any 0.14.5-b
prebuilt from the same repo). Patch 4 also requires the matching
app-side native shim (CMake project at
`app/android/app/src/main/cpp/gilbeot_litertlm_lora_shim.cpp`), which
the app's own build wires in via `app/android/app/build.gradle` →
`externalNativeBuild`.

---

## Patch 1 — `native/litert_lm/include/engine.h`

After the existing `litert_lm_engine_settings_set_max_num_images`
declaration, **add**:

```c
// Enables Gemma 4 MTP / speculative decoding. The shipped libLiteRtLm.so
// (native-v0.10.2-b) already exports this symbol and bundles the MTP runtime;
// the header simply never declared it. Requires the model to contain the
// tf_lite_mtp_drafter + tf_lite_per_layer_embedder sections.
LITERT_LM_C_API_EXPORT
void litert_lm_engine_settings_set_enable_speculative_decoding(
    LiteRtLmEngineSettings* settings, bool enable_speculative_decoding);
```

## Patch 2 — `lib/core/ffi/litert_lm_bindings.dart`

After the existing `_litert_lm_engine_settings_set_max_num_images`
binding (the late-final pair), **add**:

```dart
void litert_lm_engine_settings_set_enable_speculative_decoding(
  ffi.Pointer<LiteRtLmEngineSettings> settings,
  bool enable_speculative_decoding,
) {
  return _litert_lm_engine_settings_set_enable_speculative_decoding(
    settings,
    enable_speculative_decoding,
  );
}

late final _litert_lm_engine_settings_set_enable_speculative_decodingPtr =
    _lookup<
            ffi.NativeFunction<
                ffi.Void Function(ffi.Pointer<LiteRtLmEngineSettings>,
                    ffi.Bool)>>(
        'litert_lm_engine_settings_set_enable_speculative_decoding');
late final _litert_lm_engine_settings_set_enable_speculative_decoding =
    _litert_lm_engine_settings_set_enable_speculative_decodingPtr.asFunction<
        void Function(ffi.Pointer<LiteRtLmEngineSettings>, bool)>();
```

## Patch 3 — `lib/core/ffi/litert_lm_client.dart`

In `Future<void> initialize({...})`, **add the named param** (after
`enableAudio`):

```dart
// Gemma 4 MTP / speculative decoding.
// null (default) = auto-decide based on backend: GPU/NPU → on
// (S23 GPU measured 1.5x decode speedup); CPU → off (S10e CPU A/B
// — drafter forward-pass overhead not offset by acceptance, net
// ~10-15s slowdown vs MTP-off).
bool? enableSpeculativeDecoding,
```

Then inside the `try { ... }` block where `LiteRtLmEngineSettings` is
configured, after `set_max_num_images`, **add**:

```dart
final enableSpec = enableSpeculativeDecoding ?? (backend != 'cpu');
b.litert_lm_engine_settings_set_enable_speculative_decoding(
    settings, enableSpec);
debugPrint(
    '[LiteRtLmFfi] speculative decoding (Gemma 4 MTP): $enableSpec '
    '(backend=$backend, override=$enableSpeculativeDecoding)');
```

---

## Patch 4 — `lib/core/ffi/litert_lm_client.dart` (LoRA sidecar)

This is a larger addition than the MTP wiring above (≈80 LoC). The
exact source lives in the vendored fork at
`third_party/flutter_gemma_repo/lib/core/ffi/litert_lm_client.dart`;
the sketch below shows the three regions to add.

### 4a — typedefs near the top of the file

After the existing `_ProxyFreeStringDart` typedef, **add**:

```dart
/// Experimental 길벗 Android shim:
/// attaches a LiteRT-LM LoRA sidecar to the opaque SessionConfig before the
/// ConversationConfig snapshots it. Upstream LiteRT-LM already has the C++
/// SetScopedLoraFile hook, but flutter_gemma's public C API has no setter yet.
typedef _SetLoraPathNative = Int32 Function(
    Pointer<LiteRtLmSessionConfig> config, Pointer<Char> loraPath);
typedef _SetLoraPathDart = int Function(
    Pointer<LiteRtLmSessionConfig> config, Pointer<Char> loraPath);
```

### 4b — fields on `LiteRtLmFfiClient` + `_ensureLoraShim()` helper

Inside `class LiteRtLmFfiClient { ... }`, **add** these fields next to
the existing `_proxyLib` / `_proxyCreate` group:

```dart
DynamicLibrary? _loraShimLib;
_SetLoraPathDart? _setLoraPath;
```

And the helper that lazily `dlopen`s our app-side shim:

```dart
void _ensureLoraShim() {
  if (!Platform.isAndroid) {
    throw UnsupportedError(
      'LoRA sidecar injection for .litertlm is currently wired only on '
      'Android, because it depends on the packaged gilbeot native shim.',
    );
  }
  if (_setLoraPath != null) return;
  try {
    final lib = DynamicLibrary.open('libgilbeot_litertlm_lora.so');
    _setLoraPath = lib.lookupFunction<_SetLoraPathNative, _SetLoraPathDart>(
      'gilbeot_litertlm_session_config_set_lora_path',
    );
    _loraShimLib = lib;
    debugPrint('[LiteRtLmFfi] Gilbeot LoRA shim loaded');
  } catch (e) {
    throw StateError(
      'Failed to load Gilbeot LiteRT-LM LoRA shim. '
      'Make sure app/android/app/src/main/cpp is built and packaged. '
      'Original error: $e',
    );
  }
}
```

### 4c — call site in the chat / session start path

In the chat / session start method, **accept** a `String? loraPath`
named param, and immediately before the `ConversationConfig` is
created from the `sessionConfig`, **add**:

```dart
Pointer<Utf8>? loraPathPtr;
if (loraPath != null && loraPath.isNotEmpty) {
  _ensureLoraShim();
  loraPathPtr = loraPath.toNativeUtf8();
  final rc = _setLoraPath!(sessionConfig, loraPathPtr.cast());
  if (rc != 0) {
    calloc.free(loraPathPtr);
    throw StateError(
      'LoRA sidecar attach failed '
      '(rc=$rc, loraPath=$loraPath)',
    );
  }
  debugPrint('[LiteRtLmFfi] LoRA sidecar attached: $loraPath');
}
// ... existing ConversationConfig creation ...
// In the matching cleanup (finally) block:
if (loraPathPtr != null) calloc.free(loraPathPtr);
```

### App-side native shim (not part of the fork)

Patch 4 needs the matching app-side native code, which lives in this
repo (not in the fork):

- **`app/android/app/src/main/cpp/gilbeot_litertlm_lora_shim.cpp`** —
  exports `gilbeot_litertlm_session_config_set_lora_path(...)` as a C
  function. Its body `dlopen`s `libLiteRtLm.so` (the app already loads
  it, so no extra cost), `dlsym`s LiteRT-LM's internal C++ method
  `litert::lm::SessionConfig::SetScopedLoraFile`, and calls it with a
  `std::shared_ptr<litert::ScopedFile>` wrapping a POSIX fd opened
  from `loraPath`. The locally-defined `ScopedFile`'s first field is
  the fd, matching LiteRT-LM's ABI so `mmap` works without
  modification.
- **`app/android/app/src/main/cpp/CMakeLists.txt`** —
  `add_library(gilbeot_litertlm_lora SHARED gilbeot_litertlm_lora_shim.cpp)`
  + links `android log dl`.
- **`app/android/app/build.gradle`** — `externalNativeBuild { cmake {
  path "src/main/cpp/CMakeLists.txt" } }` makes Gradle build and
  package `libgilbeot_litertlm_lora.so` into the APK's
  `lib/arm64-v8a/`.

The shim depends on LiteRT-LM's *internal* C++ method symbol surviving
the upstream `.so` build (it does, in `native-v0.10.2-b` and
`native-v0.11.0-b` prebuilts). If a future LiteRT-LM build hides that
symbol or mangles it differently, `ResolveSetScopedLoraFile()` will
log and return `nullptr`, and `_setLoraPath` will fail with a typed
error — the app still boots, but `loraPath` will not attach. The
upstream fix is to expose a public C API setter for the sidecar; we
will request that via the same PR as the MTP wiring.

---

## How to apply

1. Clone the fork base: `git clone https://github.com/DenisovAV/flutter_gemma`
   and check out tag `v0.14.5` (the underlying `libLiteRtLm.so` already has
   the MTP symbol exported — both `native-v0.10.2-b` and `native-v0.11.0-b`
   prebuilts work).
2. Apply the four additions above (Patches 1–3 are purely additive in the
   fork; Patch 4 adds the same fork-side regions plus the app-side native
   shim noted in section 4 above).
3. Point the gilbeot app's `pubspec.yaml` at the patched fork via:
   ```yaml
   dependency_overrides:
     flutter_gemma:
       path: ../third_party/flutter_gemma_repo
   ```

That's it. After `flutter pub get` the engine settings receive the
speculative-decoding flag at startup; on a model containing
`tf_lite_mtp_drafter` (the official Gemma 4 E2B `.litertlm`), MTP is
active.

## Background reading

- LiteRT-LM v0.11.0 release notes — Gemma 4 MTP support:
  https://github.com/google-ai-edge/LiteRT-LM/releases/tag/v0.11.0
- Speculative decoding on mobile (LiteRT-LM C++ API):
  https://ai.google.dev/edge/litert-lm/cpp
- Why the LoRA-drafter mismatch limits the speedup: see the dev_log
  highlights (FU103 — 1.5x measured vs Google's 2x claim, attributed to
  the drafter being trained on the base distribution while the target
  has a LoRA applied).
