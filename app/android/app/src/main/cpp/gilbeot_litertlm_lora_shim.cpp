#include <android/log.h>
#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <unistd.h>

#include <memory>
#include <string>

#define GILBEOT_LOG_TAG "GilbeotLoRAShim"
#define GILBEOT_LOGE(...) \
  __android_log_print(ANDROID_LOG_ERROR, GILBEOT_LOG_TAG, __VA_ARGS__)
#define GILBEOT_LOGI(...) \
  __android_log_print(ANDROID_LOG_INFO, GILBEOT_LOG_TAG, __VA_ARGS__)

extern "C" struct LiteRtLmSessionConfig;

namespace litert {

// ABI mirror of LiteRT-LM runtime/util/scoped_file.h. The runtime stores and
// later consumes this through std::shared_ptr<litert::ScopedFile>. Keeping the
// first field as the POSIX fd preserves the layout LiteRT-LM uses for mmap.
class ScopedFile {
 public:
  explicit ScopedFile(int fd) : fd_(fd) {}
  ScopedFile(const ScopedFile&) = delete;
  ScopedFile& operator=(const ScopedFile&) = delete;
  ~ScopedFile() {
    if (fd_ >= 0) {
      close(fd_);
    }
  }

  int fd() const { return fd_; }
  int get() const { return fd_; }

 private:
  int fd_ = -1;
};

namespace lm {
class SessionConfig;
}  // namespace lm

}  // namespace litert

namespace {

using SetScopedLoraFileFn = void (*)(
    litert::lm::SessionConfig*,
    std::shared_ptr<litert::ScopedFile>);
using GetScopedLoraFileFn =
    std::shared_ptr<litert::ScopedFile> (*)(const litert::lm::SessionConfig*);

SetScopedLoraFileFn ResolveSetScopedLoraFile() {
  static SetScopedLoraFileFn fn = []() -> SetScopedLoraFileFn {
    void* handle = dlopen("libLiteRtLm.so", RTLD_NOW | RTLD_GLOBAL);
    if (handle == nullptr) {
      GILBEOT_LOGE("dlopen(libLiteRtLm.so) failed: %s", dlerror());
      return nullptr;
    }

    // litert::lm::SessionConfig::SetScopedLoraFile(
    //   std::__ndk1::shared_ptr<litert::ScopedFile>)
    constexpr const char* kSymbol =
        "_ZN6litert2lm13SessionConfig17SetScopedLoraFile"
        "ENSt6__ndk110shared_ptrINS_10ScopedFileEEE";
    void* raw = dlsym(handle, kSymbol);
    if (raw == nullptr) {
      GILBEOT_LOGE("dlsym(SetScopedLoraFile) failed: %s", dlerror());
      return nullptr;
    }
    GILBEOT_LOGI("resolved SessionConfig::SetScopedLoraFile");
    return reinterpret_cast<SetScopedLoraFileFn>(raw);
  }();
  return fn;
}

GetScopedLoraFileFn ResolveGetScopedLoraFile() {
  static GetScopedLoraFileFn fn = []() -> GetScopedLoraFileFn {
    void* handle = dlopen("libLiteRtLm.so", RTLD_NOW | RTLD_GLOBAL);
    if (handle == nullptr) {
      GILBEOT_LOGE("dlopen(libLiteRtLm.so) failed: %s", dlerror());
      return nullptr;
    }

    constexpr const char* kSymbol =
        "_ZNK6litert2lm13SessionConfig17GetScopedLoraFileEv";
    void* raw = dlsym(handle, kSymbol);
    if (raw == nullptr) {
      GILBEOT_LOGE("dlsym(GetScopedLoraFile) failed: %s", dlerror());
      return nullptr;
    }
    GILBEOT_LOGI("resolved SessionConfig::GetScopedLoraFile");
    return reinterpret_cast<GetScopedLoraFileFn>(raw);
  }();
  return fn;
}

litert::lm::SessionConfig* GetInnerSessionConfig(
    LiteRtLmSessionConfig* c_config) {
  if (c_config == nullptr) {
    return nullptr;
  }
  // C API struct is:
  //   struct LiteRtLmSessionConfig { std::unique_ptr<SessionConfig> config; };
  // libc++ unique_ptr with default deleter stores the managed pointer in the
  // first word, so the first word of the opaque C object is SessionConfig*.
  return *reinterpret_cast<litert::lm::SessionConfig**>(c_config);
}

}  // namespace

extern "C" __attribute__((visibility("default"))) int
gilbeot_litertlm_session_config_set_lora_path(
    LiteRtLmSessionConfig* c_config,
    const char* lora_path) {
  if (lora_path == nullptr || lora_path[0] == '\0') {
    GILBEOT_LOGE("empty lora_path");
    return -1;
  }

  auto* session_config = GetInnerSessionConfig(c_config);
  if (session_config == nullptr) {
    GILBEOT_LOGE("LiteRtLmSessionConfig has no inner SessionConfig");
    return -2;
  }

  int fd = open(lora_path, O_RDONLY | O_CLOEXEC);
  if (fd < 0) {
    GILBEOT_LOGE("open(%s) failed: errno=%d", lora_path, errno);
    return -3;
  }

  auto set_lora = ResolveSetScopedLoraFile();
  if (set_lora == nullptr) {
    close(fd);
    return -4;
  }

  auto scoped_file = std::make_shared<litert::ScopedFile>(fd);
  set_lora(session_config, scoped_file);

  auto get_lora = ResolveGetScopedLoraFile();
  if (get_lora != nullptr) {
    auto attached = get_lora(session_config);
    if (attached) {
      GILBEOT_LOGI("LoRA sidecar attached: %s (verified fd=%d)",
                   lora_path, attached->fd());
    } else {
      GILBEOT_LOGE("SetScopedLoraFile returned but GetScopedLoraFile is empty");
      return -5;
    }
  } else {
    GILBEOT_LOGI("LoRA sidecar attached: %s (verification unavailable)",
                 lora_path);
  }
  return 0;
}
