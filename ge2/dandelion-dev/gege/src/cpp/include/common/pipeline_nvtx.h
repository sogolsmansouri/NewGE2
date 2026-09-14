#pragma once

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <string>

#if defined(GEGE_CUDA) && __has_include(<cuda_profiler_api.h>)
#include <cuda_profiler_api.h>
#define GEGE_CUDA_PROFILER_API_AVAILABLE 1
#else
#define GEGE_CUDA_PROFILER_API_AVAILABLE 0
#endif

#if defined(GEGE_CUDA) && __has_include(<nvtx3/nvToolsExt.h>)
#include <nvtx3/nvToolsExt.h>
#define GEGE_PIPELINE_NVTX_AVAILABLE 1
#elif defined(GEGE_CUDA) && __has_include(<nvToolsExt.h>)
#include <nvToolsExt.h>
#define GEGE_PIPELINE_NVTX_AVAILABLE 1
#else
#define GEGE_PIPELINE_NVTX_AVAILABLE 0
#endif

namespace gege::profiling {

inline bool pipelineNvtxEnabled() {
    static const bool enabled = [] {
        const char *raw = std::getenv("GEGE_PIPELINE_NVTX");
        if (raw == nullptr || raw[0] == '\0') {
            return false;
        }
        std::string value(raw);
        std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) {
            return static_cast<char>(std::tolower(ch));
        });
        return value != "0" && value != "false" && value != "off" && value != "no";
    }();
    return enabled;
}

class ManualRange {
   public:
    ManualRange() = default;
    explicit ManualRange(bool controls_profiler) : controls_profiler_(controls_profiler) {}
    explicit ManualRange(const char *name) { start(name); }

    ManualRange(const ManualRange &) = delete;
    ManualRange &operator=(const ManualRange &) = delete;

    ~ManualRange() { stop(); }

    void start(const char *name) {
        stop();
#if GEGE_PIPELINE_NVTX_AVAILABLE
        if (pipelineNvtxEnabled()) {
#if GEGE_CUDA_PROFILER_API_AVAILABLE
            if (controls_profiler_) {
                cudaProfilerStart();
            }
#endif
            nvtxRangePushA(name);
            active_ = true;
        }
#else
        (void)name;
#endif
    }

    void stop() {
#if GEGE_PIPELINE_NVTX_AVAILABLE
        if (active_) {
            nvtxRangePop();
#if GEGE_CUDA_PROFILER_API_AVAILABLE
            if (controls_profiler_) {
                cudaProfilerStop();
            }
#endif
            active_ = false;
        }
#endif
    }

    bool active() const { return active_; }

   private:
    bool active_ = false;
    bool controls_profiler_ = false;
};

class ScopedRange {
   public:
    explicit ScopedRange(const char *name) : range_(name) {}

    ScopedRange(const ScopedRange &) = delete;
    ScopedRange &operator=(const ScopedRange &) = delete;

   private:
    ManualRange range_;
};

}  // namespace gege::profiling
