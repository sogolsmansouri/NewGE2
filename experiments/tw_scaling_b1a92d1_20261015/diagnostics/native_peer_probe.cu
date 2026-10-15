#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

// Standalone CUDA-runtime probe: raw cudaMalloc allocations, no Torch or RNG.
// Each GPU holds two buffers; --bytes is capped at 32 MiB (64 MiB per GPU).
// Run kernel modes in separate processes: a bad peer mapping can poison a context.
__global__ void peer_copy_words(uint32_t *dst, const uint32_t *src, size_t n) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += static_cast<size_t>(blockDim.x) * gridDim.x) dst[i] = src[i];
}

static std::string quote(const std::string &s) {
    std::ostringstream o; o << '"';
    for (unsigned char c : s) {
        if (c == '"' || c == '\\') o << '\\' << c;
        else if (c == '\n') o << "\\n";
        else if (c == '\r') o << "\\r";
        else if (c == '\t') o << "\\t";
        else if (c < 32) o << '?';
        else o << c;
    }
    o << '"'; return o.str();
}
struct Calls {
    std::vector<std::pair<std::string, cudaError_t>> items;
    bool ok = true;
    cudaError_t add(const char *name, cudaError_t rc) {
        items.emplace_back(name, rc); ok = ok && (rc == cudaSuccess); return rc;
    }
    std::string json() const {
        std::ostringstream o; o << '[';
        for (size_t i = 0; i < items.size(); ++i) {
            if (i) o << ',';
            o << "{\"call\":" << quote(items[i].first) << ",\"code\":"
              << static_cast<int>(items[i].second) << ",\"name\":"
              << quote(cudaGetErrorName(items[i].second)) << '}';
        }
        o << ']'; return o.str();
    }
};
static bool require(cudaError_t rc, const char *where) {
    if (rc == cudaSuccess) return true;
    std::cout << "{\"kind\":\"fatal\",\"call\":" << quote(where)
              << ",\"code\":" << static_cast<int>(rc) << ",\"name\":"
              << quote(cudaGetErrorName(rc)) << ",\"message\":"
              << quote(cudaGetErrorString(rc)) << "}" << std::endl;
    return false;
}
static int64_t mismatches(const std::vector<uint32_t> &a,
                          const std::vector<uint32_t> &b) {
    int64_t n = 0; for (size_t i = 0; i < a.size(); ++i) n += a[i] != b[i]; return n;
}
static uint32_t pattern(size_t index, uint32_t salt) {
    uint32_t x = static_cast<uint32_t>(index) + salt * 0x9e3779b9u;
    x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15; x *= 0x846ca68bu; x ^= x >> 16;
    return x == 0xa5a5a5a5u ? 0xa5a5a5a4u : x;
}

int main(int argc, char **argv) {
    size_t bytes = 4u * 1024 * 1024; int repeats = 2;
    std::string peer = "enabled", mode = "copies";
    for (int i = 1; i < argc; ++i) {
        std::string key = argv[i];
        if (key == "--help") {
            std::cout << "--peer enabled|disabled --mode copies|kernel-read|kernel-write "
                         "--bytes N (multiple of 4, <=33554432) --repeats N (1..10)\n";
            return 0;
        }
        if (++i >= argc) { std::cerr << "Missing argument\n"; return 2; }
        try {
            if (key == "--peer") peer = argv[i];
            else if (key == "--mode") mode = argv[i];
            else if (key == "--bytes") bytes = std::stoull(argv[i]);
            else if (key == "--repeats") repeats = std::stoi(argv[i]);
            else { std::cerr << "Unknown option\n"; return 2; }
        } catch (...) { std::cerr << "Bad argument\n"; return 2; }
    }
    if ((peer != "enabled" && peer != "disabled") ||
        (mode != "copies" && mode != "kernel-read" && mode != "kernel-write") ||
        bytes < 4 || bytes % 4 || bytes > 32u * 1024 * 1024 || repeats < 1 || repeats > 10 ||
        (peer == "disabled" && mode != "copies")) {
        std::cerr << "Invalid arguments (kernel modes require --peer enabled)\n"; return 2;
    }
    int count = 0, driver = 0, runtime = 0;
    if (!require(cudaGetDeviceCount(&count), "cudaGetDeviceCount") || count < 2) return 2;
    if (!require(cudaDriverGetVersion(&driver), "cudaDriverGetVersion") ||
        !require(cudaRuntimeGetVersion(&runtime), "cudaRuntimeGetVersion")) return 2;
    const char *visible = std::getenv("CUDA_VISIBLE_DEVICES");
    std::cout << "{\"kind\":\"environment\",\"driver\":" << driver
              << ",\"runtime\":" << runtime << ",\"device_count\":" << count
              << ",\"peer_mode\":" << quote(peer) << ",\"mode\":" << quote(mode)
              << ",\"bytes_per_buffer\":" << bytes << ",\"device_bytes_per_gpu\":" << bytes * 2
              << ",\"repeats\":" << repeats << ",\"CUDA_VISIBLE_DEVICES\":"
              << (visible ? quote(visible) : "null")
              << ",\"route_claim\":" << quote(peer == "enabled" ?
                  "peer access enabled; copy APIs request peer transfers; no hardware route instrumentation" :
                  "peer access disabled; driver-selected routing; success is not evidence of direct P2P")
              << "}" << std::endl;
    cudaDeviceProp props[2]{};
    for (int d = 0; d < 2; ++d) {
        char pci[32]{};
        if (!require(cudaGetDeviceProperties(&props[d], d), "cudaGetDeviceProperties") ||
            !require(cudaDeviceGetPCIBusId(pci, sizeof(pci), d), "cudaDeviceGetPCIBusId")) return 2;
        std::cout << "{\"kind\":\"device\",\"device\":" << d << ",\"name\":"
                  << quote(props[d].name) << ",\"pci_bus_id\":" << quote(pci)
                  << ",\"unified_addressing\":" << props[d].unifiedAddressing << "}" << std::endl;
        if (!require(cudaSetDevice(d), "cudaSetDevice(peer setup)")) return 2;
        int can = 0;
        if (!require(cudaDeviceCanAccessPeer(&can, d, 1 - d), "cudaDeviceCanAccessPeer")) return 2;
        auto rc = peer == "enabled" ? cudaDeviceEnablePeerAccess(1 - d, 0) :
                                      cudaDeviceDisablePeerAccess(1 - d);
        std::cout << "{\"kind\":\"peer_setup\",\"device\":" << d << ",\"peer\":" << 1 - d
                  << ",\"can_access_peer\":" << can << ",\"operation\":" << quote(peer)
                  << ",\"code\":" << static_cast<int>(rc) << ",\"name\":"
                  << quote(cudaGetErrorName(rc)) << "}" << std::endl;
        bool allowed = rc == cudaSuccess ||
            (peer == "enabled" && rc == cudaErrorPeerAccessAlreadyEnabled) ||
            (peer == "disabled" && rc == cudaErrorPeerAccessNotEnabled);
        if (!allowed) { require(rc, "peer setup"); return 2; }
        // Clear the expected already-enabled/not-enabled status before testing.
        (void)cudaGetLastError();
    }
    uint32_t *buffers[2][2]{}; cudaStream_t streams[2]{};
    for (int d = 0; d < 2; ++d) {
        if (!require(cudaSetDevice(d), "cudaSetDevice(allocate)") ||
            !require(cudaMalloc(&buffers[d][0], bytes), "cudaMalloc(source)") ||
            !require(cudaMalloc(&buffers[d][1], bytes), "cudaMalloc(destination)") ||
            !require(cudaStreamCreateWithFlags(&streams[d], cudaStreamNonBlocking), "cudaStreamCreateWithFlags")) return 2;
    }
    const size_t n = bytes / sizeof(uint32_t);
    std::vector<uint32_t> expected(n), actual(n), source_after(n), staging(n), sentinel(n, 0xa5a5a5a5u);
    int total = 0, passed_count = 0, api_failures = 0;
    bool poisoned = false;
    std::vector<std::string> methods = {"h2d_d2h_control", "self_copy_control", "host_staged_control"};
    if (mode == "copies") {
        for (const auto *method : {"peer_sync_destination_current", "peer_sync_source_current",
                "peer_async_destination_stream", "peer_async_source_stream",
                "default_sync_destination_current", "d2d_sync_destination_current",
                "default_async_destination_stream", "d2d_async_destination_stream"}) methods.emplace_back(method);
    } else methods.push_back(mode);
    for (const auto &method : methods) for (int source = 0; source < 2; ++source)
      for (int repeat = 0; repeat < repeats; ++repeat) {
        if (poisoned) break;
        const bool control_local = method == "h2d_d2h_control" || method == "self_copy_control";
        const int destination = control_local ? source : 1 - source;
        ++total;
        for (size_t i = 0; i < n; ++i) expected[i] = pattern(i, static_cast<uint32_t>(total));
        Calls calls;
        calls.add("set_source", cudaSetDevice(source));
        calls.add("source_h2d", cudaMemcpy(buffers[source][0], expected.data(), bytes, cudaMemcpyHostToDevice));
        calls.add("source_ready", cudaDeviceSynchronize());
        calls.add("source_readback_before", cudaMemcpy(source_after.data(), buffers[source][0], bytes, cudaMemcpyDeviceToHost));
        int64_t source_before_bad = calls.ok ? mismatches(source_after, expected) : -1;
        calls.add("set_destination", cudaSetDevice(destination));
        calls.add("destination_sentinel", cudaMemset(buffers[destination][1], 0xa5, bytes));
        calls.add("destination_ready", cudaDeviceSynchronize());
        calls.add("sentinel_readback", cudaMemcpy(actual.data(), buffers[destination][1], bytes, cudaMemcpyDeviceToHost));
        int64_t sentinel_bad = calls.ok ? mismatches(actual, sentinel) : -1;
        if (!calls.ok) {
            std::cout << "{\"kind\":\"case_setup_failure\",\"method\":" << quote(method)
                      << ",\"source\":" << source << ",\"destination\":" << destination
                      << ",\"calls\":" << calls.json() << "}" << std::endl;
            poisoned = true; ++api_failures; break;
        }
        int current = destination;
        if (method == "peer_sync_source_current" || method == "peer_async_source_stream" || mode == "kernel-write") current = source;
        calls.add("set_copy_device", cudaSetDevice(current));
        const auto start = std::chrono::steady_clock::now();
        if (method == "h2d_d2h_control") {
            calls.add("control_destination_h2d", cudaMemcpy(buffers[destination][1], expected.data(), bytes, cudaMemcpyHostToDevice));
        } else if (method == "self_copy_control") {
            calls.add("self_d2d", cudaMemcpy(buffers[destination][1], buffers[source][0], bytes, cudaMemcpyDeviceToDevice));
        } else if (method == "host_staged_control") {
            calls.add("set_stage_source", cudaSetDevice(source));
            calls.add("stage_d2h", cudaMemcpy(staging.data(), buffers[source][0], bytes, cudaMemcpyDeviceToHost));
            calls.add("set_stage_destination", cudaSetDevice(destination));
            calls.add("stage_h2d", cudaMemcpy(buffers[destination][1], staging.data(), bytes, cudaMemcpyHostToDevice));
            current = destination;
        } else if (method == "peer_sync_destination_current" || method == "peer_sync_source_current") {
            calls.add("cudaMemcpyPeer", cudaMemcpyPeer(buffers[destination][1], destination, buffers[source][0], source, bytes));
        } else if (method == "peer_async_destination_stream" || method == "peer_async_source_stream") {
            calls.add("cudaMemcpyPeerAsync", cudaMemcpyPeerAsync(buffers[destination][1], destination, buffers[source][0], source, bytes, streams[current]));
            calls.add("copy_stream_sync", cudaStreamSynchronize(streams[current]));
        } else if (method == "default_sync_destination_current" || method == "d2d_sync_destination_current") {
            calls.add("cudaMemcpy", cudaMemcpy(buffers[destination][1], buffers[source][0], bytes,
                method == "default_sync_destination_current" ? cudaMemcpyDefault : cudaMemcpyDeviceToDevice));
        } else if (method == "default_async_destination_stream" || method == "d2d_async_destination_stream") {
            calls.add("cudaMemcpyAsync", cudaMemcpyAsync(buffers[destination][1], buffers[source][0], bytes,
                method == "default_async_destination_stream" ? cudaMemcpyDefault : cudaMemcpyDeviceToDevice, streams[current]));
            calls.add("copy_stream_sync", cudaStreamSynchronize(streams[current]));
        } else {
            peer_copy_words<<<std::min<size_t>((n + 255) / 256, 1024), 256, 0, streams[current]>>>(buffers[destination][1], buffers[source][0], n);
            calls.add("peer_kernel_launch", cudaGetLastError());
            calls.add("peer_kernel_stream_sync", cudaStreamSynchronize(streams[current]));
        }
        calls.add("set_source_after", cudaSetDevice(source));
        calls.add("source_sync_after", cudaDeviceSynchronize());
        calls.add("set_destination_after", cudaSetDevice(destination));
        calls.add("destination_sync_after", cudaDeviceSynchronize());
        const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        bool destination_read_ok = calls.add("destination_readback", cudaMemcpy(actual.data(), buffers[destination][1], bytes, cudaMemcpyDeviceToHost)) == cudaSuccess;
        calls.add("set_source_readback", cudaSetDevice(source));
        bool source_read_ok = calls.add("source_readback_after", cudaMemcpy(source_after.data(), buffers[source][0], bytes, cudaMemcpyDeviceToHost)) == cudaSuccess;
        int64_t bad = destination_read_ok ? mismatches(actual, expected) : -1;
        int64_t source_bad = source_read_ok ? mismatches(source_after, expected) : -1;
        int64_t retained_sentinel = destination_read_ok ? std::count(actual.begin(), actual.end(), 0xa5a5a5a5u) : -1;
        bool passed = calls.ok && source_before_bad == 0 && sentinel_bad == 0 && bad == 0 && source_bad == 0;
        if (passed) ++passed_count;
        if (!calls.ok) { ++api_failures; poisoned = true; }
        std::cout << "{\"kind\":\"case\",\"method\":" << quote(method)
                  << ",\"source\":" << source << ",\"destination\":" << destination
                  << ",\"repeat\":" << repeat << ",\"copy_current_device\":" << current
                  << ",\"bytes\":" << bytes << ",\"words\":" << n << ",\"seconds\":" << seconds
                  << ",\"source_before_mismatches\":" << source_before_bad << ",\"sentinel_before_mismatches\":" << sentinel_bad
                  << ",\"destination_mismatches\":" << bad << ",\"source_after_mismatches\":" << source_bad
                  << ",\"destination_sentinel_words\":" << retained_sentinel
                  << ",\"passed\":" << (passed ? "true" : "false") << ",\"calls\":" << calls.json() << "}" << std::endl;
      }
    const int expected_cases = static_cast<int>(methods.size()) * 2 * repeats;
    bool cleanup_ok = true;
    if (!poisoned) for (int d = 0; d < 2; ++d) {
        cleanup_ok &= require(cudaSetDevice(d), "cudaSetDevice(cleanup)");
        cleanup_ok &= require(cudaStreamDestroy(streams[d]), "cudaStreamDestroy");
        cleanup_ok &= require(cudaFree(buffers[d][0]), "cudaFree(source)");
        cleanup_ok &= require(cudaFree(buffers[d][1]), "cudaFree(destination)");
    }
    bool all_passed = passed_count == expected_cases && cleanup_ok;
    std::cout << "{\"kind\":\"summary\",\"peer_mode\":" << quote(peer) << ",\"mode\":" << quote(mode)
              << ",\"expected_cases\":" << expected_cases << ",\"completed_cases\":" << total
              << ",\"passed_cases\":" << passed_count << ",\"api_failure_cases\":" << api_failures
              << ",\"context_error_stopped_remaining\":" << (poisoned ? "true" : "false")
              << ",\"all_passed\":" << (all_passed ? "true" : "false") << "}" << std::endl;
    return all_passed ? 0 : 1;
}
