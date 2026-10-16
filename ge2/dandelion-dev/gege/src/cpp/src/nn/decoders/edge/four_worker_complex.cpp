#include "nn/decoders/edge/four_worker_complex.h"
#include "nn/decoders/edge/complex.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <future>
#include <mutex>
#include <set>
#include <sstream>
#include <string>
#include <vector>
#include <c10/core/GradMode.h>
#ifdef GEGE_CUDA
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime_api.h>
#endif

namespace {
using Tensor = torch::Tensor;
using Scores = std::tuple<Tensor, Tensor, Tensor, Tensor>;

bool enabled(const char *key) {
    const auto *value = std::getenv(key);
    return value && std::string(value) != "0" && std::string(value) != "false" && std::string(value) != "False";
}

std::array<int, 4> worker_devices() {
    const auto *raw = std::getenv("GEGE_FOUR_WORKER_COMPLEX_DEVICES");
    TORCH_CHECK(raw && *raw, "GEGE_FOUR_WORKER_COMPLEX_DEVICES requires four distinct CUDA IDs");
    std::string text(raw), token;
    std::istringstream input(text);
    std::array<int, 4> result{};
    std::set<int> seen;
    size_t position = 0;
    while (std::getline(input, token, ',')) {
        TORCH_CHECK(position < result.size() && !token.empty() && token.find_first_not_of("0123456789") == std::string::npos,
                    "Invalid GEGE_FOUR_WORKER_COMPLEX_DEVICES: ", text);
        const auto device = std::stoll(token);
        TORCH_CHECK(device <= 127 && seen.insert(static_cast<int>(device)).second, "Four distinct CUDA IDs required");
        result[position++] = static_cast<int>(device);
    }
    TORCH_CHECK(position == 4 && text.back() != ',', "Exactly four CUDA worker IDs required");
    return result;
}

struct Witness {
    int64_t call_id, batch_id, owner_device, worker_slot, worker_device;
    int64_t chunk_start, chunk_count, chunk_width, positive_start, positive_rows, score_rows;
};

class Trace {
    std::mutex mutex_;
    std::ofstream stream_;
public:
    Trace() {
        const auto *value = std::getenv("GEGE_FOUR_WORKER_COMPLEX_TRACE");
        if (!value || !*value) return;
        std::filesystem::path path(value);
        TORCH_CHECK(!std::filesystem::exists(path), "Four-worker trace already exists: ", path.string());
        stream_.open(path, std::ios::out | std::ios::binary);
        TORCH_CHECK(bool(stream_), "Cannot open four-worker trace: ", path.string());
    }
    bool active() const { return stream_.is_open(); }
    void record(const char *event, const Witness &w, int64_t gradient_device = -1) {
        if (!active()) return;
        std::lock_guard<std::mutex> lock(mutex_);
        stream_ << "{\"schema\":\"pipege_four_worker_complex_trace_v1\",\"event\":\"" << event
                << "\",\"call_id\":" << w.call_id << ",\"batch_id\":" << w.batch_id
                << ",\"owner_device\":" << w.owner_device << ",\"worker_slot\":" << w.worker_slot
                << ",\"worker_device\":" << w.worker_device << ",\"chunk_start\":" << w.chunk_start
                << ",\"chunk_count\":" << w.chunk_count << ",\"chunk_width\":" << w.chunk_width
                << ",\"positive_start\":" << w.positive_start << ",\"positive_rows\":" << w.positive_rows
                << ",\"score_rows\":" << w.score_rows;
        if (gradient_device >= 0) stream_ << ",\"gradient_device\":" << gradient_device;
        stream_ << "}\n";
        stream_.flush();
        TORCH_CHECK(bool(stream_), "Cannot write four-worker trace");
    }
};

Trace &trace() { static Trace instance; return instance; }
std::atomic<int64_t> next_call{0};

Tensor pad_rows(const Tensor &value, int64_t rows) {
    TORCH_CHECK(rows >= value.size(0), "Four-worker padding would remove positive rows");
    if (rows == value.size(0)) return value;
    if (value.dim() == 1) return torch::nn::functional::pad(value, torch::nn::functional::PadFuncOptions({0, rows - value.size(0)}));
    return torch::nn::functional::pad(value, torch::nn::functional::PadFuncOptions({0, 0, 0, rows - value.size(0)}));
}

#ifdef GEGE_CUDA
void record_use(const Tensor &value, c10::cuda::CUDAStream stream) {
    TORCH_CHECK(value.is_cuda() && value.device().index() == stream.device_index(),
                "Four-worker tensor lifetime must be recorded on its device");
    c10::cuda::CUDACachingAllocator::recordStream(value.storage().data_ptr(), stream);
}
#endif
}  // namespace

bool four_worker_complex_requested() {
    return std::getenv("GEGE_FOUR_WORKER_COMPLEX_DEVICES") != nullptr;
}

Scores four_worker_complex_forward(std::shared_ptr<EdgeDecoder> decoder, Tensor edges, Tensor nodes,
                                   Tensor dst_mapping, Tensor src_mapping, int64_t batch_id,
                                   std::function<void()> post_gather_callback) {
#ifndef GEGE_CUDA
    TORCH_CHECK(false, "Four-worker ComplEx requires a CUDA-enabled build");
#else
    TORCH_CHECK(four_worker_complex_requested(), "Four-worker ComplEx was called without its opt-in setting");
    TORCH_CHECK(std::dynamic_pointer_cast<ComplEx>(decoder) && decoder->use_inverse_relations_ &&
                    decoder->decoder_method_ == EdgeDecoderMethod::CORRUPT_NODE,
                "Four-worker scoring requires inverse ComplEx node corruption");
    TORCH_CHECK(c10::GradMode::is_enabled() && nodes.defined() && nodes.requires_grad() && nodes.is_cuda() &&
                    nodes.dim() == 2 && nodes.scalar_type() == torch::kFloat32 && nodes.size(1) > 0 && nodes.size(1) % 2 == 0,
                "Four-worker scoring requires differentiable CUDA float32 embeddings of even width");
    TORCH_CHECK(edges.defined() && edges.device() == nodes.device() && edges.scalar_type() == torch::kInt64 &&
                    edges.dim() == 2 && edges.size(1) == 3 && edges.size(0) > 0,
                "Four-worker scoring requires nonempty owner-local int64 triples");
    TORCH_CHECK(dst_mapping.defined() && src_mapping.defined() && dst_mapping.device() == nodes.device() &&
                    src_mapping.device() == nodes.device() && dst_mapping.scalar_type() == torch::kInt64 &&
                    src_mapping.scalar_type() == torch::kInt64 && dst_mapping.dim() == 2 &&
                    dst_mapping.size(0) == 50 && dst_mapping.size(1) > 0 && src_mapping.sizes() == dst_mapping.sizes(),
                "Four-worker scoring requires the original two 50-chunk negative matrices");
    TORCH_CHECK(decoder->relations_.device() == nodes.device() && decoder->inverse_relations_.device() == nodes.device() &&
                    decoder->relations_.requires_grad() && decoder->inverse_relations_.requires_grad(),
                "Four-worker relation leaves must remain on the owner device");
    TORCH_CHECK(!enabled("GEGE_CSR_GATHER") && !enabled("GEGE_DOT_COMPARE_CUDA_ELEMENTWISE"),
                "Four-worker scoring currently requires native index-select gathers and bmm dot scoring");

    const auto devices = worker_devices();
    int device_count = 0;
    AT_CUDA_CHECK(cudaGetDeviceCount(&device_count));
    for (auto device : devices) TORCH_CHECK(device < device_count, "Unavailable four-worker CUDA device: ", device);
    const auto owner = nodes.device();
    TORCH_CHECK(std::find(devices.begin(), devices.end(), owner.index()) != devices.end(), "Owner must belong to the four compute workers");
    c10::cuda::CUDAGuard owner_guard(owner);
    const auto owner_stream = c10::cuda::getCurrentCUDAStream(owner.index());
    const int64_t rows = edges.size(0), chunks = dst_mapping.size(0), negatives = dst_mapping.size(1), width = nodes.size(1);
    const int64_t chunk_width = (rows + chunks - 1) / chunks;

    // Keep one original owner-side gather per tensor. Slice/copy backward
    // combines worker contributions before the original gather reductions.
    auto src = nodes.index_select(0, edges.select(1, 0));
    auto dst = nodes.index_select(0, edges.select(1, 2));
    auto forward_relations = decoder->select_relations(edges.select(1, 1), false);
    auto inverse_relations = decoder->select_relations(edges.select(1, 1), true);
    auto dst_negatives = nodes.index_select(0, dst_mapping.flatten()).reshape({chunks, negatives, width});
    auto src_negatives = nodes.index_select(0, src_mapping.flatten()).reshape({chunks, negatives, width});
    if (post_gather_callback) post_gather_callback();
    // The owner may be running on a nondefault trainer stream. In particular,
    // a same-device .to() is an identity and establishes no dependency on the
    // separate worker stream. Finish the original gathers before staging any
    // worker inputs; this conservative fence is deliberate for correctness.
    AT_CUDA_CHECK(cudaStreamSynchronize(owner_stream.stream()));
    auto &witness_trace = trace();
    const auto call_id = next_call.fetch_add(1);

    struct Work {
        Witness witness;
        Tensor src, dst, forward_relations, inverse_relations, dst_negatives, src_negatives;
        c10::cuda::CUDAStream stream;
        explicit Work(c10::cuda::CUDAStream value) : stream(value) {}
    };
    std::vector<Work> work;
    work.reserve(4);
    int64_t first_chunk = 0;
    // Stage all input copies before launching score kernels. Independent CUDA
    // streams prevent owner-worker compute from serializing later input copies.
    for (int slot = 0; slot < 4; ++slot) {
        const int64_t count = chunks / 4 + (slot < chunks % 4 ? 1 : 0);
        const int64_t first_row = first_chunk * chunk_width;
        const int64_t actual_start = std::min(first_row, rows);
        const int64_t actual_rows = std::max<int64_t>(0, std::min(rows, (first_chunk + count) * chunk_width) - actual_start);
        const auto device = torch::Device(torch::kCUDA, devices[slot]);
        work.emplace_back(c10::cuda::getStreamFromPool(false, devices[slot]));
        auto &part = work.back();
        part.witness = {call_id, batch_id, owner.index(), slot, devices[slot], first_chunk, count,
                        chunk_width, actual_start, actual_rows, count * chunk_width};
        c10::cuda::CUDAStreamGuard worker_guard(part.stream);
        part.src = src.narrow(0, actual_start, actual_rows).to(device);
        part.dst = dst.narrow(0, actual_start, actual_rows).to(device);
        part.forward_relations = forward_relations.narrow(0, actual_start, actual_rows).to(device);
        part.inverse_relations = inverse_relations.narrow(0, actual_start, actual_rows).to(device);
        part.dst_negatives = dst_negatives.narrow(0, first_chunk, count).to(device);
        part.src_negatives = src_negatives.narrow(0, first_chunk, count).to(device);
        // For the owner worker, .to(device) is an identity: these views still
        // use storage allocated on the trainer stream. Saved tensors can lose
        // their last reference as soon as backward kernels are enqueued, so a
        // forward readiness fence alone cannot protect their later lifetime.
        // Record the consumer stream for both alias and copied input storage.
        for (const auto &input : {part.src, part.dst, part.forward_relations,
                                  part.inverse_relations, part.dst_negatives, part.src_negatives}) {
            record_use(input, part.stream);
        }
        first_chunk += count;
    }
    TORCH_CHECK(first_chunk == chunks, "Four-worker chunk coverage differs");

    std::array<std::future<Scores>, 4> pending;
    for (int slot = 0; slot < 4; ++slot) {
        pending[slot] = std::async(std::launch::async, [&, slot] {
            c10::AutoGradMode gradient_mode(true);
            c10::cuda::setCurrentCUDAStream(owner_stream);
            auto &part = work[slot];
            c10::cuda::CUDAStreamGuard worker_guard(part.stream);
            ComplexHadamardOperator relation;
            auto adjusted_src = relation(part.src, part.forward_relations);
            auto adjusted_dst = relation(part.dst, part.inverse_relations);
            auto positive = pad_rows((adjusted_src * part.dst).sum(-1), part.witness.score_rows);
            auto inverse_positive = pad_rows((adjusted_dst * part.src).sum(-1), part.witness.score_rows);
            auto queries = pad_rows(adjusted_src, part.witness.score_rows).reshape({part.witness.chunk_count, chunk_width, width});
            auto inverse_queries = pad_rows(adjusted_dst, part.witness.score_rows).reshape({part.witness.chunk_count, chunk_width, width});
            auto negative = queries.bmm(part.dst_negatives.transpose(1, 2)).flatten(0, 1);
            auto inverse_negative = inverse_queries.bmm(part.src_negatives.transpose(1, 2)).flatten(0, 1);
            if (witness_trace.active()) {
                const auto witness = part.witness;
                negative.register_hook([witness](Tensor gradient) {
                    TORCH_CHECK(gradient.is_cuda() && gradient.device().index() == witness.worker_device,
                                "Four-worker backward gradient returned on the wrong device");
                    trace().record("backward_seen", witness, gradient.device().index());
                    return gradient;
                });
            }
            Scores scores{positive.to(owner), negative.to(owner), inverse_positive.to(owner), inverse_negative.to(owner)};
            AT_CUDA_CHECK(cudaStreamSynchronize(part.stream.stream()));
            AT_CUDA_CHECK(cudaStreamSynchronize(owner_stream.stream()));
            witness_trace.record("forward_done", part.witness);
            return scores;
        });
    }
    std::array<std::vector<Tensor>, 4> output;
    for (auto &future : pending) {
        auto scores = future.get();
        // Owner-worker scores also remain on their pool stream because .to()
        // is an identity. The concatenations below consume them asynchronously
        // on the trainer stream after these temporary tensor handles expire.
        record_use(std::get<0>(scores), owner_stream);
        record_use(std::get<1>(scores), owner_stream);
        record_use(std::get<2>(scores), owner_stream);
        record_use(std::get<3>(scores), owner_stream);
        output[0].push_back(std::get<0>(scores));
        output[1].push_back(std::get<1>(scores));
        output[2].push_back(std::get<2>(scores));
        output[3].push_back(std::get<3>(scores));
    }
    return {torch::cat(output[0], 0), torch::cat(output[1], 0), torch::cat(output[2], 0), torch::cat(output[3], 0)};
#endif
}
