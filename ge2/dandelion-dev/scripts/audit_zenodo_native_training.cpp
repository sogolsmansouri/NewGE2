// Standalone diagnostic linked to the original GE2 library, not PipeGE.
#include "gege.h"
#include "nn/decoders/edge/edge_decoder.h"
#include <fstream>
#include <iostream>

namespace {
int checks = 0;
double worst = 0;

void equal(const std::string& label, torch::Tensor a, torch::Tensor b,
           double rtol = 2e-4, double atol = 2e-5) {
    checks++;
    TORCH_CHECK(a.sizes() == b.sizes(), label, ": shape mismatch");
    double error = (a.detach().to(torch::kCPU) - b.detach().to(torch::kCPU)).abs().max().item<double>();
    worst = std::max(worst, error);
    TORCH_CHECK(torch::allclose(a, b.to(a.device()), rtol, atol), label, ": max abs error ", error);
}

torch::Tensor score(torch::Tensor h, torch::Tensor r, torch::Tensor t, bool complex, bool dot) {
    if (dot) return (h * t).sum(-1);
    if (!complex) return (h * r * t).sum(-1);
    auto hh = h.chunk(2, -1), rr = r.chunk(2, -1), tt = t.chunk(2, -1);
    return (hh[0] * rr[0] * tt[0] + hh[1] * rr[0] * tt[1]
            + hh[0] * rr[1] * tt[1] - hh[1] * rr[1] * tt[0]).sum(-1);
}

torch::Tensor keys(torch::Tensor edges, int64_t nodes, int64_t relations) {
    edges = edges.to(torch::kCPU).to(torch::kInt64);
    auto rel = edges.size(1) == 3 ? edges.select(1, 1) : torch::zeros_like(edges.select(1, 0));
    return std::get<0>(((edges.select(1, 0) * relations + rel) * nodes + edges.select(1, -1)).sort());
}
}

int main(int argc, char** argv) {
    try {
        TORCH_CHECK(argc >= 3, "Usage: audit CONFIG OUTPUT_JSON [--observe-padding] [--repartition]");
        bool observe_padding = false, repartition = false;
        for (int i = 3; i < argc; ++i) {
            std::string option(argv[i]);
            if (option == "--observe-padding") observe_padding = true;
            else if (option == "--repartition") repartition = true;
            else TORCH_CHECK(false, "Unknown option: ", option);
        }
        auto cfg = loadConfig(argv[1], false);
        TORCH_CHECK(cfg->training->negative_sampling_method == NegativeSamplingMethod::RNS, "RNS required");
        TORCH_CHECK(cfg->model->dense_optimizer->type == OptimizerType::ADAGRAD, "Adagrad required");
        auto initialized = gege_init(cfg, true);
        auto model = std::get<0>(initialized);
        auto storage = std::get<1>(initialized);
        auto loader = std::get<2>(initialized);
        auto decoder = std::dynamic_pointer_cast<EdgeDecoder>(model->decoder_);
        auto mem = std::dynamic_pointer_cast<MemPartitionBufferStorage>(storage->storage_ptrs_.node_embeddings);
        auto state = std::dynamic_pointer_cast<MemPartitionBufferStorage>(storage->storage_ptrs_.node_optimizer_state);
        TORCH_CHECK(mem && state, "Requires original MEM_PARTITION_BUFFER");
        bool complex = cfg->model->decoder->type == DecoderType::COMPLEX;
        int64_t batches = 0, edge_count = 0, transitions = 0;
        int64_t padding_draws = 0, candidate_draws = 0;
        torch::Tensor expected_embeddings, expected_state;
        auto dense_state = torch::zeros_like(decoder->relations_);
        auto inv_dense_state = torch::zeros_like(decoder->inverse_relations_);
        storage->storage_ptrs_.train_edges->load();
        auto wanted_edges = keys(storage->storage_ptrs_.train_edges->data_, mem->dim0_size_, decoder->num_relations_);
        for (int epoch = 0; epoch < cfg->training->num_epochs; ++epoch) {
            if (epoch && repartition) {
                // nextEpoch has flushed device updates before changing the ID-to-slot map.
                storage->rePartition();
                equal("repartition preserves canonical edge multiset",
                      keys(storage->storage_ptrs_.train_edges->data_, mem->dim0_size_, decoder->num_relations_),
                      wanted_edges, 0, 0);
            }
            loader->setTrainSet();
            loader->initializeBatches(false);
            if (epoch == 0) {
                expected_embeddings = mem->data_.clone();
                expected_state = state->data_.clone();
            }
            std::vector<torch::Tensor> seen_edges;
            torch::Tensor previous_map;
            while (loader->hasNextBatch()) {
                auto batch = loader->getBatch();
                loader->loadGPUParameters(batch);
                auto map = mem->getGlobalToLocalMap(true);
                equal("entity/state slot map", map, state->getGlobalToLocalMap(true), 0, 0);
                if (!previous_map.defined() || !torch::equal(map, previous_map)) transitions++;
                previous_map = map.clone();
                auto global = torch::nonzero(map >= 0).flatten();
                auto inverse = torch::full({mem->getNumInMemory()}, -1, torch::kInt64);
                inverse.index_put_({map.index_select(0, global)}, global);
                auto ids = inverse.index_select(0, batch->unique_node_indices_.to(torch::kCPU));
                auto valid_rows = torch::nonzero(ids >= 0).flatten();
                auto valid_ids = ids.index_select(0, valid_rows);
                TORCH_CHECK(observe_padding || valid_rows.numel() == ids.numel(), "Unmapped entity");
                auto gpu_valid_rows = valid_rows.to(batch->node_embeddings_.device());
                equal("loaded entities survive swaps", batch->node_embeddings_.index_select(0, gpu_valid_rows),
                      expected_embeddings.index_select(0, valid_ids), 0, 0);
                equal("loaded Adagrad state survives swaps", batch->node_embeddings_state_.index_select(0, gpu_valid_rows),
                      expected_state.index_select(0, valid_ids), 0, 0);
                auto rows = batch->edges_.size(0);
                auto decoded = batch->edges_.to(torch::kCPU).clone();
                decoded.select(1, 0).copy_(ids.index_select(0, decoded.select(1, 0)));
                decoded.select(1, -1).copy_(ids.index_select(0, decoded.select(1, -1)));
                TORCH_CHECK(decoded.select(1, 0).min().item<int64_t>() >= 0
                            && decoded.select(1, -1).min().item<int64_t>() >= 0, "Padding positive edge");
                seen_edges.push_back(decoded);
                auto emb = batch->node_embeddings_.detach().clone().set_requires_grad(true);
                auto rel = decoder->relations_.detach().clone().set_requires_grad(true);
                auto inv = decoder->inverse_relations_.detach().clone().set_requires_grad(true);
                auto old_state = batch->node_embeddings_state_.clone();
                bool dot = batch->edges_.size(1) == 2;
                torch::Tensor ref_loss;
                for (int direction = 0; direction < (dot ? 1 : 2); ++direction) {
                    auto h = emb.index_select(0, batch->edges_.select(1, direction ? -1 : 0));
                    auto t = emb.index_select(0, batch->edges_.select(1, direction ? 0 : -1));
                    auto r = dot ? torch::Tensor() : (direction ? inv : rel).index_select(0, batch->edges_.select(1, 1));
                    auto neg_ids = direction ? batch->src_neg_indices_mapping_ : batch->dst_neg_indices_mapping_;
                    auto candidate_global = ids.index_select(0, neg_ids.flatten().to(torch::kCPU));
                    padding_draws += (candidate_global < 0).sum().item<int64_t>();
                    candidate_draws += candidate_global.numel();
                    int64_t chunk_size = (rows + neg_ids.size(0) - 1) / neg_ids.size(0);
                    auto chunks = torch::arange(rows, batch->edges_.options()).div(chunk_size, "trunc");
                    auto selected = neg_ids.index_select(0, chunks);
                    auto neg = emb.index_select(0, selected.flatten()).view({rows, selected.size(1), -1});
                    auto pos_score = score(h, r, t, complex, dot);
                    auto neg_score = score(h.unsqueeze(1), dot ? r : r.unsqueeze(1), neg, complex, dot);
                    auto filter = direction ? batch->src_neg_filter_ : batch->dst_neg_filter_;
                    neg_score.index_put_({filter.select(1, 0), filter.select(1, 1)}, -1e9);
                    auto loss = (torch::cat({pos_score.unsqueeze(1), neg_score}, 1).logsumexp(1) - pos_score).sum();
                    ref_loss = ref_loss.defined() ? ref_loss + loss : loss;
                }
                ref_loss.backward();
                batch->dense_graph_.performMap();
                model->train_batch(batch);
                equal("native RNS entity gradient", batch->node_embeddings_.grad(), emb.grad());
                if (!dot) {
                    equal("forward relation gradient", decoder->relations_.grad(), rel.grad());
                    equal("inverse relation gradient", decoder->inverse_relations_.grad(), inv.grad());
                    torch::NoGradGuard guard;
                    dense_state.add_(decoder->relations_.grad().square());
                    inv_dense_state.add_(decoder->inverse_relations_.grad().square());
                    equal("forward relation update", decoder->relations_, rel - .1 * decoder->relations_.grad() / (dense_state.sqrt() + 1e-10));
                    equal("inverse relation update", decoder->inverse_relations_, inv - .1 * decoder->inverse_relations_.grad() / (inv_dense_state.sqrt() + 1e-10));
                }
                auto g = batch->node_embeddings_.grad();
                equal("entity state increment", batch->node_state_update_, g.square());
                equal("entity update", batch->node_gradients_, -.1 * g / ((old_state + g.square()).sqrt() + 1e-10));
                auto local_ids = batch->unique_node_indices_.index_select(0, valid_rows.to(batch->unique_node_indices_.device()));
                expected_embeddings.index_add_(0, valid_ids, batch->node_gradients_.index_select(0, gpu_valid_rows).to(torch::kCPU));
                expected_state.index_add_(0, valid_ids, batch->node_state_update_.index_select(0, gpu_valid_rows).to(torch::kCPU));
                loader->updateEmbeddings(batch, true);
                equal("GPU embedding indexAdd", mem->indexRead(local_ids, 0), expected_embeddings.index_select(0, valid_ids), 0, 0);
                equal("GPU optimizer indexAdd", state->indexRead(local_ids, 0), expected_state.index_select(0, valid_ids), 0, 0);
                loader->updateEmbeddings(batch, false);
                batch->clear();
                loader->finishedBatch();
                edge_count += rows;
                batches++;
            }
            equal("exact edge multiset per epoch", keys(torch::cat(seen_edges), mem->dim0_size_, decoder->num_relations_), wanted_edges, 0, 0);
            auto host = mem->data_, host_state = state->data_;
            loader->nextEpoch();
            equal("epoch flush entities", host, expected_embeddings, 0, 0);
            equal("epoch flush optimizer", host_state, expected_state, 0, 0);
        }
        std::ofstream out(argv[2]);
        out << "{\"passed\":" << (padding_draws ? "false" : "true")
            << ",\"math_storage_passed\":true,\"padding_candidate_draws\":" << padding_draws
            << ",\"candidate_draws\":" << candidate_draws << ",\"checks\":" << checks << ",\"batches\":" << batches
            << ",\"edges\":" << edge_count << ",\"states_observed\":" << transitions
            << ",\"epochs\":" << cfg->training->num_epochs << ",\"repartition\":" << (repartition ? "true" : "false")
            << ",\"max_absolute_difference\":" << worst << "}\n";
        std::cout << "PASS math, valid-entity storage, and edge coverage: " << checks
                  << " checks; invalid padding candidates: " << padding_draws << "\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << "\n";
        return 1;
    }
}
