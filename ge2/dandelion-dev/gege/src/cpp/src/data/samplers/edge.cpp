#include "data/samplers/edge.h"

RandomEdgeSampler::RandomEdgeSampler(shared_ptr<GraphModelStorage> graph_storage, bool without_replacement) {
    graph_storage_ = graph_storage;
    without_replacement_ = without_replacement;
}

// EdgeList RandomEdgeSampler::getEdges(shared_ptr<Batch> batch) {
//    return storage_->getEdgesRange(batch->start_idx_, batch->batch_size_).clone().to(torch::kInt64);
// }

EdgeList RandomEdgeSampler::getEdges(shared_ptr<Batch> batch, int32_t device_idx) {
    if (batch->streamed_edge_size_ > 0) {
        return graph_storage_->getCurrentStateMappedEdgesRange(batch->streamed_edge_start_, batch->streamed_edge_size_, device_idx).clone().to(torch::kInt64);
    }
    return graph_storage_->getEdgesRange(batch->start_idx_, batch->batch_size_, device_idx).clone().to(torch::kInt64);
}
