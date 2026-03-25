#pragma once

#include "common/datatypes.h"
#include "configuration/options.h"
#include "data/samplers/negative.h"
#include "nn/decoders/edge/edge_decoder.h"

// qual_embeddings: per-edge positive qualifier-value embeddings for arity-3/4, shape (batch_size, dim).
// qval_neg_embeddings: sampled qualifier-value negatives for arity-3/4, shape (num_chunks, num_negatives, dim).
// Pass an undefined tensor (default) for binary edges.

std::tuple<torch::Tensor, torch::Tensor> only_pos_forward(shared_ptr<EdgeDecoder> decoder, torch::Tensor positive_edges, torch::Tensor node_embeddings,
                                                          torch::Tensor qual_embeddings = torch::Tensor());

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> neg_and_pos_forward(shared_ptr<EdgeDecoder> decoder, torch::Tensor positive_edges,
                                                                                           torch::Tensor negative_edges, torch::Tensor node_embeddings);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> node_corrupt_forward(shared_ptr<EdgeDecoder> decoder, torch::Tensor positive_edges,
                                                                                            torch::Tensor node_embeddings, torch::Tensor dst_negs,
                                                                                            torch::Tensor src_negs = torch::Tensor(),
                                                                                            torch::Tensor qual_embeddings = torch::Tensor(),
                                                                                            torch::Tensor qval_neg_embeddings = torch::Tensor());

std::tuple<torch::Tensor, torch::Tensor> node_corrupt_ranks_chunked(shared_ptr<EdgeDecoder> decoder, torch::Tensor positive_edges,
                                                                    torch::Tensor node_embeddings, torch::Tensor dst_negs, torch::Tensor dst_filter,
                                                                    torch::Tensor src_negs = torch::Tensor(), torch::Tensor src_filter = torch::Tensor(),
                                                                    torch::Tensor qual_embeddings = torch::Tensor(), int64_t chunk_size = 131072);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> rel_corrupt_forward(shared_ptr<EdgeDecoder> decoder, torch::Tensor positive_edges,
                                                                                           torch::Tensor node_embeddings, torch::Tensor neg_rel_ids);

std::tuple<torch::Tensor, torch::Tensor> prepare_pos_embeddings(shared_ptr<EdgeDecoder> decoder, torch::Tensor positive_edges, torch::Tensor src_embeddings, torch::Tensor dst_embeddings, bool has_relations,
                                                                 torch::Tensor qual_embeddings = torch::Tensor());

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> mod_node_corrupt_forward(NegativeSamplingMethod negative_sampling_method, float negative_sampling_selected_ratio,
                                                                                                shared_ptr<NegativeSampler> negative_sampler, shared_ptr<EdgeDecoder> decoder, torch::Tensor positive_edges,
                                                                                                torch::Tensor node_embeddings, torch::Tensor dst_negs, torch::Tensor src_negs,
                                                                                                torch::Tensor node_embeddings_g, torch::Tensor qual_embeddings = torch::Tensor(),
                                                                                                torch::Tensor qval_neg_embeddings = torch::Tensor());

std::tuple<torch::Tensor, torch::Tensor> get_rewards(shared_ptr<EdgeDecoder> decoder, torch::Tensor positive_edges, torch::Tensor node_embeddings, torch::Tensor dst_negs, torch::Tensor src_negs,
                                                      torch::Tensor qual_embeddings = torch::Tensor());

torch::Tensor forward_g(shared_ptr<EdgeDecoder> decoder, torch::Tensor positive_edges, torch::Tensor node_embeddings_g, torch::Tensor dst_negs, torch::Tensor src_negs, torch::Tensor reward, torch::Tensor inv_reward,
                        torch::Tensor qual_embeddings = torch::Tensor());
