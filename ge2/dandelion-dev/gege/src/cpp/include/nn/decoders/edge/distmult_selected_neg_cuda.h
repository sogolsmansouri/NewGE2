#pragma once

#include "common/datatypes.h"

torch::Tensor distmult_selected_neg_scores(torch::Tensor chunked_adjusted_embeddings,
                                           torch::Tensor negative_embeddings,
                                           torch::Tensor selected_neg_indices);

torch::Tensor distmult_selected_neg_scores_cuda_forward(torch::Tensor chunked_adjusted_embeddings,
                                                        torch::Tensor negative_embeddings,
                                                        torch::Tensor selected_neg_indices);

std::tuple<torch::Tensor, torch::Tensor> distmult_selected_neg_scores_cuda_backward(torch::Tensor grad_out,
                                                                                    torch::Tensor chunked_adjusted_embeddings,
                                                                                    torch::Tensor negative_embeddings,
                                                                                    torch::Tensor selected_neg_indices);
