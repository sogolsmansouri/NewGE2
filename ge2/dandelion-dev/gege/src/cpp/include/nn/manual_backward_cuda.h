#pragma once

#include <torch/types.h>

torch::Tensor negative_log_mass_backward_cuda(torch::Tensor scores, torch::Tensor log_mass, torch::Tensor grad_mass);
torch::Tensor ordered_gradient_sum_cuda(torch::Tensor components);
