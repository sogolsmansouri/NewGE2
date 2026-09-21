#include "nn/manual_backward_cuda.h"

#include <ATen/TensorIterator.h>
#include <ATen/native/cuda/Loops.cuh>
#include <c10/cuda/CUDAGuard.h>

torch::Tensor negative_log_mass_backward_cuda(torch::Tensor scores, torch::Tensor log_mass, torch::Tensor grad_mass) {
    TORCH_CHECK(scores.is_cuda() && scores.scalar_type() == torch::kFloat32, "Expected CUDA float32 scores");
    const c10::cuda::CUDAGuard guard(scores.device());
    auto output = torch::empty_like(scores);
    auto iter = at::TensorIteratorConfig()
        .add_output(output)
        .add_input(scores)
        .add_input(log_mass)
        .add_input(grad_mass)
        .build();
    // Keep the reference subtract -> exp -> multiply rounding, but avoid two
    // full score-matrix temporaries and their reads/writes.
    at::native::gpu_kernel(iter, [] GPU_LAMBDA(float score, float mass, float grad) -> float {
        return grad * expf(score - mass);
    });
    return output;
}

torch::Tensor ordered_gradient_sum_cuda(torch::Tensor components) {
    TORCH_CHECK(components.is_cuda() && components.scalar_type() == torch::kFloat32 && components.dim() == 3,
                "Expected CUDA float32 gradient components");
    const c10::cuda::CUDAGuard guard(components.device());
    TORCH_CHECK(components.size(0) == 3 || components.size(0) == 4, "Expected three or four independent gradients");
    auto output = components.select(0, 0);
    auto config = at::TensorIteratorConfig();
    config.add_output(output);
    std::vector<torch::Tensor> inputs;
    for (int64_t index = 0; index < components.size(0); ++index) inputs.push_back(components.select(0, index));
    for (const auto &input : inputs) config.add_input(input);
    auto iter = config.build();
    // Do not reassociate the independent index_select backward reductions.
    if (components.size(0) == 3) {
        at::native::gpu_kernel(iter, [] GPU_LAMBDA(float a, float b, float c) -> float { return (a + b) + c; });
    } else {
        at::native::gpu_kernel(iter, [] GPU_LAMBDA(float a, float b, float c, float d) -> float { return ((a + b) + c) + d; });
    }
    return output;
}
