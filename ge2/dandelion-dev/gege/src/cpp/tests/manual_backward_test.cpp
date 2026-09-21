#include <torch/cuda.h>
#include <iostream>
#include <limits>

#include "nn/loss.h"
#include "nn/manual_backward_cuda.h"

int main() {
    torch::manual_seed(813);
    bool passed = true;
    auto same = [](torch::Tensor a, torch::Tensor b) {
        return torch::equal(torch::isnan(a), torch::isnan(b)) &&
               torch::equal(torch::nan_to_num(a), torch::nan_to_num(b));
    };
    for (const auto device : {torch::kCPU, torch::kCUDA}) {
        for (const auto dtype : {torch::kFloat32, torch::kFloat64}) {
            for (const auto reduction : {LossReduction::SUM, LossReduction::MEAN}) {
                auto options = std::make_shared<LossOptions>();
                options->loss_reduction = reduction;
                SoftmaxCrossEntropy loss(options);
                auto tensor_options = torch::TensorOptions().device(device).dtype(dtype);
                for (int64_t negatives : {1, 7, 32, 1000}) {
                    for (bool strided : {false, true}) {
                        auto positive = (torch::randn({37}, tensor_options) * 20).requires_grad_();
                        auto negative = torch::randn({37, strided ? negatives * 2 : negatives}, tensor_options) * 20;
                        if (strided) negative = negative.slice(1, 0, negatives * 2, 2);
                        negative.select(0, 0).fill_(-std::numeric_limits<double>::infinity());
                        negative.requires_grad_();
                        auto objective = loss(positive, negative, true);
                        objective.backward();
                        auto actual = loss.score_gradients(positive, negative);
                        bool ok = same(std::get<0>(actual).squeeze(1), positive.grad()) && same(std::get<1>(actual), negative.grad());
                        passed &= ok;
                        std::cout << "score_gradients device=" << device << " dtype=" << dtype
                                  << " mean=" << (reduction == LossReduction::MEAN) << " negatives=" << negatives
                                  << " strided=" << strided << " pass=" << ok << std::endl;
                    }
                }
            }
        }
    }
    for (int64_t count : {3, 4}) {
        auto components = torch::randn({count, 1003, 100}, torch::TensorOptions().device(torch::kCUDA));
        // Cancellation makes a change in association visible.
        components.select(0, 0).fill_(1e8);
        components.select(0, 1).fill_(-1e8);
        auto expected = components.select(0, 0).clone();
        for (int64_t index = 1; index < count; ++index) expected.add_(components.select(0, index));
        auto actual = ordered_gradient_sum_cuda(components);
        bool ok = torch::equal(expected, actual);
        passed &= ok;
        std::cout << "ordered_gradient_sum count=" << count << " pass=" << ok << std::endl;
    }
    torch::cuda::synchronize();
    return passed ? 0 : 1;
}
