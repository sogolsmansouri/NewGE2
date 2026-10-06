#include <iostream>
#include <stdexcept>
#include "data/order_window.h"

void check(bool condition) {
    if (!condition) throw std::runtime_error("bounded-order unit check failed");
}

void validate(const std::vector<std::vector<int64_t>> &states,
              const std::vector<int64_t> &order, int lanes, int64_t window, int64_t max_admits) {
    check(order.size() == states.size());
    auto sorted = order;
    std::sort(sorted.begin(), sorted.end());
    for (std::size_t i = 0; i < order.size(); ++i) {
        check(sorted[i] == static_cast<int64_t>(i));
        check(std::abs(order[i] - static_cast<int64_t>(i)) <= window);
        if (i >= static_cast<std::size_t>(lanes)) {
            int64_t admits = 0;
            for (auto partition : states[order[i]]) {
                const auto &previous = states[order[i - lanes]];
                admits += std::find(previous.begin(), previous.end(), partition) == previous.end();
            }
            check(admits <= max_admits);
        }
        for (auto j = i / lanes * lanes; j < i; ++j)
            for (auto partition : states[order[i]])
                check(std::find(states[order[j]].begin(), states[order[j]].end(), partition) == states[order[j]].end());
    }
}

int main() {
    std::vector<std::vector<int64_t>> states = {{0, 1}, {1, 2}, {3, 4}, {4, 5}};
    auto order = stateflow::boundedOrderPermutation(states, 2, 1, 1);
    validate(states, order, 2, 1, 1);
    check(order == stateflow::boundedOrderPermutation(states, 2, 1, 1));
    check(stateflow::boundedOrderPermutation(states, 2, 0, 1).empty());
    check(stateflow::boundedOrderPermutation(states, 2, 1, 0).empty());
    states.push_back({1, 6});
    validate(states, stateflow::boundedOrderPermutation(states, 2, 1, 1), 2, 1, 1);
    std::vector<std::vector<int64_t>> four_lanes = {{0}, {1}, {2}, {3}, {4}, {5}, {6}, {7}};
    validate(four_lanes, stateflow::boundedOrderPermutation(four_lanes, 4, 0, 1), 4, 0, 1);
    check(stateflow::boundedOrderPermutation({{0}, {0}}, 2, 2, 1).empty());
    check(stateflow::boundedOrderPermutation({{0, 0}, {1}}, 2, 2, 1).empty());
    check(stateflow::boundedOrderPermutation(states, 0, 1, 1).empty());
    check(stateflow::boundedOrderPermutation(states, 2, -1, 1).empty());
    check(stateflow::boundedOrderPermutation(states, 2, 1, 1, 0).empty());
    std::cout << "bounded-order unit tests passed\n";
}
