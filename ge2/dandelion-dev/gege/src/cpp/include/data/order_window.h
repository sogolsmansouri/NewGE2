#pragma once

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <numeric>
#include <set>
#include <utility>
#include <vector>

namespace stateflow {
// A bounded search preserves training locality without changing bucket ownership.
inline std::vector<int64_t> boundedOrderPermutation(
        std::vector<std::vector<int64_t>> states, int lanes, int64_t window,
        int64_t max_admits, std::size_t beam_width = 128) {
    const auto count = static_cast<int64_t>(states.size());
    if (count == 0 || lanes < 1 || window < 0 || beam_width == 0) return {};
    for (auto &state : states) {
        std::sort(state.begin(), state.end());
        if (state.empty() || std::adjacent_find(state.begin(), state.end()) != state.end()) return {};
    }
    auto admissions = [&](int64_t previous, int64_t next) {
        if (previous < 0) return static_cast<int64_t>(states[next].size());
        int64_t retained = 0;
        for (auto partition : states[next])
            retained += std::binary_search(states[previous].begin(), states[previous].end(), partition);
        return static_cast<int64_t>(states[next].size()) - retained;
    };
    auto disjoint = [&](int64_t first, int64_t second) {
        for (auto partition : states[first])
            if (std::binary_search(states[second].begin(), states[second].end(), partition)) return false;
        return true;
    };
    struct Prefix {
        std::vector<int64_t> permutation;
        std::vector<int64_t> previous;
        std::vector<bool> used;
        int64_t displacement = 0;
        int64_t admits = 0;
    };
    std::vector<Prefix> beam(1);
    beam.front().previous.assign(lanes, -1);
    beam.front().used.assign(count, false);
    for (int64_t position = 0; position < count; position += lanes) {
        const int round_size = static_cast<int>(std::min<int64_t>(lanes, count - position));
        std::vector<Prefix> expanded;
        for (const auto &prefix : beam) {
            int64_t anchor = 0;
            while (anchor < count && prefix.used[anchor]) ++anchor;
            std::vector<int64_t> round;
            std::size_t generated = 0;
            // The earliest outstanding state must advance; it cannot starve behind lookahead.
            std::function<void(int)> search = [&](int lane) {
                if (generated >= beam_width * 8) return;
                if (lane == round_size) {
                    if (std::find(round.begin(), round.end(), anchor) == round.end()) return;
                    Prefix next = prefix;
                    for (int slot = 0; slot < round_size; ++slot) {
                        auto state = round[slot];
                        next.displacement += std::abs(state - (position + slot));
                        next.admits += admissions(next.previous[slot], state);
                        next.previous[slot] = state;
                        next.used[state] = true;
                        next.permutation.push_back(state);
                    }
                    expanded.emplace_back(std::move(next));
                    ++generated;
                    return;
                }
                const auto at = position + lane;
                for (auto state = std::max<int64_t>(0, at - window);
                     state < count && state <= at + window; ++state) {
                    if (prefix.used[state] || std::find(round.begin(), round.end(), state) != round.end()) continue;
                    if (prefix.previous[lane] >= 0 && max_admits >= 0 &&
                        admissions(prefix.previous[lane], state) > max_admits) continue;
                    if (lane + 1 == round_size && state != anchor &&
                        std::find(round.begin(), round.end(), anchor) == round.end()) continue;
                    if (std::any_of(round.begin(), round.end(), [&](auto other) { return !disjoint(state, other); })) continue;
                    round.push_back(state);
                    search(lane + 1);
                    round.pop_back();
                }
            };
            search(0);
        }
        if (expanded.empty()) return {};
        std::sort(expanded.begin(), expanded.end(), [](const Prefix &first, const Prefix &second) {
            if (first.displacement != second.displacement) return first.displacement < second.displacement;
            if (first.admits != second.admits) return first.admits < second.admits;
            return first.permutation < second.permutation;
        });
        beam.clear();
        std::set<std::pair<std::vector<bool>, std::vector<int64_t>>> seen;
        for (auto &prefix : expanded) {
            if (!seen.emplace(prefix.used, prefix.previous).second) continue;
            beam.emplace_back(std::move(prefix));
            if (beam.size() == beam_width) break;
        }
    }
    return beam.front().permutation;
}
}
