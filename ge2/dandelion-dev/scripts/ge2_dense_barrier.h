#pragma once

#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <stdexcept>

// Complete both reduction and participant retirement before opening the next round.
class Ge2DenseBarrier {
    std::mutex mutex_;
    std::condition_variable cv_;
    std::uint64_t generation_ = 0;
    int arrivals_ = 0, departures_ = 0, expected_ = 0;
    bool reduced_ = false;

public:
    template <class Participants, class Reduce, class Finish>
    void run(Participants participants, Reduce reduce, Finish finish) {
        std::unique_lock<std::mutex> lock(mutex_);
        const auto generation = generation_;
        if (arrivals_ == 0) {
            expected_ = participants();
            if (expected_ <= 0) throw std::logic_error("No active dense-sync participants");
        }
        if (++arrivals_ == expected_) {
            reduce();
            reduced_ = true;
            cv_.notify_all();
        } else {
            cv_.wait(lock, [&] { return reduced_; });
        }
        finish();
        if (++departures_ == expected_) {
            arrivals_ = departures_ = 0;
            reduced_ = false;
            ++generation_;
            cv_.notify_all();
        } else {
            cv_.wait(lock, [&] { return generation_ != generation; });
        }
    }
};
