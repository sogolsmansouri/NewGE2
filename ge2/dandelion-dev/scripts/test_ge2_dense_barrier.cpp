#include "ge2_dense_barrier.h"
#include <atomic>
#include <cassert>
#include <chrono>
#include <iostream>
#include <thread>
#include <vector>

int main() {
    for (int lanes : {2, 4}) {
        for (int trial = 0; trial < 20; ++trial) {
            Ge2DenseBarrier barrier;
            std::atomic<int> active{lanes}, finished{0};
            std::vector<int> completed(lanes, 0);
            int reductions = 0;
            std::vector<std::thread> threads;
            for (int lane = 0; lane < lanes; ++lane) {
                threads.emplace_back([&, lane] {
                    const int batches = 100 + lane * 11;
                    for (int batch = 0; batch < batches; ++batch) {
                        if ((lane + batch + trial) % 7 == 0)
                            std::this_thread::sleep_for(std::chrono::microseconds(10));
                        barrier.run([&] { return active.load(); }, [&] {
                            for (int j = 0; j < lanes; ++j)
                                assert(completed[j] == std::min(reductions, 100 + j * 11));
                            ++reductions;
                        }, [&] {
                            ++completed[lane];
                            ++finished;
                            if (batch + 1 == batches) --active;
                        });
                    }
                });
            }
            for (auto &thread : threads) thread.join();
            assert(reductions == 100 + (lanes - 1) * 11);
            assert(finished == lanes * 100 + 11 * lanes * (lanes - 1) / 2);
            assert(active == 0);
        }
    }
    std::cout << "PASS: 2/4 lanes, skewed arrival and retirement, exactly one reduction per round\n";
}
