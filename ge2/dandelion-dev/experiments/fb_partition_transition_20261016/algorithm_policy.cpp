#include <ATen/Context.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
__attribute__((constructor)) void controlled_algorithm_policy() {
    const char* v = std::getenv("GEGE_DETERMINISTIC_ALGORITHMS");
    if (!v || (std::strcmp(v,"0") != 0 && std::strcmp(v,"1") != 0)) {
        std::fprintf(stderr,"GEGE_DETERMINISTIC_ALGORITHMS must be 0 or 1\n"); std::abort();
    }
    at::globalContext().setDeterministicAlgorithms(std::strcmp(v,"1")==0, false);
    at::globalContext().setDeterministicFillUninitializedMemory(false);
    std::fprintf(stderr,"[controlled_algorithm_policy] deterministic=%d warn_only=0 fill_uninitialized=%d\n",int(at::globalContext().deterministicAlgorithms()),int(at::globalContext().deterministicFillUninitializedMemory()));
}
