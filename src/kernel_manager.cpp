#include "kernel_manager.hpp"
#include "xbyak.h"
#include <mutex>
#include <cstring>

// Minimal JIT AVX-512 kernel using Xbyak
static matvec_kernel_fn jit_avx512_kernel(size_t M, size_t K) {
    struct Kernel : Xbyak::CodeGenerator {
        Kernel(size_t M, size_t K) : Xbyak::CodeGenerator(4096) {
            // Arguments: A (rdi), x (rsi), y (rdx), M (rcx), K (r8)
            // For now, only support K divisible by 16 (no tail handling)
            using namespace Xbyak;
            // Prologue
            push(rbp); mov(rbp, rsp);
            mov(r9, rdi); // r9 = A
            mov(r10, rsi); // r10 = x
            mov(r11, rdx); // r11 = y
            xor_(r12, r12); // i = 0
            L("row_loop");
            cmp(r12, rcx); jge("done");
            vxorps(zmm0, zmm0, zmm0); // acc = 0
            xor_(r13, r13); // k = 0
            L("k_loop");
            cmp(r13, r8); jge("k_done");
            mov(rax, r9); add(rax, r12); imul(rax, r8, 4); add(rax, r13, 4); // A[i*K + k]
            vmovups(zmm1, ptr[rax]); // load 16 floats from A
            mov(rax, r10); add(rax, r13, 4); // x[k]
            vmovups(zmm2, ptr[rax]); // load 16 floats from x
            vfmadd231ps(zmm0, zmm1, zmm2); // acc += A * x
            add(r13, 16);
            jmp("k_loop");
            L("k_done");
            mov(rax, r11); add(rax, r12, 4); // y[i]
            vmovss(ptr[rax], xmm0); // store acc
            add(r12, 1);
            jmp("row_loop");
            L("done");
            pop(rbp);
            ret();
        }
        matvec_kernel_fn get() { return getCode<matvec_kernel_fn>(); }
    } code(M, K);
    return code.get();
}

// Naive fallback kernel (for now)
static void fallback_kernel(const float* A, const float* x, float* y, size_t M, size_t K) {
    for (size_t i = 0; i < M; ++i) {
        float acc = 0.0f;
        for (size_t k = 0; k < K; ++k) acc += A[i*K + k] * x[k];
        y[i] = acc;
    }
}

matvec_kernel_fn KernelManager::get_or_create(size_t M, size_t K, bool avx512) {
    static std::mutex mtx;
    std::lock_guard<std::mutex> lock(mtx);
    auto key = std::make_tuple(M, K, avx512);
    auto it = cache_.find(key);
    if (it != cache_.end()) return it->second;
    matvec_kernel_fn fn = avx512 ? jit_avx512_kernel(M, K) : fallback_kernel;
    cache_[key] = fn;
    return fn;
}
