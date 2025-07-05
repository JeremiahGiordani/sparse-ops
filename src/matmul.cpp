// matmul.cpp
#include <cstddef>

extern "C" {

// A × B = C
// A: (m x k)
// B: (k x n)
// C: (m x n)
void matmul(const float* A, const float* B, float* C, int m, int k, int n) {
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            float sum = 0.0f;
            for (int p = 0; p < k; ++p) {
                sum += A[i * k + p] * B[p * n + j];
            }
            C[i * n + j] = sum;
        }
    }
}

}
