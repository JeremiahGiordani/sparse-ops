// ultrasparse_basics.cpp

#include <iostream>
#include <vector>
#include <cassert>

using Matrix = std::vector<std::vector<float>>;

Matrix matmul(const Matrix& A, const Matrix& B) {
    assert(!A.empty() && !B.empty());
    size_t m = A.size();
    size_t k = A[0].size();
    size_t n = B[0].size();
    assert(B.size() == k);

    Matrix result(m, std::vector<float>(n, 0.0f));
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            for (size_t l = 0; l < k; ++l) {
                result[i][j] += A[i][l] * B[l][j];
            }
        }
    }
    return result;
}

void print_matrix(const Matrix& M) {
    for (const auto& row : M) {
        for (float val : row) {
            std::cout << val << " ";
        }
        std::cout << "\n";
    }
}

int main() {
    Matrix A = {
        {1, 2},
        {3, 4}
    };
    Matrix B = {
        {5, 6},
        {7, 8}
    };

    Matrix C = matmul(A, B);
    print_matrix(C);

    return 0;
}
