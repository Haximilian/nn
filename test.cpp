#include <vector>

#include "matrix.hpp"
#include "nn.hpp"
#include "dataset.hpp"

#define SEED 1024

int main(int argc, char** argv) {
    srand(SEED);

    std::vector<std::vector<double>> a {
        {1.0, 2.0},
        {3.0, 4.0}
    };

    Matrix A(a);

    A.Print();

    std::vector<std::vector<double>> b {
        {5.0, 7.0, 9.0},
        {6.0, 8.0, 10.0}
    };

    Matrix B(b);

    B.Print();

    Matrix C = A * B;

    C.Print();

    Matrix D = C.Transpose();

    D.Print();

    Matrix E = 0.5 * D;

    E.Print();

    Matrix F = Identity(5);

    F.Print();

    return 0;
}