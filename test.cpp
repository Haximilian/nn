#include <stdio.h>

#include "m.cpp"

#define SEED 1024

int main(int argc, char** argv) {
    srand(SEED);

    std::cout << sizeof(matrix<double, 16, 16>(return_random)) << std::endl;
    std::cout << sizeof(matrix<double, 8, 8>(return_random)) << std::endl;

    // abstract_matrix<double> m = matrix<double, 16, 16>(return_random)

    // matrix<double, 16, 16> m(return_random);

    // m.print();

    // matrix<double, 16, 2> v(return_random);

    // v.print();

    // matrix<double, 16, 2> r = m * v;

    // r.print();

    return 0;
}