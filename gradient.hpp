// (y - h(weights)) ^ 2
// 2 * (y - h(weights)) * -1 * h'(weights)

// sigma(Wp)
// sigma`(Wp) * d/dW Wp
// t = Wp

// derivative of activation matrix

// sigma`(t0) 0 0
// 0 sigma`(t1) 0
// 0 0 sigma`(t2)

// delta matrix

// re-write matrix library to support * and - operators

// w11 w12
// w21 w22

// -2 * residual * sigma'(W0 * p1) * vector with 1 * row_p1
// x0 x1 x2
// x0 x1 x2
// x0 x1 x2
// ---
// p0 * sigma(t0) p1 * sigma(t0) p2 * sigma(t0)
// p0 * sigma(t1)
// p0 * sigma(t2)

// -2 * residual * sigma'(W0 * p1(W1)) * W0 * sigma`(W1 * p2) * vector with 1 * row_p2
// -2 * residual * sigma'(W0 * p1(W2)) * W0 * sigma`(W1 * p2(W2)) * W1 * sigma`(W2 * p3) * vector with 1 * row_p3

// residuals could be a vector

#include <vector>
#include <iostream>

#include <math.h>

#include "matrix.hpp"

double TanHPrime(double);


// -2 * residual * sigma'(W0 * p1) * vector with 1 * row_p1
// -2 * residual * sigma'(W0 * p1(W1)) * W0 * sigma`(W1* p2) * vector with 1 * row_p2
// -2 * residual * sigma'(W0 * p1(W2)) * W0 * sigma`(W1 * p2(W2)) * W1 * sigma`(W2 * p3) * vector with 1 * row_p3
std::vector<Matrix> Derivatives()
{
    std::vector<Matrix> weights;
    std::vector<Vector> activations;
    Matrix residuals(1, activations[0].Size());

    std::vector<Matrix> derivatives;

    Matrix t = (-2 * residuals) * *Diag((weights[0] * activations[1]).Apply(TanHPrime));
    Matrix derivative = t * VectorOfAllOnes(weights[0].Rows()) * VectorToRowMatrix(activations[1]);
    derivatives[0] = derivative;

    for (int i = 1; i < weights.size(); i++){
        t = t * weights[i - 1] * *Diag((weights[i] * activations[1 + 1]).Apply(TanHPrime));
        derivative = t * VectorOfAllOnes(weights[i].Rows()) * VectorToRowMatrix(activations[i + 1]);
        derivatives[i] = derivative;
    }

    return derivatives;
}