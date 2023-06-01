#pragma once

#include <vector>

#include "matrix.hpp"

class NetworkMetadata
{
private:
    std::vector<int> metadata;

public:
    NetworkMetadata(std::vector<int> metadata);

    int operator[](int index) const;

    int Size() const;
};

typedef Vector activation_fn(Vector);
typedef activation_fn* activation_fn_ptr;

class Network
{
private:
    // std::vector<Matrix*> weights;
    std::vector<activation_fn_ptr> activation_fns;

public:
    std::vector<Matrix> weights;

    Network(NetworkMetadata metadata);

    std::vector<Vector> ForwardPropagation(Vector in);

    // void Epoch(std::vector<std::vector<double>> in, std::vector<std::vector<double>> out);

    // Vector* Residual(Vector* in, Vector* out);

    void Print();
};

std::vector<Matrix> Gradients(
    std::vector<Matrix> weights,
    std::vector<Vector> activations,
    Vector actual
);

double cross_entropy(Vector predicted, Vector expected);

