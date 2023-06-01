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

    std::vector<Matrix> CalculateGradient(std::vector<Vector> activations, Vector actual);

    void Print();
};

double cross_entropy(Vector predicted, Vector expected);