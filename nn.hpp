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

class Network
{
private:
    // std::vector<Matrix*> weights;

public:
    std::vector<Matrix*> weights;

    Network(NetworkMetadata metadata);

    std::vector<Vector*> ForwardPropogation(Vector in);

    std::vector<Vector> Activations(Vector in);

    void Epoch(std::vector<std::vector<double>> in, std::vector<std::vector<double>> out);

    Vector* Residual(Vector* in, Vector* out);

    void Print();
};

std::vector<Matrix> Gradients(
    std::vector<Matrix> weights,
    std::vector<Vector> activations,
    Vector actual
);

std::vector<Matrix> Gradients(
    std::vector<Matrix> weights,
    std::vector<std::vector<Vector>> activations,
    std::vector<Vector> actual
); 

std::vector<Matrix> dereference(std::vector<Matrix*> in);