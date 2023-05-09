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

    // std::vector<Matrix*> Derivatives(std::vector<Vector*> activations, Vector* out);

    Vector* Residual(Vector* in, Vector* out);

    void Print();
};