#include <vector>

#include "matrix.hpp"
#include "nn.hpp"
#include "dataset.hpp"

#include "der.cpp"

#define SEED 1023

int main(int argc, char** argv) {
    srand(SEED);

    // std::vector<std::vector<double>> a {
    //     {1.0, 2.0},
    //     {3.0, 4.0}
    // };

    // Matrix A(a);

    // A.Print();

    // std::vector<std::vector<double>> b {
    //     {5.0, 7.0, 9.0},
    //     {6.0, 8.0, 10.0}
    // };

    // Matrix B(b);

    // B.Print();

    // Matrix C = A * B;

    // C.Print();

    // Matrix D = C.Transpose();

    // D.Print();

    // Matrix E = 0.5 * D;

    // E.Print();

    // Matrix F = Identity(5);

    // F.Print();

    NetworkMetadata metadata(std::vector<int>{2, 3, 2});

    Network network(metadata);

    network.Print();

    Dataset dataset("../train.csv");

    Vector in(dataset.in[0]);

    in.Print();

    Vector out(dataset.out[0]);

    out.Print();

    std::vector<std::vector<double>> hidden {
        {1.0, 1.0, 1.0, 1.0},
        {1.0, 1.0, 1.0, 1.0},
    };
    network.weights[1] = hidden;
    std::cout << "start weights" << std::endl;
    for (int i = 0; i < weights.size(); i++) {
        weights[i].Print();
        network.weights[i] = &weights[i];
    }
    std::cout << "end weights" << std::endl;
    std::vector<Vector> activations = network.Activations(in);
    std::cout << "activations" << std::endl;
    for (Vector activation: activations) {
        activation.Print();
    }
    std::cout << "end" << std::endl;
    auto gradients = Gradients(weights, activations, out);
    std::cout << "start gradients" << std::endl;
    for(auto gradient: gradients) {
        gradient.Print();
    }
    std::cout << "end gradients" << std::endl;

    // for (int i = 0; i < 200; i++) {
    //     network.Epoch(dataset.in, dataset.out);
    // }

    return 0;
}