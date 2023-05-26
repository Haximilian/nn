#include <vector>

#include "matrix.hpp"
#include "nn.hpp"
#include "dataset.hpp"

#define SEED 1024

int main(int argc, char** argv) {
    srand(SEED);

    // std::vector<std::vector<double>> a {
    //     {1.0, -2.0, 0.0},
    //     {-3.0, 4.0, 0.2},
    //     {5.0, -6.0, 0.8}
    // };

    // Matrix A(a);

    // std::vector<std::vector<double>> b {
    //     {0.5, 1.0/3.0, 1.0, 0.3},
    //     {1.0, 0.25, 0.5, -0.7},
    // };

    // Matrix B(b);

    NetworkMetadata metadata(std::vector<int>{2, 3, 2});

    Network network(metadata);

    // network.weights[0] = A;
    // network.weights[1] = B;

    network.Print();

    std::vector<double> t_in {
        3.0,
        4.0
    };
    Vector in(t_in);
    in.Print();

    std::vector<double> t_out {
        1,
        0
    };
    Vector out(t_out);
    out.Print();

    for (int j = 0; j < 16; j++)
    {
        std::vector<Vector> activations = network.ForwardPropagation(in);
        // for (auto activation : activations)
        // {
        //     activation.Print();
        // }
        activations.back().Print();

        std::vector<Matrix> gradients = Gradients(
            network.weights,
            activations,
            out);

        for (int i = 0; i < gradients.size(); i++)
        {
            network.weights[i] = network.weights[i] - 0.03 * gradients[i];
        }
    }
}