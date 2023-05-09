#include <vector>

#include "matrix.hpp"
#include "nn.hpp"
#include "dataset.hpp"

#include "der.cpp"

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

    NetworkMetadata metadata(std::vector<int>{2, 3, 2});

    Network network(metadata);

    network.Print();

    Dataset dataset("../train.csv");

    Vector in(dataset.in[0]);

    in.Print();

    // std::vector<Vector*> activations = network.ForwardPropogation(in);

    // for (Vector* activation: activations) {
    //     activation->Print();
    // }

    std::cout << "activations" << std::endl;
    std::vector<Vector> activations = network.Activations(in);

    for (Vector activation: activations) {
        activation.Print();
    }

    std::cout << "end" << std::endl;

    Vector out(dataset.out[0]);

    out.Print();

    std::vector<Matrix> gradients = Gradients(
        dereference(network.weights),
        activations,
        out
    );

    for (Matrix gradient: gradients) {
        gradient.Print();
    }

    // std::vector<Matrix*> derivatives = network.Derivatives(activations, &out);

    // for (Matrix* derivatives: derivatives) {
    //     derivatives->Print();
    // }
 
    return 0;
}