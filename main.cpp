#include <vector>
#include <array>

#include "matrix.hpp"
#include "nn.hpp"
#include "dataset.hpp"

#define SEED 1024
#define BATCH_SIZE 512
#define EPOCH_COUNT 64

float cross_entropy_dataset(Dataset<float> d, Network<float> n) {
    float r = 0;
    for (int i = 0; i < d.in.size(); i++) {
        Vector<float> predicted = n.ForwardPropagation(d.in[i]).back();
        r = r + cross_entropy(predicted, d.out[i]);
    }
    return r;
}

int main(int argc, char** argv) {
    srand(SEED);

    NetworkMetadata metadata(std::vector<int>{2, 8, 8, 2});

    Network<float> network(metadata);

    Dataset<float> dataset("./train.csv");

    for (int k = 0; k < EPOCH_COUNT * BATCH_SIZE; k++) {
        int j = k % BATCH_SIZE;
        Vector<float> in(dataset.in[j]);

        // in.Print();

        Vector<float> out(dataset.out[j]);

        // out.Print();

        std::vector<Vector<float>> activations = network.ForwardPropagation(in);
        // for (auto activation : activations)
        // {
        //     activation.Print();
        // }
        // activations.back().Print();

        std::vector<Matrix<float>> gradients = network.CalculateGradient(
            activations,
            out);

        for (int i = 0; i < gradients.size(); i++)
        {
            network.weights[i] = network.weights[i] - gradients[i] * 0.03;
        }

        if (j % 128 == 0) {
            // std::cout << "dataset cross entropy" << std::endl;
            std::cout << cross_entropy_dataset(dataset, network) << "," << std::endl;
        }
    }

    // for (int i = 1024; i < 1024 + 16; i++)
    // {
    //     Vector in(dataset.in[i]);
    //     in.Print();

    //     Vector out(dataset.out[i]);

    //     out.Print();

    //     std::vector<Vector> activations = network.ForwardPropagation(in);
    //     activations.back().Print();
    // }
    
    // for (double i = -8; i <= 8; i = i + 0.5) {
    //     for (double j = -8; j <= 8; j = j + 0.5) {
    //         std::vector<double> in {
    //             i, j
    //         };
    //         double out = network.ForwardPropagation(in).back().Get(0);
    //         std::cout << i << "," << j << "," << out << std::endl;
    //     }
    // }

    // network.Print();
}