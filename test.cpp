#include <stdio.h>

#include "network.hpp"
#include "dataset.hpp"

#define SEED 1024
#define BATCH_SIZE 512
#define EPOCH_COUNT 64

template<typename T>
void print(std::vector<T> in)
{
    std::cout << "---------- Vector Print ----------" << std::endl;

    for (T element : in)
    {
        std::cout << element << std::endl;
    }
}

Network<float> network;

int main(int argc, char** argv) {

    Dataset<float> dataset("./train.csv");

    network = Network<float>();
    network.print();

    for (int k = 0; k < EPOCH_COUNT * BATCH_SIZE; k++) {
        int j = k % BATCH_SIZE;
        std::vector<float> in(dataset.in[j]);

        std::vector<std::vector<float>> activations = network.ForwardPropagation(in);

        print(activations.back());


    }

    return 0;
}