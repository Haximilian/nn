#include <vector>

#include "matrix.hpp"

double TanHPrime(double);

std::vector<Matrix> dereference(std::vector<Matrix*> in) {
    std::vector<Matrix> toReturn;
    for (Matrix* m: in) {
        toReturn.push_back(*m);
    }
    return toReturn;
}

// returns vector of matrix
// matrix dimensions
// rows: dim(a_{0})
// columns: dim(a_{i})
std::vector<Matrix> SigmaJsigmaA(
    std::vector<Matrix> weights,
    std::vector<Vector> activations
)
{
    std::vector<Matrix> toReturn(activations.size());

    toReturn[0] = Identity(activations[0].Size());

    for (int i = 1; i < activations.size(); i++) {
        // current equals sigma a_{i - 1} / sigma a_{i}
        Matrix current = *Diag((weights[i - 1] * activations[i]).Apply(TanHPrime)) * weights[i - 1];

        toReturn[i] = toReturn[i - 1] * current;
    }

    return toReturn;
}

// calculates the gradient for a given input
// activation[0] equals (tanh(z) + 1) / 2
// it's also the output of the neural network
std::vector<Matrix> Gradients(
    std::vector<Matrix> weights,
    std::vector<Vector> activations,
    Vector actual
)
{
    std::vector<Matrix> sigmaJsigmaA = SigmaJsigmaA(weights, activations);

    std::vector<Matrix> gradients(weights.size());

    for (int i = 0; i < weights.size(); i++) {
        Matrix sigmaAsigmaZ = *Diag((weights[i] * activations[i + 1]).Apply(TanHPrime));

        gradients[i] = Matrix(weights[i].Rows(), weights[i].Columns());

        for (int j = 0; j < weights[i].Rows(); j++) {
            for (int k = 0; k < weights[i].Columns(); k++) {
                std::vector<double> t(activations[i].Size(), 0.0);

                t[j] = activations[i + 1].Get(k);

                Vector sigmaZsigmaW = Vector(t);

                std::cout << "sigmaAsigmaZ" << std::endl;
                sigmaAsigmaZ.Print();
                std::cout << "sigmaZsigmaW" << std::endl;
                sigmaZsigmaW.Print();

                Vector sigmaAsigmaW = sigmaAsigmaZ * sigmaZsigmaW;

                Vector sigmaJsigmaW = sigmaJsigmaA[i] * sigmaAsigmaW;

                // (output_of_nn + 1) / 2
                // this transformation makes it into a probability
                // cross entropy loss...
                Matrix sigmaCrossEntropySigmaPredicted = VectorToRowMatrix(actual - activations[0]);

                double gradient = (sigmaCrossEntropySigmaPredicted * sigmaJsigmaW).Get(0);

                gradients[i].Set(j, k, gradient);
            }
        }
    }

    return gradients;
}