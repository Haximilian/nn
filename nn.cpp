#include <math.h>

#include "nn.hpp" 

// returns value in range (0, 1)
double RandomBetweenZeroAndOne()
{
    double t = static_cast<double>(rand()) / static_cast<double>(RAND_MAX);

    return t;
}

double TanHPrime(double in) {
    double t = tanh(in);

    return (1.0 - t * t);
}

Vector TanH(Vector in) {
    return Vector(in.Apply(tanh));
}

Vector Softmax(Vector in) {
    std::vector<double> out(in.Size());

    double sum = 0;
    for (int i = 0; i < in.Size(); i++) {
        sum += exp(in.Get(i));
    }

    for (int i = 0; i < in.Size(); i++) {
        out[i] = exp(in.Get(i)) / sum;
    }

    return Vector(out);
}

NetworkMetadata::NetworkMetadata(std::vector<int> metadata) {
    assert(metadata.size() > 1);

    this->metadata = metadata;
}

int NetworkMetadata::operator[](int index) const 
{
    return this->metadata[index];
}

int NetworkMetadata::Size() const
{
    return this->metadata.size();
}

// todo: create a seperate method to initialize the activation functions
Network::Network(NetworkMetadata metadata) {
    this->weights = std::vector<Matrix>(metadata.Size() - 1);
    this->activation_fns = std::vector<activation_fn_ptr>(metadata.Size() - 1);

    for (int i = 1; i < metadata.Size(); i++) {
        int bias = 1;
        int rows = metadata[i];
        int columns = metadata[i - 1] + bias;

        std::vector<std::vector<double>> layer(rows);

        for (int j = 0; j < rows; j++) {
            layer[j] = std::vector<double>(columns);

            for (int k = 0; k < columns; k++) {
                layer[j][k] = RandomBetweenZeroAndOne() - 0.5;
            }
        }

        this->weights[i - 1] = Matrix(layer);
        this->activation_fns[i - 1] = *TanH;
    }

    this->activation_fns.back() = *Softmax;
}

std::vector<Vector> Network::ForwardPropagation(Vector in) {
    std::vector<Vector> activations(this->weights.size() + 1);

    activations[0] = Vector(in);

    for (int i = 1; i < activations.size(); i++) {
        Vector t(activations[i - 1]);
        t.AppendToBack(1.0);

        activations[i] = this->activation_fns[i - 1](this->weights[i - 1] * t);
    }

    return activations;
}

Vector* Network::Residual(Vector* in, Vector* out) {
    return new Vector(*in - *out);
}

double cross_entropy(Vector predicted, Vector expected) {
    assert(predicted.Size() == expected.Size());

    double acc = 0;
    for (int i = 0; i < predicted.Size(); i++) {
        acc += expected.Get(i) * log(predicted.Get(i));
    }

    return -1 * acc;
}

void Network::Epoch(std::vector<std::vector<double>> in, std::vector<std::vector<double>> out) {
    std::vector<Vector> updated_in; 
    for (auto row: in) {
        updated_in.push_back(Vector(row));
    }

    std::vector<Vector> updated_out; 
    for (auto row: out) {
        updated_out.push_back(Vector(row));
    }

    std::vector<std::vector<Vector>> activations;

    double loss = 0;

    // for (int i = 0; i < updated_in.size(); i++) {
    //     Vector row = updated_in[i];
    //     std::vector<Vector> activation = this->Activations(row);
    //     Vector predicted = activation[0];
    //     loss += cross_entropy(predicted, updated_out[i]);
    //     activations.push_back(activation);
    // }

    // for (int i = 5; i < 15; i++) {
    //     std::cout << "prediction" << std::endl;
    //     softmax(activations[i][0]).Print();
    //     std::cout << "expected" << std::endl;
    //     updated_out[i].Print();
    // }

    // std::cout << "Epoch Loss: " << loss << std::endl;
    std::cout << "Epoch Loss: " << loss / updated_in.size() << std::endl;

    // std::vector<Matrix> weights = dereference(this->weights);
    std::vector<Matrix> gradients = Gradients(
        weights,
        activations,
        updated_out
    );

    for (int i = 0; i < this->weights.size(); i++) {
        // (*this->weights[i]).Print();
        gradients[i].Print();
        // *this->weights[i] = weights[i] - (gradients[i] * 0.03);
    }
}

void Network::Print() {
    std::cout << "---------- Network Print ----------" << std::endl;

    for (Matrix matrix: this->weights) {
        matrix.Print();
    }
}

std::vector<Matrix> dereference(std::vector<Matrix*> in) {
    std::vector<Matrix> toReturn;
    for (Matrix* m: in) {
        toReturn.push_back(*m);
    }
    return toReturn;
}

// returns vector of matrix
// matrix dimensions
// sigma theta / sigma a[i]
// rows: dim(theta)
// columns: dim(a_{i})
std::vector<Matrix> SigmaThetaSigmaA(
    std::vector<Matrix> weights,
    std::vector<Vector> activations
)
{
    std::vector<Matrix> toReturn(activations.size());

    toReturn[0] = Identity(activations[0].Size());
    // remove column
    toReturn[1] = weights[0].RemoveLastColumn();

    for (int i = 2; i < activations.size(); i++) {
        Matrix weightWithoutBias = weights[i - 1].RemoveLastColumn(); 
        // current equals sigma a_{i - 1} / sigma a_{i}
        Matrix current = *Diag((weights[i - 1] * activations[i]).Apply(TanHPrime)) * weightWithoutBias;

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
    std::vector<Matrix> sigmaThetaSigmaA = SigmaThetaSigmaA(weights, activations);

    std::vector<Matrix> gradients(weights.size());

    Vector predicted = activations[0];

    Matrix sigmaCrossEntropySigmaPredicted = VectorToRowMatrix(predicted - actual);
    sigmaCrossEntropySigmaPredicted.Print();

    for (int i = 0; i < weights.size(); i++) {
        Matrix sigmaAsigmaZ = *Diag((weights[i] * activations[i + 1]).Apply(TanHPrime));
        if (i == 0) {
            sigmaAsigmaZ = Identity(weights[0].Rows());
        }

        gradients[i] = Matrix(weights[i].Rows(), weights[i].Columns());

        for (int j = 0; j < weights[i].Rows(); j++) {
            for (int k = 0; k < weights[i].Columns(); k++) {
                int sigmaZsigmaWSize = activations[i].Size(); 
                if (i > 0) {
                    sigmaZsigmaWSize = activations[i].Size() - 1;
                }
                std::vector<double> t(sigmaZsigmaWSize, 0.0);
                t[j] = activations[i + 1].Get(k);
                Vector sigmaZsigmaW = Vector(t);
                // sigmaAsigmaZ.Print();
                // sigmaZsigmaW.Print();

                Vector sigmaAsigmaW = sigmaAsigmaZ * sigmaZsigmaW;

                Vector sigmaJsigmaW = sigmaThetaSigmaA[i] * sigmaAsigmaW;
                // sigmaThetaSigmaA[i].Print();

                double gradient = (sigmaCrossEntropySigmaPredicted * sigmaJsigmaW).Get(0);
                // sigmaJsigmaW.Print();

                gradients[i].Set(j, k, gradient);
                // gradients[i].Print();
            }
        }
    }

    return gradients;
}

std::vector<Matrix> Gradients(
    std::vector<Matrix> weights,
    std::vector<std::vector<Vector>> activations,
    std::vector<Vector> actual
) 
{
    assert(activations.size() == actual.size());
    assert(activations.size() > 0);

    int size = activations.size();
    int batch = 1;
    int i = int((size - 1) * RandomBetweenZeroAndOne());

    // std::cout << i << std::endl;

    std::vector<Matrix> gradients = Gradients(
        weights,
        activations[i],
        actual[i]
    );

    // for (int i = 1; i < size; i++) {
    //     std::vector<Matrix> t = Gradients(
    //         weights,
    //         activations[i],
    //         actual[i]
    //     );

    //     for (int j = 0; j < gradients.size(); j++) {
    //         gradients[j] = gradients[j] + (t[j] / double(size));
    //     }
    // }

    return gradients;
}