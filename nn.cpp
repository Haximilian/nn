#include <math.h>

#include "nn.hpp" 

// currently only returns positive values
double Random()
{
    double t = static_cast<double>(rand()) / static_cast<double>(RAND_MAX);

    return t;
}

double TanHPrime(double in) {
    double t = tanh(in);

    return (1.0 - t * t);
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

Network::Network(NetworkMetadata metadata) {
    std::vector<Matrix*> network(metadata.Size() - 1);

    for (int i = 1; i < metadata.Size(); i++) {
        int rows = metadata[i];
        int columns = metadata[i - 1];

        std::vector<std::vector<double>> layer(rows);

        for (int j = 0; j < rows; j++) {
            layer[j] = std::vector<double>(columns);

            for (int k = 0; k < columns; k++) {
                layer[j][k] = Random();
            }
        }

        network[i - 1] = new Matrix(layer);
    }

    std::reverse(network.begin(), network.end());

    this->weights = network;
}

std::vector<Vector*> Network::ForwardPropogation(Vector in) {
    std::vector<Vector*> activations(this->weights.size() + 1);

    activations[0] = new Vector(in);

    for (int i = 1; i < activations.size(); i++) {
        activations[i] = new Vector((*this->weights[this->weights.size() - i] * *activations[i - 1]).Apply(tanh));
    }

    std::reverse(activations.begin(), activations.end());

    return activations;
}

std::vector<Vector> Network::Activations(Vector in) {
    std::vector<Vector> activations(this->weights.size() + 1);

    activations[0] = Vector(in);

    for (int i = 1; i < activations.size(); i++) {
        activations[i] = Vector((*this->weights[this->weights.size() - i] * activations[i - 1]).Apply(tanh));
    }

    std::reverse(activations.begin(), activations.end());

    return activations;
}

Vector* Network::Residual(Vector* in, Vector* out) {
    return new Vector(*in - *out);
}

// double cross_entropy()


void Network::Epoch(std::vector<std::vector<double>> in, std::vector<std::vector<double>> out) {
    std::vector<Vector> updated_in; 

    std::vector<std::vector<Vector>> activations;

    double loss = 0;

    for (Vector row: updated_in) {
        std::vector<Vector> activation = this->Activations(row);
        activations.push_back(activation);
    }
}

void Network::Print() {
    std::cout << "---------- Network Print ----------" << std::endl;

    for (Matrix* matrix: this->weights) {
        matrix->Print();
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

Vector softmax(Vector in) {
    std::vector<double> out(in.Size());
    double sum = 0;
    for (int i = 0; i < out.size(); i++) {
        double v = in.Get(i);
        sum += v;
    }
    for (int i = 0; i < in.Size(); i++) {
        out[i] = in.Get(i) / sum;
    }
    return Vector(out);
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

    Vector predicted = softmax(activations[0]);

    Matrix sigmaCrossEntropySigmaPredicted = VectorToRowMatrix(actual - predicted);

    for (int i = 0; i < weights.size(); i++) {
        Matrix sigmaAsigmaZ = *Diag((weights[i] * activations[i + 1]).Apply(TanHPrime));

        gradients[i] = Matrix(weights[i].Rows(), weights[i].Columns());

        for (int j = 0; j < weights[i].Rows(); j++) {
            for (int k = 0; k < weights[i].Columns(); k++) {
                std::vector<double> t(activations[i].Size(), 0.0);
                t[j] = activations[i + 1].Get(k);
                Vector sigmaZsigmaW = Vector(t);

                Vector sigmaAsigmaW = sigmaAsigmaZ * sigmaZsigmaW;

                Vector sigmaJsigmaW = sigmaJsigmaA[i] * sigmaAsigmaW;

                double gradient = (sigmaCrossEntropySigmaPredicted * sigmaJsigmaW).Get(0);

                gradients[i].Set(j, k, gradient);
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

    std::vector<Matrix> gradients = Gradients(
        weights,
        activations[0],
        actual[0]
    );

    for (int i = 1; i < size; i++) {
        std::vector<Matrix> t = Gradients(
            weights,
            activations[i],
            actual[i]
        );

        for (int j = 0; j < gradients.size(); j++) {
            gradients[j] = gradients[j] + (t[j] / double(size));
        }
    }

    return gradients;
}