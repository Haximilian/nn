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

// double Normalize(double in) {
//     // std::cout << "Normalize Function" << std::endl;
//     // std::cout << in << std::endl;
//     return (in + 1) / 2;
// }

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

std::vector<Vector> Network::Activations(Vector in) {
    std::vector<Vector> activations(this->weights.size() + 1);

    activations[0] = Vector(in);

    for (int i = 1; i < activations.size(); i++) {
        activations[i] = Vector((*this->weights[this->weights.size() - i] * activations[i - 1]).Apply(tanh));
    }

    std::reverse(activations.begin(), activations.end());

    // activations[0] = activations[0].Apply(Normalize);
    // activations[0] = softmax(activations[0]);

    return activations;
}

Vector* Network::Residual(Vector* in, Vector* out) {
    return new Vector(*in - *out);
}

// std::vector<Matrix*> Network::Derivatives(std::vector<Vector*> activations, Vector* out)
// {
//     Matrix residuals = VectorToRowMatrix(*this->Residual(activations[0], out));

//     std::vector<Matrix*> derivatives(weights.size());

//     Matrix t = (-2 * residuals) * *Diag((*this->weights[0] * *activations[1]).Apply(TanHPrime));
//     Matrix derivative = t * VectorOfAllOnes(this->weights[0]->Rows()) * VectorToRowMatrix(*activations[1]);
//     derivatives[0] = new Matrix(derivative);

//     for (int i = 1; i < weights.size(); i++) {
//         t = t * *weights[i - 1] * *Diag((*weights[i] * *activations[1 + 1]).Apply(TanHPrime));
//         derivative = t * VectorOfAllOnes(weights[i]->Rows()) * VectorToRowMatrix(*activations[i + 1]);
//         derivatives[i] = new Matrix(derivative);
//     }

//     return derivatives;
// }

void Network::Print() {
    std::cout << "---------- Network Print ----------" << std::endl;

    for (Matrix* matrix: this->weights) {
        matrix->Print();
    }
}