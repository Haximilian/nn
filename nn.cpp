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

void Network::Print() {
    std::cout << "---------- Network Print ----------" << std::endl;
 
    for (Matrix matrix: this->weights) {
        matrix.Print();
    }
}

double cross_entropy(Vector predicted, Vector expected) {
    assert(predicted.Size() == expected.Size());

    double acc = 0;
    for (int i = 0; i < predicted.Size(); i++) {
        acc += expected.Get(i) * log(predicted.Get(i));
    }

    return -1 * acc;
}

Matrix tanh_derivative(Vector activation) {
    int rows = activation.Size();
    int columns = activation.Size();

    Matrix out(rows, columns);

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < columns; j++) {
            double t;

            if (i == j) {
                t = 1 - activation[i] * activation[j];
            } else {
                t = 0;
            }

            out.Set(i, j, t);
        }
    }

    return out;
}

Matrix softmax_derivative(Vector activation) {
    int rows = activation.Size();
    int columns = activation.Size();

    Matrix out(rows, columns);

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < columns; j++) {
            double t;

            if (i == j) {
                t = activation[i] * (1 - activation[j]);
            } else {
                t = activation[i] * (0 - activation[j]);
            }

            out.Set(i, j, t);
        }
    }

    return out;
}

std::vector<Matrix> Gradients(
    std::vector<Matrix> weights,
    std::vector<Vector> activations,
    Vector actual
) {
    // reverse activations
    // reverse weights

    std::vector<Matrix> activation_derivative(
        activations.size()
    );
    std::vector<Matrix> cumulative_derivative(
        activations.size()
    );

    activation_derivative[0] = Identity(activations[0].Size());

    int i = 1;
    for (; i < activations.size() - 1; i++) {
        activation_derivative[i] = tanh_derivative(activations[i]);
    }
    activation_derivative[i] = softmax_derivative(activations[i]);

    Matrix ha(1, actual.Size());
    for (int j = 0; j < actual.Size(); j++) {
        double t = -1 * actual.Get(j) / activations.back().Get(j);
        ha.Set(0, j, t);
    }

    cumulative_derivative[i] = ha * activation_derivative[i];
    
    for (; i > 0;) {
        i--;
        cumulative_derivative[i] = cumulative_derivative[i + 1] * weights[i].RemoveLastColumn() * activation_derivative[i];
    }

    // std::cout << "begin activation_derivative" << std::endl;
    // for (auto a:activation_derivative) {
    //     a.Print();
    // }
    // std::cout << "end activation_derivative" << std::endl;
    
    // std::cout << "begin cumulative_derivative" << std::endl;
    // for (auto c:cumulative_derivative) {
    //     c.Print();
    // }
    // std::cout << "end cumulative_derivative" << std::endl;
    
    std::vector<Matrix> network_derivative(weights.size());
    for (int i = 0; i < weights.size(); i++) {
        int rows = weights[i].Rows();
        int columns = weights[i].Columns();
        Matrix weight_derivative(rows, columns);

        for (int j = 0; j < rows; j++) {
            for (int k = 0; k < columns; k++) {
                std::vector<double> t(weights[i].Rows(), 0.0);

                t[j] = k >= activations[i].Size() ? 1 : activations[i].Get(k);
                Vector zw(t);
                double hw = (cumulative_derivative[i + 1] * zw).Get(0);
                weight_derivative.Set(j, k, hw);
            }
        }

        network_derivative[i] = weight_derivative;
    }

    // for (auto n:network_derivative) {
    //     n.Print();
    // }

    return network_derivative;
}