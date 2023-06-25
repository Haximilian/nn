#include "nn.hpp"
#include "der.hpp"

// #include "m.cpp"

// returns value in range (0, 1)
float RandomBetweenZeroAndOne()
{
    float t = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);

    return t;
}

NetworkMetadata::NetworkMetadata(std::vector<int> metadata)
{
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
template<typename T>
Network<T>::Network(NetworkMetadata metadata)
{
    this->weights = std::vector<Matrix<T>>(metadata.Size() - 1);
    this->activation_fns = std::vector<activation_fn_ptr>(metadata.Size() - 1);

    for (int i = 1; i < metadata.Size(); i++)
    {
        int bias = 1;
        int rows = metadata[i];
        int columns = metadata[i - 1] + bias;

        std::vector<std::vector<float>> layer(rows);

        for (int j = 0; j < rows; j++)
        {
            layer[j] = std::vector<float>(columns);

            for (int k = 0; k < columns; k++)
            {
                layer[j][k] = RandomBetweenZeroAndOne() - 0.5;
            }
        }

        this->weights[i - 1] = Matrix<T>(layer);
        this->activation_fns[i - 1] = *tanh_vector;
    }

    this->activation_fns.back() = *softmax_vector<float>;
}

template<typename T>
std::vector<Vector<T>> Network<T>::ForwardPropagation(Vector<T> in)
{
    std::vector<Vector<T>> activations(this->weights.size() + 1);

    activations[0] = Vector<T>(in);

    for (int i = 1; i < activations.size(); i++)
    {
        Vector<T> t(activations[i - 1]);
        t.AppendToBack(1.0);

        activations[i] = this->activation_fns[i - 1](this->weights[i - 1] * t);
    }

    return activations;
}

template<typename T>
void Network<T>::Print()
{
    std::cout << "---------- Network Print ----------" << std::endl;

    for (Matrix<T> matrix : this->weights)
    {
        matrix.Print();
    }
}

float cross_entropy(Vector<float> predicted, Vector<float> expected)
{
    assert(predicted.Size() == expected.Size());

    float acc = 0;
    for (int i = 0; i < predicted.Size(); i++)
    {
        acc += expected.Get(i) * log(predicted.Get(i));
    }

    return -1 * acc;
}

template<typename T>
std::vector<Matrix<T>> Network<T>::CalculateGradient(
    std::vector<Vector<T>> activations,
    Vector<T> actual)
{
    std::vector<Matrix<T>> activation_derivative(
        activations.size());
    std::vector<Matrix<T>> cumulative_derivative(
        activations.size());

    activation_derivative[0] = Identity(activations[0].Size());

    int i = 1;
    for (; i < activations.size() - 1; i++)
    {
        activation_derivative[i] = tanh_derivative(activations[i]);
    }
    activation_derivative[i] = softmax_derivative(activations[i]);

    Matrix<T> ha(1, actual.Size());
    for (int j = 0; j < actual.Size(); j++)
    {
        float t = -1 * actual.Get(j) / activations.back().Get(j);
        ha.Set(0, j, t);
    }

    cumulative_derivative[i] = ha * activation_derivative[i];

    for (; i > 0;)
    {
        i--;
        cumulative_derivative[i] = cumulative_derivative[i + 1] * weights[i].RemoveLastColumn() * activation_derivative[i];
    }

    std::vector<Matrix<T>> network_derivative(weights.size());
    for (int i = 0; i < weights.size(); i++)
    {
        int rows = weights[i].Rows();
        int columns = weights[i].Columns();
        Matrix<T> weight_derivative(rows, columns);

        for (int j = 0; j < rows; j++)
        {
            for (int k = 0; k < columns; k++)
            {
                std::vector<float> t(weights[i].Rows(), 0.0);

                t[j] = k >= activations[i].Size() ? 1 : activations[i].Get(k);
                Vector<T> zw(t);
                T hw = (cumulative_derivative[i + 1] * zw).Get(0);
                weight_derivative.Set(j, k, hw);
            }
        }

        network_derivative[i] = weight_derivative;
    }

    return network_derivative;
}