#include <vector>
#include <math.h>

#include "m.hpp"

template<typename T>
std::vector<T> tanh_vector(std::vector<T> in)
{
    std::vector<T> out(in.size());

    for (int i = 0; i < in.size(); i++)
    {
        out[i] = tanh(in[i]);
    }

    return out;
}

template<typename T, size_t R, size_t C>
void tanh_derivative(std::unique_ptr<matrix<T, R, C>> in, std::vector<T> activation)
{
    for (int i = 0; i < R; i++)
    {
        for (int j = 0; j < C; j++)
        {
            T t;

            if (i == j)
            {
                t = 1 - activation[i] * activation[j];
            }
            else
            {
                t = 0;
            }

            in->set(i, j, t);
        }
    }
}


template<typename T>
std::vector<T> softmax_vector(std::vector<T> in)
{
    std::vector<T> out(in.size());

    float sum = 0;
    for (int i = 0; i < in.size(); i++)
    {
        sum += exp(in[i]);
    }

    for (int i = 0; i < in.size(); i++)
    {
        out[i] = exp(in[i]) / sum;
    }

    return out;
}

template<typename T>
struct Gradients {
{% for (i, R, C) in dimensions %}
    matrix<T, {{R}}, {{C}}> L_{{i}}_R_{{R}}_C_{{C}};
{% endfor %}
};

typedef std::vector<float> activation_fn(std::vector<float>);
typedef activation_fn* activation_fn_ptr;

template<typename T>
class Network
{
private:
{% for (i, R, C) in dimensions %}
    matrix<T, {{R}}, {{C}}> L_{{i}}_R_{{R}}_C_{{C}};
{% endfor %}

    std::vector<activation_fn_ptr> activation_fns;

public:
    Network();

    std::vector<std::vector<T>> ForwardPropagation(std::vector<T>);

    void Gradient(std::shared_ptr<Gradients<T>>, std::vector<std::vector<T>>, std::vector<T>);

    void print() const;
 };

template<typename T>
Network<T>::Network() :
{% for (i, R, C) in dimensions %}
    L_{{i}}_R_{{R}}_C_{{C}}(return_random_float) {% if (i + 1) != dimensions|length %},{% endif %}
{% endfor %}
{
    this->activation_fns = std::vector<activation_fn_ptr>({{ dimensions|length }});

    for (int i = 0; i < {{ dimensions|length }}; i++)
    {
        this->activation_fns[i] = *tanh_vector<T>;
    }

    this->activation_fns.back() = *softmax_vector<T>;
}

template<typename T>
std::vector<std::vector<T>> Network<T>::ForwardPropagation(std::vector<T> in)
{
    std::vector<std::vector<T>> activations({{ dimensions|length + 1 }});

    activations[0] = in;

{% for (i, R, C) in dimensions %}
    std::vector<T> t_{{i}}(activations[{{ i }}]);
    t_{{i}}.push_back(1.0);

    activations[{{ i + 1 }}] = L_{{i}}_R_{{R}}_C_{{C}} * t_{{i}};
{% endfor %}

    return activations;
}

template<typename T>
void Network<T>::Gradient(
    std::shared_ptr<Gradients<T>> gradients, 
    std::vector<std::vector<T>> activations,
    std::vector<T> actual)
{
{% for i, l_size in size %}
    std::shared_ptr<matrix<T, {{l_size}}, {{l_size}}>> D_{{i}}_R_{{l_size}}_C_{{l_size}} = new matrix<T, {{l_size}}, {{l_size}}>();
{% endfor %}

// first one need to be an identity

{% for i, l_size in size[1:-1] %}
    tanh_derivative(D_{{i}}_R_{{l_size}}_C_{{l_size}}, activations[{{ i }}]);
{% endfor %}
}

// template<typename T>
// std::vector<Matrix<T>> Network<T>::CalculateGradient(
//     std::vector<Vector<T>> activations,
//     Vector<T> actual)
// {
//     std::vector<Matrix<T>> activation_derivative(
//         activations.size());
//     std::vector<Matrix<T>> cumulative_derivative(
//         activations.size());

//     activation_derivative[0] = Identity(activations[0].Size());

//     int i = 1;
//     for (; i < activations.size() - 1; i++)
//     {
//         activation_derivative[i] = tanh_derivative(activations[i]);
//     }
//     activation_derivative[i] = softmax_derivative(activations[i]);

//     Matrix<T> ha(1, actual.Size());
//     for (int j = 0; j < actual.Size(); j++)
//     {
//         float t = -1 * actual.Get(j) / activations.back().Get(j);
//         ha.Set(0, j, t);
//     }

//     cumulative_derivative[i] = ha * activation_derivative[i];

//     for (; i > 0;)
//     {
//         i--;
//         cumulative_derivative[i] = cumulative_derivative[i + 1] * weights[i].RemoveLastColumn() * activation_derivative[i];
//     }

//     std::vector<Matrix<T>> network_derivative(weights.size());
//     for (int i = 0; i < weights.size(); i++)
//     {
//         int rows = weights[i].Rows();
//         int columns = weights[i].Columns();
//         Matrix<T> weight_derivative(rows, columns);

//         for (int j = 0; j < rows; j++)
//         {
//             for (int k = 0; k < columns; k++)
//             {
//                 std::vector<float> t(weights[i].Rows(), 0.0);

//                 t[j] = k >= activations[i].Size() ? 1 : activations[i].Get(k);
//                 Vector<T> zw(t);
//                 T hw = (cumulative_derivative[i + 1] * zw).Get(0);
//                 weight_derivative.Set(j, k, hw);
//             }
//         }

//         network_derivative[i] = weight_derivative;
//     }

//     return network_derivative;
// }

template<typename T>
void Network<T>::print() const 
{
{% for (i, R, C) in dimensions %}
    L_{{i}}_R_{{R}}_C_{{C}}.print();
{% endfor %}
}