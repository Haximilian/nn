#include <math.h>

#include "matrix.hpp"

Vector<float> tanh_vector(Vector<float> in)
{
    return in.Apply(tanh);
}

Matrix<float> tanh_derivative(Vector<float> activation)
{
    int rows = activation.Size();
    int columns = activation.Size();

    Matrix<float> out(rows, columns);

    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < columns; j++)
        {
            float t;

            if (i == j)
            {
                t = 1 - activation[i] * activation[j];
            }
            else
            {
                t = 0;
            }

            out.Set(i, j, t);
        }
    }

    return out;
}

template<typename T>
Vector<T> softmax_vector(Vector<T> in)
{
    std::vector<float> out(in.Size());

    float sum = 0;
    for (int i = 0; i < in.Size(); i++)
    {
        sum += exp(in.Get(i));
    }

    for (int i = 0; i < in.Size(); i++)
    {
        out[i] = exp(in.Get(i)) / sum;
    }

    return Vector<T>(out);
}

template<typename T>
Matrix<T> softmax_derivative(Vector<T> activation)
{
    int rows = activation.Size();
    int columns = activation.Size();

    Matrix<T> out(rows, columns);

    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < columns; j++)
        {
            float t;

            if (i == j)
            {
                t = activation[i] * (1 - activation[j]);
            }
            else
            {
                t = activation[i] * (0 - activation[j]);
            }

            out.Set(i, j, t);
        }
    }

    return out;
}