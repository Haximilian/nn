#include <math.h>

#include "matrix.hpp"

Vector tanh_vector(Vector in)
{
    return in.Apply(tanh);
}

Matrix tanh_derivative(Vector activation)
{
    int rows = activation.Size();
    int columns = activation.Size();

    Matrix out(rows, columns);

    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < columns; j++)
        {
            double t;

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

Vector softmax_vector(Vector in)
{
    std::vector<double> out(in.Size());

    double sum = 0;
    for (int i = 0; i < in.Size(); i++)
    {
        sum += exp(in.Get(i));
    }

    for (int i = 0; i < in.Size(); i++)
    {
        out[i] = exp(in.Get(i)) / sum;
    }

    return Vector(out);
}

Matrix softmax_derivative(Vector activation)
{
    int rows = activation.Size();
    int columns = activation.Size();

    Matrix out(rows, columns);

    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < columns; j++)
        {
            double t;

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