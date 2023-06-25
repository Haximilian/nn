#pragma once

#include <vector>
#include <iostream>

#include <stdio.h>
#include <cassert>

template<typename T>
class Vector
{
public:
    Vector();
    Vector(std::vector<T> v);
    Vector(const Vector &v);
    Vector Apply(T (*f)(T));
    T operator[](const int index) const;
    int Size() const;
    T Get(int i) const;
    void Print();
    void AppendToBack(T d);

private:
    std::vector<T> vector;
};

template<typename T>
class Matrix
{
public:
    Matrix();
    Matrix(int row, int columns);
    Matrix(std::vector<std::vector<T>> m);
    int Rows() const;
    int Columns() const;
    Vector<T> operator*(const Vector<T> &v);
    Matrix<T> operator*(const Matrix<T> &m);
    Matrix<T> operator/(const T &d);
    Matrix<T> operator*(const T &d);
    Matrix<T> operator-(const Matrix<T> &m);
    T Get(int row, int column) const;
    void Set(int row, int column, T value);
    void Print();
    Matrix<T> RemoveLastColumn();

private:
    int rows;
    int columns;
    std::vector<T> matrix;
};

template<typename T>
Matrix<T> Identity(int size);





template<typename T>
Vector<T>::Vector()
{
    this->vector = std::vector<T>();
}

template<typename T>
Vector<T>::Vector(std::vector<T> v)
{
    this->vector = v;
}

template<typename T>
Vector<T>::Vector(const Vector<T> &v)
{
    this->vector = std::vector<T>(v.vector);
}

template<typename T>
void Vector<T>::AppendToBack(T d)
{
    this->vector.push_back(d);
}

template<typename T>
Vector<T> Vector<T>::Apply(T (*f)(T))
{
    Vector toReturn = Vector(*this);

    for (int i = 0; i < this->Size(); i++)
    {
        toReturn.vector[i] = f(toReturn.vector[i]);
    }

    return toReturn;
}

template<typename T>
T Vector<T>::Get(int i) const
{
#ifdef DYNAMIC_ASSERT
    assert(i < this->vector.size());
#endif

    return this->vector[i];
}

template<typename T>
T Vector<T>::operator[](const int index) const
{
#ifdef DYNAMIC_ASSERT
    assert(index >= 0);
#endif

    return this->vector[index];
}


template<typename T>
int Vector<T>::Size() const
{
    return this->vector.size();
}

template<typename T>
void Vector<T>::Print()
{
    std::cout << "---------- Vector Print ----------" << std::endl;

    for (T element : this->vector)
    {
        std::cout << element << std::endl;
    }
}

template<typename T>
Matrix<T>::Matrix()
{
    this->matrix = std::vector<T>();
}

template<typename T>
Matrix<T>::Matrix(int in_rows, int in_columns)
{
#ifdef DYNAMIC_ASSERT
    assert(in_rows > 0);
    assert(in_columns > 0);
#endif

    this->rows = in_rows;
    this->columns = in_columns;

    int size = this->rows * this->columns;
    this->matrix = std::vector<T>(size);
}

template<typename T>
Matrix<T>::Matrix(std::vector<std::vector<T>> in)
{
#ifdef DYNAMIC_ASSERT
    assert(in.size() > 0);
    assert(in.front().size() > 0);
#endif

    this->rows = in.size();
    this->columns = in.front().size();

    int size = this->rows * this->columns;
    this->matrix = std::vector<T>(size);

    for (int i = 0; i < this->Rows(); i++) {
        for (int j = 0; j < this->Columns(); j++) {
            this->Set(i, j, in[i][j]);
        }
    }
}

template<typename T>
int Matrix<T>::Rows() const
{
    return this->rows;
}

template<typename T>
int Matrix<T>::Columns() const
{
    return this->columns;
}

template<typename T>
Vector<T> Matrix<T>::operator*(const Vector<T> &v)
{
#ifdef DYNAMIC_ASSERT
    assert(this->Columns() == v.Size());
#endif

    std::vector<T> out(this->Rows());

    for (int j = 0; j < this->Rows(); j++)
    {
        T t = 0;

        for (int i = 0; i < this->Columns(); i++)
        {
            t += this->Get(j, i) * v[i];
        }

        out[j] = t;
    }

    return out;
}

template<typename T>
Matrix<T> Matrix<T>::operator*(const Matrix<T> &m)
{
#ifdef DYNAMIC_ASSERT
    assert(this->Columns() == m.Rows());
#endif

    Matrix<T> toReturn(this->Rows(), m.Columns());

    for (int i = 0; i < this->Rows(); i++)
    {
        for (int j = 0; j < m.Columns(); j++)
        {
            T t = 0;

            for (int k = 0; k < this->Columns(); k++)
            {
                t += this->Get(i, k) * m.Get(k, j);
            }

            toReturn.Set(i, j, t);
        }
    }

    return toReturn;
}

template<typename T>
Matrix<T> Matrix<T>::operator/(const T &d)
{
    Matrix<T> toReturn(*this);

    for (int i = 0; i < this->Rows(); i++)
    {
        for (int j = 0; j < this->Columns(); j++)
        {
            T left = this->Get(i, j);
            toReturn.Set(i, j, left / d);
        }
    }

    return toReturn;
}

template<typename T>
Matrix<T> Matrix<T>::operator*(const T &d)
{
    Matrix<T> toReturn(*this);

    for (int i = 0; i < this->Rows(); i++)
    {
        for (int j = 0; j < this->Columns(); j++)
        {
            T left = this->Get(i, j);
            toReturn.Set(i, j, left * d);
        }
    }

    return toReturn;
}

template<typename T>
Matrix<T> Matrix<T>::RemoveLastColumn()
{
    Matrix<T> toReturn = Matrix(this->Rows(), this->Columns() - 1);

    for (int i = 0; i < toReturn.Rows(); i++)
    {
        for (int j = 0; j < toReturn.Columns(); j++)
        {
            T t = this->Get(i, j);
            toReturn.Set(i, j, t);
        }
    }

    return toReturn;
}

template<typename T>
Matrix<T> Matrix<T>::operator-(const Matrix<T> &m)
{
#ifdef DYNAMIC_ASSERT
    assert(m.Rows() == this->Rows());
    assert(m.Columns() == this->Columns());
#endif

    Matrix<T> toReturn(*this);

    for (int i = 0; i < this->Rows(); i++)
    {
        for (int j = 0; j < this->Columns(); j++)
        {
            T left = this->Get(i, j);
            T right = m.Get(i, j);
            toReturn.Set(i, j, left - right);
        }
    }

    return toReturn;
}

template<typename T>
T Matrix<T>::Get(int row, int column) const
{
    return this->matrix[row * this->Columns() + column];
}

template<typename T>
void Matrix<T>::Set(int row, int column, T value)
{
#ifdef DYNAMIC_ASSERT
    assert(row <= this->Rows());
    assert(column <= this->Columns());
#endif

    this->matrix[row * this->Columns() + column] = value;
}

template<typename T>
void Matrix<T>::Print()
{
    std::cout << "---------- Matrix Print ----------" << std::endl;

    for (int i = 0; i < this->Rows(); i++)
    {
        printf("%9.4f", this->Get(i, 0));

        for (int j = 1; j < this->Columns(); j++)
        {
            std::cout << " ";
            printf("%9.4f", this->Get(i, j));
        }

        std::cout << std::endl;
    }
}

template<typename T>
Matrix<T> operator*(const T c, const Matrix<T> &A)
{
    std::vector<std::vector<T>> in(A.Rows());

    for (int i = 0; i < A.Rows(); i++)
    {
        in[i] = std::vector<T>(A.Columns());

        for (int j = 0; j < A.Columns(); j++)
        {
            in[i][j] = c * A.Get(i, j);
        }
    }

    Matrix<T> toReturn(in);

    return toReturn;
}

template<typename T>
Matrix<T> Identity(int size)
{
    std::vector<std::vector<T>> out(size);

    for (int i = 0; i < size; i++)
    {
        out[i] = std::vector<T>(size);

        for (int j = 0; j < size; j++)
        {
            out[i][j] = 0;
        }

        out[i][i] = 1;
    }

    Matrix<T> toReturn(out);

    return toReturn;
}