#pragma once

#include <array>
#include <iostream>
#include <stdio.h>

#include "matrix.hpp"

double return_zero()
{
    return 0;
}

double return_random()
{
    return static_cast<double>(rand()) / static_cast<double>(RAND_MAX) - 0.5;
}

float return_zero_float()
{
    return 0;
}

float return_random_float()
{
    return static_cast<float>(rand()) / static_cast<float>(RAND_MAX) - 0.5;
}

template <typename T>
class abstract_matrix
{
    virtual std::vector<T> operator*(const std::vector<T>) const = 0;

    virtual abstract_matrix<T> operator*(const abstract_matrix<T>) const = 0;
    virtual abstract_matrix<T> operator-(const abstract_matrix<T>) const = 0;

    virtual abstract_matrix<T> operator*(const T) const = 0;
    virtual abstract_matrix<T> operator/(const T) const = 0;

    virtual abstract_matrix<T> remove_last_column() const = 0;

    virtual void print() const = 0;
};

template <typename T, size_t R, size_t C>
class matrix
{
private:
    std::array<T, R * C> arr;

public:
    matrix();

    matrix(T (*f)());

    matrix(std::array<T, R * C>, T (*f)());

    T get(size_t, size_t) const;
    void set(size_t, size_t, T);

    std::array<T, R> operator*(const std::array<T, C>) const;
    std::vector<T> operator*(const std::vector<T>) const;

    template <size_t U, size_t V>
    matrix<T, R, V> operator*(const matrix<T, U, V>) const;
    matrix<T, R, C> operator-(const matrix<T, R, C>) const;

    matrix<T, R, C> operator*(const T) const;
    matrix<T, R, C> operator/(const T) const;

    matrix<T, R, C> remove_last_column() const;

    void print() const;
};

template<typename T, size_t R, size_t C>
matrix<T, R, C>::matrix() {
    this->arr = std::array<T, R * C>();
}

template<typename T, size_t R, size_t C>
matrix<T, R, C>::matrix(T (*f)()) {
    matrix<T, R, C>();

    for (size_t i = 0; i < R; i++)
    {
        T t = 0;

        for (int j = 0; j < C; j++)
        {
            this->set(i, j, f());
        }
    }
}

template<typename T, size_t R, size_t C>
matrix<T, R, C>::matrix(std::array<T, R * C> arr, T(*f)()) {
    this->arr = arr;

    for (size_t i = 0; i < R; i++)
    {
        T t = 0;

        for (int j = 0; j < C; j++)
        {
            this->set(i, j, f());
        }
    }
}

template<typename T, size_t R, size_t C>
T matrix<T, R, C>::get(size_t i, size_t j) const {
    return this->arr[C * i + j];
}

template<typename T, size_t R, size_t C>
void matrix<T, R, C>::set(size_t i, size_t j, T v) {
    this->arr[C * i + j] = v;
}

template<typename T, size_t R, size_t C>
std::array<T, R> matrix<T, R, C>::operator*(const std::array<T, C> arr) const {
    std::array<T, R> out{};

    for (size_t i = 0; i < R; i++)
    {
        T t = 0;

        for (int j = 0; j < C; j++)
        {
            t += this->get(i, j) * arr[j];
        }

        out[i] = t;
    }

    return out;
}

template<typename T, size_t R, size_t C>
std::vector<T> matrix<T, R, C>::operator*(const std::vector<T>) const {
    std::vector<T> out(R);

    for (size_t i = 0; i < R; i++)
    {
        T t = 0;

        for (int j = 0; j < C; j++)
        {
            t += this->get(i, j) * arr[j];
        }

        out[i] = t;
    }

    return out;
};

template<typename T, size_t R, size_t C>
template<size_t U, size_t V>
matrix<T, R, V> matrix<T, R, C>::operator*(const matrix<T, U, V> in) const {
    matrix<T, R, V> out{};

    for (int i = 0; i < R; i++)
    {
        for (int j = 0; j < V; j++)
        {
            T t = 0;

            for (int k = 0; k < C; k++)
            {
                t += this->get(i, k) * in.get(k, j);
            }

            out.set(i, j, t);
        }
    }

    return out;
}

template<typename T, size_t R, size_t C>
matrix<T, R, C> matrix<T, R, C>::operator-(const matrix<T, R, C> in) const {
    matrix<T, R, C> out{};

    for (int i = 0; i < R; i++)
    {
        for (int j = 0; j < C; j++)
        {
            T t = this->get(i, j) - in.get(i, j);

            out.set(i, j, t);
        }
    }

    return out;
}

template<typename T, size_t R, size_t C>
matrix<T, R, C> matrix<T, R, C>::operator*(const T sc) const {
    matrix<T, R, C> out{};

    for (int i = 0; i < R; i++)
    {
        for (int j = 0; j < C; j++)
        {
            T t = this->get(i, j) * sc;

            out.set(i, j, t);
        }
    }

    return out;
}

template<typename T, size_t R, size_t C>
matrix<T, R, C> matrix<T, R, C>::operator/(const T sc) const {
    matrix<T, R, C> out{};

    for (int i = 0; i < R; i++)
    {
        for (int j = 0; j < C; j++)
        {
            T t = this->get(i, j) / sc;

            out.set(i, j, t);
        }
    }

    return out;
}

template<typename T, size_t R, size_t C>
matrix<T, R, C> matrix<T, R, C>::remove_last_column() const {
   matrix<T, R, C - 1> out{};

    for (int i = 0; i < R; i++)
    {
        for (int j = 0; j < C - 1; j++)
        {
            T t = this->get(i, j);

            out.set(i, j, t);
        }
    }

    return out;
}

template<typename T, size_t R, size_t C>
void matrix<T, R, C>::print() const {
    std::cout << "---------- Matrix Print ----------" << std::endl;

    for (int i = 0; i < R; i++)
    {
        printf("%9.4f", this->get(i, 0));

        for (int j = 1; j < C; j++)
        {
            std::cout << " ";
            printf("%9.4f", this->get(i, j));
        }

        std::cout << std::endl;
    }
}
