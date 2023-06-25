#pragma once

#include "m.hpp"

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
