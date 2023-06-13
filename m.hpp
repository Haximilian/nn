#include <array>

#include <iostream>

#include <stdio.h>

double return_zero() {
    return 0;
}

double return_random() {
    return static_cast<double>(rand()) / static_cast<double>(RAND_MAX) - 0.5;
}

float return_zero_float() {
    return 0;
}

float return_random_float() {
    return static_cast<float>(rand()) / static_cast<double>(RAND_MAX) - 0.5;
}

template <typename T, size_t R, size_t C>
class matrix {
    private:
        std::array<T, R * C> arr;

    public:
        matrix();
        
        matrix(T (*f)());

        double get(size_t, size_t) const;
        void set(size_t, size_t, T);

        std::array<T, R> operator*(const std::array<T, C>) const;
        
        template<size_t U, size_t V>
        matrix<T, R, V> operator*(const matrix<T, U, V>) const;
        matrix<T, R, C> operator-(const matrix<T, R, C>) const;

        matrix<T, R, C> operator*(const T) const;
        matrix<T, R, C> operator/(const T) const;

        matrix<T, R, C> remove_last_column() const;

        void print() const;
};