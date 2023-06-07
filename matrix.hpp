#pragma once

#include <vector>
#include <iostream>

#include <stdio.h>
#include <cassert>

class Matrix;

class Vector
{
public:
    Vector();
    Vector(std::vector<double> v);
    Vector(const Vector &v);
    Vector Apply(double (*f)(double));
    double operator[](const int index) const;
    Vector operator-(const Vector r) const;
    int Size() const;
    double Get(int i) const;
    void Print();
    void AppendToBack(double d);

private:
    std::vector<double> vector;
};

class Matrix
{
public:
    Matrix();
    Matrix(int row, int columns);
    Matrix(std::vector<std::vector<double>> m);
    int Rows() const;
    int Columns() const;
    Vector operator*(const Vector &v);
    Matrix operator*(const Matrix &m);
    Matrix operator/(const double &d);
    Matrix operator*(const double &d);
    Matrix operator-(const Matrix &m);
    Matrix Transpose();
    double Get(int row, int column) const;
    void Set(int row, int column, double value);
    void Print();
    Matrix RemoveLastColumn();

private:
    int rows;
    int columns;
    std::vector<double> matrix;
};

Matrix operator*(const double c, const Matrix &A);

Matrix Diag(Vector v);

Matrix Identity(int size);