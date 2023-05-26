
#include "matrix.hpp"

Vector::Vector()
{
    this->vector = std::vector<double>();
}

Vector::Vector(std::vector<double> v)
{
    this->vector = v;
}

Vector::Vector(const Vector &v)
{
    this->vector = std::vector<double>(v.vector);
}

void Vector::AppendToBack(double d) {
    this->vector.push_back(d);
}

Vector Vector::Apply(double (*f)(double))
{
    Vector toReturn = Vector(*this);

    for (int i = 0; i < this->Size(); i++)
    {
        toReturn.vector[i] = f(toReturn.vector[i]);
    }

    return toReturn;
}

double Vector::Get(int i) const
{
    assert(i < this->vector.size());
    return this->vector[i];
}

Matrix Vector::operator*(const Matrix &m)
{
    Matrix a = VectorToColumnMatrix(*this);

    return a * m;
}

double Vector::operator[](const int index) const
{
    assert(index >= 0);

    return this->vector[index];
}

Vector Vector::operator-(const Vector r) const {
    assert(this->Size() == r.Size());

    std::vector<double> in(this->Size());

    for (int i = 0; i < this->Size(); i++) {
        in[i] = (*this)[i] - r[i];
    }

    Vector toReturn(in);

    return toReturn;
}

int Vector::Size() const
{
    return this->vector.size();
}

void Vector::Print()
{
    std::cout << "---------- Vector Print ----------" << std::endl;

    for (double element : this->vector)
    {
        std::cout << element << std::endl;
    }
}

Matrix::Matrix() {
    this->matrix = std::vector<std::vector<double>>();
}

Matrix::Matrix(int row, int column)
{
    assert(row > 0);
    assert(column > 0);

    this->matrix = std::vector<std::vector<double>>(row);

    for (int i = 0; i < row; i++)
    {
        this->matrix[i] = std::vector<double>(column);
    }
}

Matrix::Matrix(std::vector<std::vector<double>> m)
{
    this->matrix = m;
}

int Matrix::Rows() const
{
    return this->matrix.size();
}

int Matrix::Columns() const
{
    return this->matrix[0].size();
}

Vector Matrix::operator*(const Vector &v)
{
    assert(this->Columns() == v.Size());

    std::vector<double> toReturn(this->Rows());

    for (int j = 0; j < this->Rows(); j++)
    {
        double t = 0;

        for (int i = 0; i < this->Columns(); i++)
        {
            t += this->matrix[j][i] * v[i];
        }

        toReturn[j] = t;
    }

    return toReturn;
}

Matrix Matrix::operator*(const Matrix &m)
{
    assert(this->Columns() == m.Rows());

    Matrix toReturn(this->Rows(), m.Columns());

    for (int i = 0; i < this->Rows(); i++)
    {
        for (int j = 0; j < m.Columns(); j++)
        {
            toReturn.matrix[i][j] = 0;

            for (int k = 0; k < this->Columns(); k++)
            {
                toReturn.matrix[i][j] += this->matrix[i][k] * m.matrix[k][j];
            }
        }
    }

    return toReturn;
}

Matrix Matrix::operator+(const Matrix &m) {
    assert(m.Rows() == this->Rows());
    assert(m.Columns() == this->Columns());

    Matrix toReturn(*this);

    for (int i = 0; i < this->Rows(); i++) {
        for (int j = 0; j < this->Columns(); j++) {
            double left = this->Get(i, j);
            double right = m.Get(i, j);
            toReturn.Set(i, j, left + right);
        }
    }

    return toReturn;
}

Matrix Matrix::operator/(const double &d) {
    Matrix toReturn(*this);

    for (int i = 0; i < this->Rows(); i++) {
        for (int j = 0; j < this->Columns(); j++) {
            double left = this->Get(i, j);
            toReturn.Set(i, j, left / d);
        }
    }

    return toReturn;
}

Matrix Matrix::operator*(const double &d)
{
    Matrix toReturn(*this);

    for (int i = 0; i < this->Rows(); i++) {
        for (int j = 0; j < this->Columns(); j++) {
            double left = this->Get(i, j);
            toReturn.Set(i, j, left * d);
        }
    }

    return toReturn;
}

Matrix Matrix::RemoveLastColumn() {
    Matrix toReturn = Matrix(this->Rows(), this->Columns() - 1);

    for (int i = 0; i < toReturn.Rows(); i++) {
        for (int j = 0; j < toReturn.Columns(); j++) {
            double t = this->Get(i, j);
            toReturn.Set(i, j, t);
        }
    }

    return toReturn;
}

Matrix Matrix::operator-(const Matrix &m)
{
    assert(m.Rows() == this->Rows());
    assert(m.Columns() == this->Columns());

    Matrix toReturn(*this);

    for (int i = 0; i < this->Rows(); i++) {
        for (int j = 0; j < this->Columns(); j++) {
            double left = this->Get(i, j);
            double right = m.Get(i, j);
            toReturn.Set(i, j, left - right);
        }
    }

    return toReturn;
}

Matrix Matrix::Transpose()
{
    Matrix toReturn(this->Columns(), this->Rows());

    for (int i = 0; i < this->Rows(); i++) {
        for (int j = 0; j < this->Columns(); j++) {
            toReturn.matrix[j][i] = this->matrix[i][j];
        }
    }

    return toReturn;
}

double Matrix::Get(int row, int column) const
{
    return this->matrix[row][column];
}

void Matrix::Set(int row, int column, double value) 
{
    this->matrix[row][column] = value;
}

void Matrix::Print()
{
    std::cout << "---------- Matrix Print ----------" << std::endl;

    for (int i = 0; i < this->Rows(); i++)
    {
        printf("%9.4f", this->matrix[i][0]);

        for (int j = 1; j < this->Columns(); j++)
        {
            std::cout << " ";
            printf("%9.4f", this->matrix[i][j]);
        }

        std::cout << std::endl;
}
}

Matrix operator*(const double c, const Matrix &A)
{
    std::vector<std::vector<double>> in(A.Rows());

    for (int i = 0; i < A.Rows(); i++) {
        in[i] = std::vector<double>(A.Columns());

        for (int j = 0; j < A.Columns(); j++) {
            in[i][j] = c * A.Get(i, j);
        }
    }

    Matrix toReturn(in);

    return toReturn;
}

Matrix Diag(Vector v) {
    std::vector<std::vector<double>> out(v.Size());

    for (int i = 0; i < v.Size(); i++) {
        out[i] = std::vector<double>(v.Size());

        for (int j = 0; j < v.Size(); j++) {
            out[i][j] = 0;
        }

        out[i][i] = v[i];
    }

    return Matrix(out);
}

Matrix Identity(int size) {
    std::vector<std::vector<double>> out(size);

    for (int i = 0; i < size; i++) {
        out[i] = std::vector<double>(size);

        for (int j = 0; j < size; j++) {
            out[i][j] = 0;
        }

        out[i][i] = 1;
    }

    Matrix toReturn(out);

    return toReturn;
}

Matrix VectorToRowMatrix(Vector v)
{
    std::vector<std::vector<double>> out(1);
    std::vector<double> in(v.Size());
    for (int i = 0; i < v.Size(); i++) {
        in[i] = v[i];
    }

    out[0] = in;
    
    Matrix toReturn(out);

    return toReturn;
}

Matrix VectorToColumnMatrix(Vector v)
{
    std::vector<std::vector<double>> row(1);
    std::vector<double> column(v.Size());

    for (int i = 0; i < v.Size(); i++)
    {
        column[i] = v[i];
    }

    row[0] = column;

    return Matrix(row);
}

Vector VectorOfAllOnes(int size) {
    std::vector<double> t(size);

    for (int i = 0; i < size; i++) {
        t[i] = 1;
    }

    Vector toReturn(t);

    return toReturn;
}