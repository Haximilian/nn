// #define DYNAMIC_ASSERT

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

void Vector::AppendToBack(double d)
{
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
#ifdef DYNAMIC_ASSERT
    assert(i < this->vector.size());
#endif

    return this->vector[i];
}

Matrix Vector::operator*(const Matrix &m)
{
    Matrix a = VectorToColumnMatrix(*this);

    return a * m;
}

double Vector::operator[](const int index) const
{
#ifdef DYNAMIC_ASSERT
    assert(index >= 0);
#endif

    return this->vector[index];
}

Vector Vector::operator-(const Vector r) const
{
#ifdef DYNAMIC_ASSERT
    assert(this->Size() == r.Size());
#endif

    std::vector<double> in(this->Size());

    for (int i = 0; i < this->Size(); i++)
    {
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

Matrix::Matrix()
{
    this->matrix = std::vector<double>();
}

Matrix::Matrix(int in_rows, int in_columns)
{
#ifdef DYNAMIC_ASSERT
    assert(in_rows > 0);
    assert(in_columns > 0);
#endif

    this->rows = in_rows;
    this->columns = in_columns;

    int size = this->rows * this->columns;
    this->matrix = std::vector<double>(size);
}

Matrix::Matrix(std::vector<std::vector<double>> in)
{
#ifdef DYNAMIC_ASSERT
    assert(in.size() > 0);
    assert(in.front().size() > 0);
#endif

    this->rows = in.size();
    this->columns = in.front().size();

    int size = this->rows * this->columns;
    this->matrix = std::vector<double>(size);

    for (int i = 0; i < this->Rows(); i++) {
        for (int j = 0; j < this->Columns(); j++) {
            this->Set(i, j, in[i][j]);
        }
    }
}

int Matrix::Rows() const
{
    return this->rows;
}

int Matrix::Columns() const
{
    return this->columns;
}

Vector Matrix::operator*(const Vector &v)
{
#ifdef DYNAMIC_ASSERT
    assert(this->Columns() == v.Size());
#endif

    std::vector<double> out(this->Rows());

    for (int j = 0; j < this->Rows(); j++)
    {
        double t = 0;

        for (int i = 0; i < this->Columns(); i++)
        {
            t += this->Get(j, i) * v[i];
        }

        out[j] = t;
    }

    return out;
}

Matrix Matrix::operator*(const Matrix &m)
{
#ifdef DYNAMIC_ASSERT
    assert(this->Columns() == m.Rows());
#endif

    Matrix toReturn(this->Rows(), m.Columns());

    for (int i = 0; i < this->Rows(); i++)
    {
        for (int j = 0; j < m.Columns(); j++)
        {
            double t = 0;

            for (int k = 0; k < this->Columns(); k++)
            {
                t += this->Get(i, k) * m.Get(k, j);
            }

            toReturn.Set(i, j, t);
        }
    }

    return toReturn;
}

Matrix Matrix::operator+(const Matrix &m)
{
#ifdef DYNAMIC_ASSERT
    assert(m.Rows() == this->Rows());
    assert(m.Columns() == this->Columns());
#endif

    Matrix toReturn(*this);

    for (int i = 0; i < this->Rows(); i++)
    {
        for (int j = 0; j < this->Columns(); j++)
        {
            double left = this->Get(i, j);
            double right = m.Get(i, j);
            toReturn.Set(i, j, left + right);
        }
    }

    return toReturn;
}

Matrix Matrix::operator/(const double &d)
{
    Matrix toReturn(*this);

    for (int i = 0; i < this->Rows(); i++)
    {
        for (int j = 0; j < this->Columns(); j++)
        {
            double left = this->Get(i, j);
            toReturn.Set(i, j, left / d);
        }
    }

    return toReturn;
}

Matrix Matrix::operator*(const double &d)
{
    Matrix toReturn(*this);

    for (int i = 0; i < this->Rows(); i++)
    {
        for (int j = 0; j < this->Columns(); j++)
        {
            double left = this->Get(i, j);
            toReturn.Set(i, j, left * d);
        }
    }

    return toReturn;
}

Matrix Matrix::RemoveLastColumn()
{
    Matrix toReturn = Matrix(this->Rows(), this->Columns() - 1);

    for (int i = 0; i < toReturn.Rows(); i++)
    {
        for (int j = 0; j < toReturn.Columns(); j++)
        {
            double t = this->Get(i, j);
            toReturn.Set(i, j, t);
        }
    }

    return toReturn;
}

Matrix Matrix::operator-(const Matrix &m)
{
#ifdef DYNAMIC_ASSERT
    assert(m.Rows() == this->Rows());
    assert(m.Columns() == this->Columns());
#endif

    Matrix toReturn(*this);

    for (int i = 0; i < this->Rows(); i++)
    {
        for (int j = 0; j < this->Columns(); j++)
        {
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

    for (int i = 0; i < this->Rows(); i++)
    {
        for (int j = 0; j < this->Columns(); j++)
        {
            toReturn.Set(j, i, this->Get(i, j));
        }
    }

    return toReturn;
}

double Matrix::Get(int row, int column) const
{
    return this->matrix[row * this->Columns() + column];
}

void Matrix::Set(int row, int column, double value)
{
#ifdef DYNAMIC_ASSERT
    assert(row <= this->Rows());
    assert(column <= this->Columns());
#endif

    this->matrix[row * this->Columns() + column] = value;
}

void Matrix::Print()
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

Matrix operator*(const double c, const Matrix &A)
{
    std::vector<std::vector<double>> in(A.Rows());

    for (int i = 0; i < A.Rows(); i++)
    {
        in[i] = std::vector<double>(A.Columns());

        for (int j = 0; j < A.Columns(); j++)
        {
            in[i][j] = c * A.Get(i, j);
        }
    }

    Matrix toReturn(in);

    return toReturn;
}

Matrix Diag(Vector v)
{
    std::vector<std::vector<double>> out(v.Size());

    for (int i = 0; i < v.Size(); i++)
    {
        out[i] = std::vector<double>(v.Size());

        for (int j = 0; j < v.Size(); j++)
        {
            out[i][j] = 0;
        }

        out[i][i] = v[i];
    }

    return Matrix(out);
}

Matrix Identity(int size)
{
    std::vector<std::vector<double>> out(size);

    for (int i = 0; i < size; i++)
    {
        out[i] = std::vector<double>(size);

        for (int j = 0; j < size; j++)
        {
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
    for (int i = 0; i < v.Size(); i++)
    {
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

Vector VectorOfAllOnes(int size)
{
    std::vector<double> t(size);

    for (int i = 0; i < size; i++)
    {
        t[i] = 1;
    }

    Vector toReturn(t);

    return toReturn;
}