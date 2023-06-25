#include <vector>

#include "include/rapidcsv.h"
#include "matrix.hpp"

template<typename T>
class Dataset
{
public:
    std::vector<std::vector<T>> in;
    std::vector<std::vector<T>> out;

    Dataset(std::string filename)
    {
        rapidcsv::Document document(filename);

        for (size_t i = 0; i < document.GetRowCount(); i++)
        {
            std::vector<T> t = document.GetRow<T>(i);

            std::vector<T> arg{
                t[0], t[1]};
            in.push_back(arg);

            if (t[2] > 0.5) {
                std::vector<T> res
                {
                    1, 0
                };
                out.push_back(res);
            } else {
                std::vector<T> res
                {
                    0, 1
                };
                out.push_back(res);
            }
        }
    }
};