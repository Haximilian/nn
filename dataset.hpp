#include <vector>

#include "include/rapidcsv.h"
#include "matrix.hpp"

class Dataset
{
public:
    std::vector<std::vector<double>> in;
    std::vector<std::vector<double>> out;

    Dataset(std::string filename)
    {
        rapidcsv::Document document(filename);

        for (size_t i = 0; i < document.GetRowCount(); i++)
        {
            std::vector<double> t = document.GetRow<double>(i);

            std::vector<double> arg{
                t[0], t[1]};
            in.push_back(arg);

            if (t[2] > 0.5) {
                std::vector<double> res
                {
                    1, 0
                };
                out.push_back(res);
            } else {
                std::vector<double> res
                {
                    0, 1
                };
                out.push_back(res);
            }
        }
    }
};