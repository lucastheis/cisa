#ifndef UTILS_H
#define UTILS_H

#include "Eigen/Core"
#include <vector>

using namespace Eigen;
using std::vector;

#define PI 3.141592653589793

Array<float, 1, Dynamic> logsumexp(const ArrayXXf& array);
Array<float, 1, Dynamic> logmeanexp(const ArrayXXf& array);

ArrayXXf sampleNormal(int m = 1, int n = 1);
ArrayXXf sampleGamma(int m = 1, int n = 1, int k = 1);

VectorXi argsort(const VectorXf& data);
MatrixXf covariance(const MatrixXf& data);
MatrixXf corrcoef(const MatrixXf& data);
MatrixXf normalize(const MatrixXf& matrix);

float logDetPD(const MatrixXf& matrix);

MatrixXf deleteRows(const MatrixXf& matrix, vector<int> indices);
MatrixXf deleteCols(const MatrixXf& matrix, vector<int> indices);

#endif
