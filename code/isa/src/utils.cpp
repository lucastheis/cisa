#include "Eigen/Cholesky"
#include "utils.h"
#include <algorithm>
#include <vector>
#include <iostream>
#include <cstdlib>

#ifdef __GXX_EXPERIMENTAL_CXX0X__
#include <random>
#endif

using namespace std;

Array<float, 1, Dynamic> logsumexp(const ArrayXXf& array) {
	Array<float, 1, Dynamic> arrayMax = array.colwise().maxCoeff() - 1.;
	return arrayMax + (array.rowwise() - arrayMax).exp().colwise().sum().log();
}



Array<float, 1, Dynamic> logmeanexp(const ArrayXXf& array) {
	Array<float, 1, Dynamic> arrayMax = array.colwise().maxCoeff() - 1.;
	return arrayMax + (array.rowwise() - arrayMax).exp().colwise().mean().log();
}



#ifdef __GXX_EXPERIMENTAL_CXX0X__
ArrayXXf sampleNormal(int m, int n) {
	mt19937 gen(rand());
	normal_distribution<float> normal;
	ArrayXXf samples(m, n);

	for(int i = 0; i < samples.size(); ++i)
		samples(i) = normal(gen);

	return samples;
}
#else
#warning "No C++11 support. Using my own implementation of the Box-Muller transform."
ArrayXXf sampleNormal(int m, int n) {
	ArrayXXf U1 = ArrayXXf::Random(m, n).abs();
	ArrayXXf U2 = ArrayXXf::Random(m, n).abs();
	// Box-Muller transform
	return (-2. * U1.log()).sqrt() * (2. * PI * U2).cos();
}
#endif



ArrayXXf sampleGamma(int m, int n, int k) {
	ArrayXXf samples = ArrayXXf::Zero(m, n);

	for(int i = 0; i < k; ++i)
		samples -= ArrayXXf::Random(m, n).abs().log();

	return samples;
}



VectorXi argsort(const VectorXf& data) {
	// create pairs of values and indices
	vector<pair<float, int> > pairs(data.size());
	for(int i = 0; i < data.size(); ++i) {
		pairs[i].first = data[i];
		pairs[i].second = i;
	}

	// sort values in descending order
	sort(pairs.begin(), pairs.end(), greater<pair<float, int> >());

	// store indices
	VectorXi indices(data.size());
	for(int i = 0; i < data.size(); ++i)
		indices[pairs[i].second] = i;

	return indices;
}



MatrixXf covariance(const MatrixXf& data) {
	MatrixXf data_centered = data.colwise() - data.rowwise().mean().eval();
	return data_centered * data_centered.transpose() / data.cols();
}



MatrixXf corrcoef(const MatrixXf& data) {
	MatrixXf C = covariance(data);
	VectorXf c = C.diagonal();
	return C.array() / (c * c.transpose()).array().sqrt();
}



MatrixXf normalize(const MatrixXf& matrix) {
	return matrix.array().rowwise() / matrix.colwise().norm().eval().array();
}



float logDetPD(const MatrixXf& matrix) {
	return 2. * matrix.llt().matrixLLT().diagonal().array().log().sum();
}



MatrixXf deleteRows(const MatrixXf& matrix, vector<int> indices) {
	MatrixXf result = ArrayXXf::Zero(matrix.rows() - indices.size(), matrix.cols());

	sort(indices.begin(), indices.end());

	unsigned int idx = 0;

	for(int i = 0; i < matrix.rows(); ++i) {
		if(idx < indices.size() && indices[idx] == i) {
			++idx;
			continue;
		}
		result.row(i - idx) = matrix.row(i);
	}

	return result;
}



MatrixXf deleteCols(const MatrixXf& matrix, vector<int> indices) {
	MatrixXf result = ArrayXXf::Zero(matrix.rows(), matrix.cols() - indices.size());

	sort(indices.begin(), indices.end());

	unsigned int idx = 0;

	for(int i = 0; i < matrix.cols(); ++i) {
		if(idx < indices.size() && indices[idx] == i) {
			++idx;
			continue;
		}
		result.col(i - idx) = matrix.col(i);
	}

	return result;
}
