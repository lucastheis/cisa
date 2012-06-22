#include "Eigen/Cholesky"
#include "utils.h"
#include <algorithm>
#include <vector>
#include <iostream>

using namespace std;

Array<double, 1, Dynamic> logsumexp(const ArrayXXd& array) {
	Array<double, 1, Dynamic> array_max = array.colwise().maxCoeff();
	return array_max + (array.rowwise() - array_max).exp().colwise().sum().log();
}



Array<double, 1, Dynamic> logmeanexp(const ArrayXXd& array) {
	Array<double, 1, Dynamic> array_max = array.colwise().maxCoeff() - 1.;
	return array_max + (array.rowwise() - array_max).exp().colwise().mean().log();
}



ArrayXXd sampleNormal(int m, int n) {
	ArrayXXd U1 = ArrayXXd::Random(m, n) / 2. + 0.5;
	ArrayXXd U2 = ArrayXXd::Random(m, n) / 2. + 0.5;
	// Box-Muller transform
	return (-2. * U1.log()).sqrt() * (2. * PI * U2).cos();
}



VectorXi argsort(const VectorXd& data) {
	// create pairs of values and indices
	vector<pair<double, int> > pairs(data.size());
	for(int i = 0; i < data.size(); ++i) {
		pairs[i].first = data[i];
		pairs[i].second = i;
	}

	// sort values in descending order
	sort(pairs.begin(), pairs.end(), greater<pair<double, int> >());

	// store indices
	VectorXi indices(data.size());
	for(int i = 0; i < data.size(); ++i)
		indices[pairs[i].second] = i;

	return indices;
}



MatrixXd covariance(const MatrixXd& data) {
	MatrixXd data_centered = data.colwise() - data.rowwise().mean().eval();
	return data_centered * data_centered.transpose() / data.cols();
}



MatrixXd corrcoef(const MatrixXd& data) {
	MatrixXd C = covariance(data);
	VectorXd c = C.diagonal();
	return C.array() / (c * c.transpose()).array().sqrt();
}



MatrixXd normalize(const MatrixXd& matrix) {
	return matrix.array().rowwise() / matrix.colwise().norm().eval().array();
}



double logDetPD(const MatrixXd& matrix) {
	return 2. * matrix.llt().matrixLLT().diagonal().array().log().sum();
}



MatrixXd deleteRows(const MatrixXd& matrix, vector<int> indices) {
	MatrixXd result = ArrayXXd::Zero(matrix.rows() - indices.size(), matrix.cols()) - 8.;

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



MatrixXd deleteCols(const MatrixXd& matrix, vector<int> indices) {
	MatrixXd result = ArrayXXd::Zero(matrix.rows(), matrix.cols() - indices.size()) - 9.;

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
