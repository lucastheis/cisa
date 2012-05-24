#include "gsm.h"
#include <iostream>

GSM::GSM(int dim, int numScales) : mDim(dim), mNumScales(numScales) {
	mScales = 1. + ArrayXd::Random(mNumScales) / 4.;
	mScales /= mScales.array().mean();
}


void GSM::train(const MatrixXd& data, int max_iter, double tol) {
	if(data.rows() != mDim)
		throw Exception("Data has wrong dimensionality.");

	MatrixXd sqNorms = data.array().square().colwise().sum();

	for(int i = 0; i < max_iter; ++i) {
		MatrixXd posterior = -0.5 * sqNorms * mScales.transpose()
			- mDim * mScales.array().log().matrix();
//		posterior = (post.colwise() - 
	}
}
