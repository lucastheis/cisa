#ifndef GSM_H
#define GSM_H

#include "Eigen/Core"
#include "exception.h"
#include <iostream>

using namespace Eigen;

class GSM {
	public:
		GSM(int dim = 1, int numScales = 10);

		inline int dim();
		inline int numScales();

		inline VectorXd scales();
		inline void setScales(MatrixXd scales);

		void train(const MatrixXd& data, int max_iter = 100, double tol = 1e-5);

//		MatrixXd sample(int num_samples = 1);
//
//		MatrixXd loglikelihood(MatrixXd& data);
//		MatrixXd energy(const MatrixXd& data);
//		MatrixXd energyGradient(const MatrixXd& data);

	protected:
		int mDim;
		int mNumScales;
		VectorXd mScales;
};



inline int GSM::dim() {
	return mDim;
}



inline int GSM::numScales() {
	return mNumScales;
}



inline VectorXd GSM::scales() {
	return mScales;
}



inline void GSM::setScales(MatrixXd scales) {
	// turn row vectors into column vectors
	if(scales.cols() > scales.rows())
		scales.transposeInPlace();

	if(scales.rows() != mNumScales || scales.cols() != 1)
		throw Exception("Wrong number of scales.");

	mScales = scales;
}

#endif
