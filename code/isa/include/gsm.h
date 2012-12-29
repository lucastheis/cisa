#ifndef GSM_H
#define GSM_H

#include "Eigen/Core"
#include "distribution.h"
#include "exception.h"
#include <iostream>
#include <cmath>

using namespace Eigen;
using std::sqrt;

class GSM : public Distribution {
	public:
		GSM(int dim = 1, int numScales = 10);

		inline int dim();
		inline int numScales();

		inline ArrayXd priors() const;
		inline void setPriors(MatrixXd priors);

		inline ArrayXd scales() const;
		inline void setScales(MatrixXd scales);

		inline double variance();
		inline void normalize();

		virtual bool train(const MatrixXd& data, int maxIter = 100, double tol = 1e-5);

		virtual MatrixXd sample(int numSamples = 1);

		virtual Array<double, 1, Dynamic> samplePosterior(const MatrixXd& data);

		virtual ArrayXXd posterior(const MatrixXd& data);
		virtual ArrayXXd posterior(const MatrixXd& data, const RowVectorXd& sqNorms);

		virtual ArrayXXd logJoint(const MatrixXd& data);
		virtual ArrayXXd logJoint(const MatrixXd& data, const RowVectorXd& sqNorms);

		virtual Array<double, 1, Dynamic> logLikelihood(const MatrixXd& data);
		virtual Array<double, 1, Dynamic> logLikelihood(const MatrixXd& data, const RowVectorXd& sqNorms);

		virtual Array<double, 1, Dynamic> energy(const MatrixXd& data);
		virtual Array<double, 1, Dynamic> energy(const MatrixXd& data, const RowVectorXd& sqNorms);

		virtual ArrayXXd energyGradient(const MatrixXd& data);

	protected:
		int mDim;
		int mNumScales;
		ArrayXd mPriors;
		ArrayXd mScales;
};



inline int GSM::dim() {
	return mDim;
}



inline int GSM::numScales() {
	return mNumScales;
}



inline ArrayXd GSM::priors() const {
	return mPriors;
}



inline void GSM::setPriors(MatrixXd priors) {
	// turn row vector into column vector
	if(priors.cols() > priors.rows())
		priors.transposeInPlace();

	if(priors.rows() != mNumScales || priors.cols() != 1)
		throw Exception("Wrong number of prior weights.");

	mPriors = priors / priors.sum();
}



inline ArrayXd GSM::scales() const {
	return mScales;
}



inline double GSM::variance() {
	return mScales.square().mean();
}



inline void GSM::normalize() {
	mScales /= sqrt(variance());
}



inline void GSM::setScales(MatrixXd scales) {
	// turn row vector into column vector
	if(scales.cols() > scales.rows())
		scales.transposeInPlace();

	if(scales.rows() != mNumScales || scales.cols() != 1)
		throw Exception("Wrong number of scales.");

	mScales = scales;
}

#endif
