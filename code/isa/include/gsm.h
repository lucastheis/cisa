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

		inline ArrayXf scales();
		inline void setScales(MatrixXf scales);

		inline float variance();
		inline void normalize();

		virtual bool train(const MatrixXf& data, int maxIter = 100, double tol = 1e-5);

		virtual MatrixXf sample(int numSamples = 1);

		virtual Array<float, 1, Dynamic> samplePosterior(const MatrixXf& data);

		virtual ArrayXXf posterior(const MatrixXf& data);
		virtual ArrayXXf posterior(const MatrixXf& data, const RowVectorXf& sqNorms);

		virtual ArrayXXf logJoint(const MatrixXf& data);
		virtual ArrayXXf logJoint(const MatrixXf& data, const RowVectorXf& sqNorms);

		virtual Array<float, 1, Dynamic> logLikelihood(const MatrixXf& data);
		virtual Array<float, 1, Dynamic> logLikelihood(const MatrixXf& data, const RowVectorXf& sqNorms);

		virtual Array<float, 1, Dynamic> energy(const MatrixXf& data);
		virtual Array<float, 1, Dynamic> energy(const MatrixXf& data, const RowVectorXf& sqNorms);

		virtual ArrayXXf energyGradient(const MatrixXf& data);

	protected:
		int mDim;
		int mNumScales;
		ArrayXf mScales;
};



inline int GSM::dim() {
	return mDim;
}



inline int GSM::numScales() {
	return mNumScales;
}



inline ArrayXf GSM::scales() {
	return mScales;
}



inline float GSM::variance() {
	return mScales.square().mean();
}



inline void GSM::normalize() {
	mScales /= sqrt(variance());
}



inline void GSM::setScales(MatrixXf scales) {
	// turn row vector into column vector
	if(scales.cols() > scales.rows())
		scales.transposeInPlace();

	if(scales.rows() != mNumScales || scales.cols() != 1)
		throw Exception("Wrong number of scales.");

	mScales = scales;
}

#endif
