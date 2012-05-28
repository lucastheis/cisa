#include "gsm.h"
#include <iostream>
#include <cmath>
#include <cstdlib>

using std::log;
using std::rand;

#define PI 3.141592653589793

Array<double, 1, Dynamic> logsumexp(const ArrayXXd& array) {
	Array<double, 1, Dynamic> array_max = array.colwise().maxCoeff();
	return array_max + (array.rowwise() - array_max).exp().colwise().sum().log();
}



Array<double, 1, Dynamic> logmeanexp(const ArrayXXd& array) {
	Array<double, 1, Dynamic> array_max = array.colwise().maxCoeff();
	return array_max + (array.rowwise() - array_max).exp().colwise().mean().log();
}



ArrayXXd sampleNormal(int m = 1, int n = 1) {
	ArrayXXd U1 = ArrayXXd::Random(m, n) / 2. + 0.5;
	ArrayXXd U2 = ArrayXXd::Random(m, n) / 2. + 0.5;
	return (-2. * U1.log()).sqrt() * (2. * PI * U2).cos();
}



GSM::GSM(int dim, int numScales) : mDim(dim), mNumScales(numScales) {
	mScales = 1. + ArrayXd::Random(mNumScales) / 4.;
	mScales /= mScales.mean();
}



void GSM::train(const MatrixXd& data, int max_iter, double tol) {
	if(data.rows() != mDim)
		throw Exception("Data has wrong dimensionality.");

	RowVectorXd sqNorms = data.colwise().squaredNorm();

	for(int i = 0; i < max_iter; ++i) {
		// compute unnormalized posterior over mixture components (E)
		ArrayXXd post = posterior(data, sqNorms);

		// update parameters (M)
		mScales = ((post.rowwise() * sqNorms.array()).rowwise().mean()
			/ (mDim * post.rowwise().mean())).sqrt();
	}
}



MatrixXd GSM::sample(int num_samples) {
	Array<double, 1, Dynamic> scales(1, num_samples);

	// pick random standard deviations
	for(int i = 0; i < num_samples; ++i)
		scales[i] = mScales[rand() % mNumScales];

	// scale normal samples
	return sampleNormal(mDim, num_samples).rowwise() * scales;
}



Array<double, 1, Dynamic> GSM::samplePosterior(const MatrixXd& data) {
	Array<double, 1, Dynamic> scales(data.cols());
	ArrayXXd post = posterior(data);

	for(int j = 0; j < post.cols(); ++j) {
		int i = 0;
		double urand = static_cast<double>(rand()) / (static_cast<long>(RAND_MAX) + 1l);
		double cdf;

		// compute index
		for(cdf = post(0, j); cdf < urand; cdf += post(i, j))
			++i;

		scales[j] = mScales[i];
	}

	return scales;
}



ArrayXXd GSM::posterior(const MatrixXd& data) {
	return posterior(data, data.colwise().squaredNorm());
}



ArrayXXd GSM::posterior(const MatrixXd& data, const RowVectorXd& sqNorms) {
	// compute unnormalized log-posterior
	ArrayXXd posterior = logJoint(data, sqNorms);

	// normalize posterior in a numerically stable way
	posterior.rowwise() -= posterior.colwise().maxCoeff().eval();
	posterior = posterior.exp();
	posterior.rowwise() /= posterior.colwise().sum().eval();

	return posterior;
}



ArrayXXd GSM::logJoint(const MatrixXd& data) {
	return logJoint(data, data.colwise().squaredNorm());
}



ArrayXXd GSM::logJoint(const MatrixXd& data, const RowVectorXd& sqNorms) {
	return (-0.5 * mScales.square().inverse().matrix() * sqNorms).colwise()
		- mDim * mScales.log().matrix();
}



Array<double, 1, Dynamic> GSM::logLikelihood(const MatrixXd& data) {
	return -energy(data).array() - mDim / 2. * log(2. * PI);
}



Array<double, 1, Dynamic> GSM::logLikelihood(const MatrixXd& data, const RowVectorXd& sqNorms) {
	return -energy(data, sqNorms).array() - mDim / 2. * log(2. * PI);
}



Array<double, 1, Dynamic> GSM::energy(const MatrixXd& data) {
	return -logmeanexp(logJoint(data));
}



Array<double, 1, Dynamic> GSM::energy(const MatrixXd& data, const RowVectorXd& sqNorms) {
	return -logmeanexp(logJoint(data, sqNorms));
}



ArrayXXd GSM::energyGradient(const MatrixXd& data) {
	return data.array().rowwise() * (posterior(data).colwise() * mScales.square().inverse()).colwise().sum();
}
