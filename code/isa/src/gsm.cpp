#include "gsm.h"
#include "utils.h"
#include <iostream>
#include <cmath>
#include <cstdlib>

using std::log;
using std::rand;

GSM::GSM(int dim, int numScales) : mDim(dim), mNumScales(numScales) {
	mPriors = ArrayXd::Ones(mNumScales) / mNumScales;
	mScales = 1. + ArrayXd::Random(mNumScales) / 4.;
	mScales /= mScales.mean();
}



bool GSM::train(const MatrixXd& data, int maxIter, double tol) {
	if(data.rows() != mDim)
		throw Exception("Data has wrong dimensionality.");

	RowVectorXd sqNorms = data.colwise().squaredNorm();

	double logLik = logLikelihood(data, sqNorms).mean();

	for(int i = 0; i < maxIter; ++i) {
		// compute unnormalized posterior over mixture components (E)
		ArrayXXd post = posterior(data, sqNorms);

		// update parameters (M)
		mPriors = post.rowwise().mean() + 1e-6;
		mPriors /= mPriors.sum();
		mScales = (((post.rowwise() * sqNorms.array()).rowwise().mean() + 1e-9)
			/ (mDim * post.rowwise().mean() + 3e-9)).sqrt();

		if(tol > 0. && i % 5 == 0) {
			double logLikNew = logLikelihood(data, sqNorms).mean();

			// check for convergence
			if(logLikNew - logLik < tol)
				return true;

			logLik = logLikNew;
		}
	}

	return false;
}



MatrixXd GSM::sample(int numSamples) {
	Array<double, 1, Dynamic> scales(1, numSamples);

	#pragma omp parallel for
	for(int j = 0; j < numSamples; ++j) {
		int i = 0;
		double urand = static_cast<double>(rand()) / (static_cast<long>(RAND_MAX) + 1l);

		// compute index
		for(double cdf = mPriors[0]; cdf < urand; cdf += mPriors[i])
			++i;

		scales[j] = mScales[i];
	}

	// scale normal samples
	return sampleNormal(mDim, numSamples).rowwise() * scales;
}



Array<double, 1, Dynamic> GSM::samplePosterior(const MatrixXd& data) {
	Array<double, 1, Dynamic> scales(data.cols());
	ArrayXXd post = posterior(data);

	#pragma omp parallel for
	for(int j = 0; j < post.cols(); ++j) {
		int i = 0;
		double urand = static_cast<double>(rand()) / (static_cast<long>(RAND_MAX) + 1l);

		// compute index
		for(double cdf = post(0, j); cdf < urand; cdf += post(i, j))
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
		+ (mPriors.log() - mDim * mScales.log()).matrix();
}



Array<double, 1, Dynamic> GSM::logLikelihood(const MatrixXd& data) {
	return -energy(data).array() - mDim / 2. * log(2. * PI);
}



Array<double, 1, Dynamic> GSM::logLikelihood(const MatrixXd& data, const RowVectorXd& sqNorms) {
	return -energy(data, sqNorms).array() - mDim / 2. * log(2. * PI);
}



Array<double, 1, Dynamic> GSM::energy(const MatrixXd& data) {
	return -logsumexp(logJoint(data));
}



Array<double, 1, Dynamic> GSM::energy(const MatrixXd& data, const RowVectorXd& sqNorms) {
	return -logsumexp(logJoint(data, sqNorms));
}



ArrayXXd GSM::energyGradient(const MatrixXd& data) {
	return data.array().rowwise() * (posterior(data).colwise() * mScales.square().inverse()).colwise().sum();
}
