#include "gsm.h"
#include "utils.h"
#include <iostream>
#include <cmath>
#include <cstdlib>

using std::log;
using std::rand;

GSM::GSM(int dim, int numScales) : mDim(dim), mNumScales(numScales) {
	mScales = 1. + ArrayXf::Random(mNumScales) / 4.;
	mScales /= mScales.mean();
}



bool GSM::train(const MatrixXf& data, int maxIter, double tol) {
	if(data.rows() != mDim)
		throw Exception("Data has wrong dimensionality.");

	RowVectorXf sqNorms = data.colwise().squaredNorm();

	float logLik = logLikelihood(data, sqNorms).mean();

	for(int i = 0; i < maxIter; ++i) {
		// compute unnormalized posterior over mixture components (E)
		ArrayXXf post = posterior(data, sqNorms);

		// update parameters (M)
		mScales = (((post.rowwise() * sqNorms.array()).rowwise().mean() + 1e-9)
			/ (mDim * post.rowwise().mean() + 3e-9)).sqrt();

		if(tol > 0. && i % 5 == 0) {
			float logLikNew = logLikelihood(data, sqNorms).mean();

			// check for convergence
			if(logLikNew - logLik < tol)
				return true;

			logLik = logLikNew;
		}
	}

	return false;
}



MatrixXf GSM::sample(int numSamples) {
	Array<float, 1, Dynamic> scales(1, numSamples);

	// pick random standard deviations
	for(int i = 0; i < numSamples; ++i)
		scales[i] = mScales[rand() % mNumScales];

	// scale normal samples
	return sampleNormal(mDim, numSamples).rowwise() * scales;
}



Array<float, 1, Dynamic> GSM::samplePosterior(const MatrixXf& data) {
	Array<float, 1, Dynamic> scales(data.cols());
	ArrayXXf post = posterior(data);

	for(int j = 0; j < post.cols(); ++j) {
		int i = 0;
		float urand = static_cast<float>(rand()) / (static_cast<long>(RAND_MAX) + 1l);
		float cdf;

		// compute index
		for(cdf = post(0, j); cdf < urand; cdf += post(i, j))
			++i;

		scales[j] = mScales[i];
	}

	return scales;
}



ArrayXXf GSM::posterior(const MatrixXf& data) {
	return posterior(data, data.colwise().squaredNorm());
}



ArrayXXf GSM::posterior(const MatrixXf& data, const RowVectorXf& sqNorms) {
	// compute unnormalized log-posterior
	ArrayXXf posterior = logJoint(data, sqNorms);

	// normalize posterior in a numerically stable way
	posterior.rowwise() -= posterior.colwise().maxCoeff().eval();
	posterior = posterior.exp();
	posterior.rowwise() /= posterior.colwise().sum().eval();

	return posterior;
}



ArrayXXf GSM::logJoint(const MatrixXf& data) {
	return logJoint(data, data.colwise().squaredNorm());
}



ArrayXXf GSM::logJoint(const MatrixXf& data, const RowVectorXf& sqNorms) {
	return (-0.5 * mScales.square().inverse().matrix() * sqNorms).colwise()
		- mDim * mScales.log().matrix();
}



Array<float, 1, Dynamic> GSM::logLikelihood(const MatrixXf& data) {
	return -energy(data).array() - mDim / 2. * log(2. * PI);
}



Array<float, 1, Dynamic> GSM::logLikelihood(const MatrixXf& data, const RowVectorXf& sqNorms) {
	return -energy(data, sqNorms).array() - mDim / 2. * log(2. * PI);
}



Array<float, 1, Dynamic> GSM::energy(const MatrixXf& data) {
	return -logmeanexp(logJoint(data));
}



Array<float, 1, Dynamic> GSM::energy(const MatrixXf& data, const RowVectorXf& sqNorms) {
	return -logmeanexp(logJoint(data, sqNorms));
}



ArrayXXf GSM::energyGradient(const MatrixXf& data) {
	return data.array().rowwise() * (posterior(data).colwise() * mScales.square().inverse()).colwise().sum();
}
