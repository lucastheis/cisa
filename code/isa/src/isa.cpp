#include "isa.h"
#include "Eigen/LU"
#include <iostream>

ISA::ISA(int numVisibles, int numHiddens, int sSize, int numScales) :
	mNumVisibles(numVisibles), mNumHiddens(numHiddens)
{
	if(mNumHiddens < mNumVisibles)
		mNumHiddens = mNumVisibles;
	mBasis = MatrixXd::Random(mNumVisibles, mNumHiddens);

	for(int i = 0; i < mNumHiddens / sSize; ++i)
		mSubspaces.push_back(GSM(sSize, numScales));

	if(mNumHiddens % sSize)
		mSubspaces.push_back(GSM(mNumHiddens % sSize, numScales));
}



ISA::~ISA() {
}



void ISA::train(const MatrixXd& data, Parameters params) {
	if(params.callback)
		// call callback function once before training
		if(!(*params.callback)(0, *this))
			return;

	for(int i = 0; i < params.maxIter; ++i) {
		// optimize basis
		bool improved = trainSGD(data, basis(), params);

		if(params.adaptive)
			params.SGD.stepWidth *= improved ? 1.1 : 0.5;

		// optimize marginal distributions
		trainPrior(basis().inverse() * data, params);

		if(params.callback)
			if(!(*params.callback)(i + 1, *this))
				break;
	}
}



void ISA::trainPrior(const MatrixXd& states, const Parameters params) {
	// TODO: parallelize
	for(int from = 0, i = 0; i < numSubspaces(); from += mSubspaces[i].dim(), ++i)
		mSubspaces[i].train(states.middleRows(from, mSubspaces[i].dim()),
			params.GSM.maxIter,
			params.GSM.tol);
}



bool ISA::trainSGD(
	const MatrixXd& complData,
	const MatrixXd& complBasis,
	const Parameters params)
{
	// filter matrix and momentum
	MatrixXd W = complBasis.inverse();
	MatrixXd P = MatrixXd::Zero(complBasis.rows(), complBasis.cols());

	for(int i = 0; i < params.SGD.maxIter; ++i) {
		for(int j = 0; j + params.SGD.batchSize <= complData.cols(); j += params.SGD.batchSize) {
			MatrixXd X = complData.middleCols(j, params.SGD.batchSize);

			// update momentum with natural gradient
			P = params.SGD.momentum * P + W
				- priorEnergyGradient(W * X) * X.transpose() * (W.transpose() * W);

			// update filter matrix
			W += params.SGD.stepWidth / params.SGD.batchSize * P;
		}
	}

	// update basis
	setBasis(W.inverse().leftCols(numHiddens()));

	return true;
}



MatrixXd ISA::sample(int numSamples) {
	return basis() * samplePrior(numSamples);
}



MatrixXd ISA::samplePrior(int numSamples) {
	MatrixXd samples = MatrixXd::Zero(numHiddens(), numSamples);

	for(int from = 0, i = 0; i < numSubspaces(); from += mSubspaces[i].dim(), ++i)
		samples.middleRows(from, mSubspaces[i].dim()) =
			mSubspaces[i].sample(numSamples);

	return samples;
}



MatrixXd ISA::samplePosterior(const MatrixXd& data) {
	// TODO: implement Gibbs sampling
	return samplePrior(data.cols());
}



MatrixXd ISA::priorEnergy(const MatrixXd& states) {
	MatrixXd energy = MatrixXd::Zero(states.rows(), states.cols());

	// TODO: parallelize
	for(int from = 0, i = 0; i < numSubspaces(); from += mSubspaces[i].dim(), ++i)
		energy.middleRows(from, mSubspaces[i].dim()) =
			mSubspaces[i].energy(states.middleRows(from, mSubspaces[i].dim()));

	return energy.colwise().sum();
}



MatrixXd ISA::priorEnergyGradient(const MatrixXd& states) {
	MatrixXd gradient = MatrixXd::Zero(states.rows(), states.cols());

	// TODO: parallelize
	for(int from = 0, i = 0; i < numSubspaces(); from += mSubspaces[i].dim(), ++i)
		gradient.middleRows(from, mSubspaces[i].dim()) =
			mSubspaces[i].energyGradient(states.middleRows(from, mSubspaces[i].dim()));

	return gradient;
}



MatrixXd ISA::logLikelihood(const MatrixXd& data) {
	// LU decomposition
	PartialPivLU<MatrixXd> basisLU(mBasis);

	// compute log-determinant of basis
	double logDet = basisLU.matrixLU().diagonal().array().abs().log().sum();

	MatrixXd states = basisLU.solve(data);
	MatrixXd logLik = MatrixXd::Zero(states.rows(), states.cols());

	// TODO: parallelize
	for(int from = 0, i = 0; i < numSubspaces(); from += mSubspaces[i].dim(), ++i)
		logLik.middleRows(from, mSubspaces[i].dim()) =
			mSubspaces[i].logLikelihood(states.middleRows(from, mSubspaces[i].dim()));

	return logLik.colwise().sum().array() - logDet;
}
