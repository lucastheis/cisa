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
	for(int i = 0; i < params.maxIter; ++i) {
		bool improved = trainSGD(data, basis(), params);

		if(params.adaptive)
			params.SGD.stepWidth *= improved ? 1.1 : 0.5;
	}
}



bool ISA::trainSGD(
	const MatrixXd& complData,
	const MatrixXd& complBasis,
	Parameters params)
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



MatrixXd ISA::priorEnergyGradient(const MatrixXd& states) {
	MatrixXd gradient = MatrixXd::Zero(states.rows(), states.cols());

	// TODO: parallelize
	for(int from = 0, i = 0; i < numSubspaces(); from += mSubspaces[i].dim(), ++i)
		gradient.middleRows(from, mSubspaces[i].dim()) =
			mSubspaces[i].energyGradient(states.middleRows(from, mSubspaces[i].dim()));

	return gradient;
}
