#include "isa.h"
#include "Eigen/LU"
#include <iostream>

ISA::ISA(int numVisibles, int numHiddens, int sSize, int numScales) :
	mNumVisibles(numVisibles), mNumHiddens(numHiddens) 
{
	if(mNumHiddens < 0)
		mNumHiddens = mNumVisibles;
	mBasis = MatrixXd::Random(mNumVisibles, mNumHiddens);
}



ISA::~ISA() {
}



void ISA::train(const MatrixXd& data, Parameters params) {
	trainSGD(data, basis(), params);
}



MatrixXd ISA::trainSGD(
	const MatrixXd& data,
	const MatrixXd& basis,
	Parameters params)
{
	MatrixXd W = basis.inverse();
	MatrixXd P = MatrixXd::Zero(basis.rows(), basis.cols());

	for(int i = 0; i < params.SGD.maxIter; ++i) {
		for(int j = 0; j + params.SGD.batchSize <= data.cols(); j += params.SGD.batchSize) {
			MatrixXd X = data.middleCols(j, params.SGD.batchSize);

			// update momentum with natural gradient
			P = params.SGD.momentum * P + W
				- priorEnergyGradient(W * X) * X.transpose() * (W.transpose() * W);

			// update filter matrix
			W += params.SGD.stepWidth / params.SGD.batchSize * P;
		}
	}

	return W.inverse();
}



MatrixXd ISA::sample(int num_samples) {
	return MatrixXd::Random(1, 1);
}



MatrixXd ISA::samplePosterior(const MatrixXd& data) {
	return MatrixXd::Random(1, 1);
}



MatrixXd ISA::priorEnergyGradient(const MatrixXd& states) {
	return MatrixXd::Zero(states.rows(), states.cols());
}
