#include "isa.h"
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



void ISA::train(MatrixXd data, Parameters params) {
  	trainSGD(data, basis(), params);
}



MatrixXd ISA::trainSGD(
	MatrixXd data,
	MatrixXd basis,
	Parameters params)
{
 	MatrixXd dBasis = basis;
 
 	for(int i = 0; i < params.SGD.maxIter; ++i) {
  		for(int j = 0; j + params.SGD.batchSize <= data.cols(); j += params.SGD.batchSize) {
  			MatrixXd batch = data.middleCols(j, params.SGD.batchSize);
  		}
 	}

	return this->basis();
}



MatrixXd ISA::sample(int num_samples) {
	return MatrixXd::Random(1, 1);
}



MatrixXd ISA::samplePosterior(const MatrixXd& data) {
	return MatrixXd::Random(1, 1);
}
