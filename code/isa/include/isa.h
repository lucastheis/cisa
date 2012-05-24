#ifndef ISA_H
#define ISA_H

#include "Eigen/Core"
#include <string>

using namespace Eigen;
using std::string;

struct Parameters {
	string trainingMethod;
	string samplingMethod;
	bool adaptive;

	struct {
		int maxIter;
		int batchSize;
		double stepWidth;
		double momentum;
		bool shuffle;
		bool pocket;
	} SGD;

	Parameters() {
		// default parameters
		trainingMethod = "SGD";
		samplingMethod = "Gibbs";
		adaptive = true;

		SGD.maxIter = 1;
		SGD.batchSize = 100;
		SGD.stepWidth = 0.001;
		SGD.momentum = 0.8;
		SGD.shuffle = true;
		SGD.pocket = true;
	}
};

class ISA {
	public:
		ISA(int numVisibles, int numHiddens = -1, int sSize = 1, int numScales = 10);
		virtual ~ISA();

		inline int numVisibles();
		inline int numHiddens();

		inline MatrixXd basis();
		inline void setBasis(const MatrixXd& basis);

		virtual void train(const MatrixXd& data, Parameters params = Parameters());
		virtual MatrixXd trainSGD(
			const MatrixXd& data,
			const MatrixXd& basis,
			Parameters params = Parameters());

		virtual MatrixXd sample(int num_samples = 1);
		virtual MatrixXd samplePosterior(const MatrixXd& data);

		virtual MatrixXd priorEnergyGradient(const MatrixXd& states);

	protected:
		int mNumVisibles;
		int mNumHiddens;
		MatrixXd mBasis;
};



inline int ISA::numVisibles() {
	return mNumVisibles;
}



inline int ISA::numHiddens() {
	return mNumVisibles;
}



inline MatrixXd ISA::basis() {
	return mBasis;
}



inline void ISA::setBasis(const MatrixXd& basis) {
	mBasis = basis;
}

#endif
