#ifndef ISA_H
#define ISA_H

#include "Eigen/Core"
#include "gsm.h"
#include <string>
#include <vector>

using namespace Eigen;
using std::string;
using std::vector;

class ISA;

class Callback {
	public:
		virtual ~Callback() { }
		virtual bool operator()(int iter, const ISA& isa) = 0;
};

struct Parameters {
	string trainingMethod;
	string samplingMethod;
	int maxIter;
	bool adaptive;
	Callback* callback;

	struct {
		int maxIter;
		int batchSize;
		double stepWidth;
		double momentum;
		bool shuffle;
		bool pocket;
	} SGD;

	struct {
		int maxIter;
		double tol;
	} GSM;

	Parameters() {
		// default parameters
		trainingMethod = "SGD";
		samplingMethod = "Gibbs";
		maxIter = 10;
		adaptive = true;
		callback = 0;

		SGD.maxIter = 1;
		SGD.batchSize = 100;
		SGD.stepWidth = 0.005;
		SGD.momentum = 0.8;
		SGD.shuffle = true;
		SGD.pocket = true;

		GSM.maxIter = 10;
		GSM.tol = 1e-8;
	}
};

class ISA {
	public:
		ISA(int numVisibles, int numHiddens = -1, int sSize = 1, int numScales = 10);
		virtual ~ISA();

		inline int numVisibles();
		inline int numHiddens();
		inline bool complete();
		inline int numSubspaces();

		inline MatrixXd basis();
		inline void setBasis(const MatrixXd& basis);

		virtual void initialize(const MatrixXd& data);

		virtual void train(const MatrixXd& data, Parameters params = Parameters());
		virtual void trainPrior(const MatrixXd& states, const Parameters params = Parameters());
		virtual bool trainSGD(
			const MatrixXd& complData,
			const MatrixXd& complBasis,
			const Parameters params = Parameters());

		virtual MatrixXd sample(int numSamples = 1);
		virtual MatrixXd samplePrior(int numSamples = 1);
		virtual MatrixXd samplePosterior(const MatrixXd& data);

		virtual MatrixXd priorEnergy(const MatrixXd& states);
		virtual MatrixXd priorEnergyGradient(const MatrixXd& states);

		virtual MatrixXd logLikelihood(const MatrixXd& data);

	protected:
		int mNumVisibles;
		int mNumHiddens;
		MatrixXd mBasis;
		vector<GSM> mSubspaces;
};



inline int ISA::numVisibles() {
	return mNumVisibles;
}



inline int ISA::numHiddens() {
	return mNumVisibles;
}



inline bool ISA::complete() {
	return mNumVisibles == mNumHiddens;
}



inline int ISA::numSubspaces() {
	return mSubspaces.size();
}



inline MatrixXd ISA::basis() {
	return mBasis;
}



inline void ISA::setBasis(const MatrixXd& basis) {
	mBasis = basis;
}

#endif
