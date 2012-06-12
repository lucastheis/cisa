#ifndef ISA_H
#define ISA_H

#include "Eigen/Core"
#include "distribution.h"
#include "gsm.h"
#include <string>
#include <vector>

using namespace Eigen;
using std::string;
using std::vector;

class ISA : public Distribution {
	public:
		class Callback {
			public:
				virtual ~Callback();
				virtual Callback* copy() = 0;
				virtual bool operator()(int iter, const ISA& isa) = 0;
		};

		struct Parameters {
			public:
				int verbosity;
				string trainingMethod;
				string samplingMethod;
				int maxIter;
				bool adaptive;
				bool trainPrior;
				bool persistent;
				Callback* callback;

				struct {
					int maxIter;
					int batchSize;
					double stepWidth;
					double momentum;
					bool shuffle;
					bool pocket;
				} sgd;

				struct {
					int maxIter;
					double tol;
				} gsm;

				struct {
					int iniIter;
					int numIter;
				} gibbs;

				Parameters();
				Parameters(const Parameters& params);
				virtual ~Parameters();
				virtual Parameters& operator=(const Parameters& params);
		};

		ISA(int numVisibles, int numHiddens = -1, int sSize = 1, int numScales = 10);
		virtual ~ISA();

		inline int dim();
		inline int numVisibles();
		inline int numHiddens();
		inline bool complete();
		inline int numSubspaces();

		inline vector<GSM> subspaces();
		inline void setSubspaces(vector<GSM> subspaces);

		inline MatrixXd basis();
		inline void setBasis(const MatrixXd& basis);

		virtual MatrixXd nullspaceBasis();

		virtual void initialize();
		virtual void initialize(const MatrixXd& data);

		virtual void train(const MatrixXd& data, Parameters params = Parameters());
		virtual void trainPrior(
			const MatrixXd& states,
			const Parameters params = Parameters());
		virtual bool trainSGD(
			const MatrixXd& complData,
			const MatrixXd& complBasis,
			const Parameters params = Parameters());

		virtual MatrixXd sample(int numSamples = 1);
		virtual MatrixXd samplePrior(int numSamples = 1);
		virtual MatrixXd sampleScales(const MatrixXd& states);
		virtual MatrixXd samplePosterior(const MatrixXd& data, const Parameters params = Parameters());
		virtual MatrixXd sampleNullspace(const MatrixXd& data, const Parameters params = Parameters());

		virtual MatrixXd priorEnergy(const MatrixXd& states);
		virtual MatrixXd priorEnergyGradient(const MatrixXd& states);

		virtual Array<double, 1, Dynamic> logLikelihood(const MatrixXd& data);

	protected:
		int mNumVisibles;
		int mNumHiddens;
		MatrixXd mBasis;
		vector<GSM> mSubspaces;
};



inline int ISA::dim() {
	return mNumVisibles;
}



inline int ISA::numVisibles() {
	return mNumVisibles;
}



inline int ISA::numHiddens() {
	return mNumHiddens;
}



inline bool ISA::complete() {
	return mNumVisibles == mNumHiddens;
}



inline int ISA::numSubspaces() {
	return mSubspaces.size();
}



inline vector<GSM> ISA::subspaces() {
	return mSubspaces;
}


inline void ISA::setSubspaces(vector<GSM> subspaces) {
	int dim = 0;
	for(size_t i = 0; i < subspaces.size(); ++i)
		dim += subspaces[i].dim();

	if(dim != numHiddens())
		throw Exception("Subspace dimensionality should correspond to the number of hidden units.");

	mSubspaces = subspaces;
}



inline MatrixXd ISA::basis() {
	return mBasis;
}



inline void ISA::setBasis(const MatrixXd& basis) {
	mBasis = basis;
}

#endif
