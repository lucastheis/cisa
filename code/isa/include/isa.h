#ifndef ISA_H
#define ISA_H

#include "Eigen/Core"
#include "distribution.h"
#include "gsm.h"
#include <string>
#include <vector>
#include <iostream>

using namespace Eigen;
using std::string;
using std::vector;
using std::pair;

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
				bool trainBasis;
				bool learnGaussianity;
				bool mergeSubspaces;
				bool persistent;
				bool orthogonalize;
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
					int numGrad;
				} lbfgs;

				struct {
					Callback* callback;
					int maxIter;
					int batchSize;
					double stepWidth;
					double momentum;
					int numCoeff;
				} mp;

				struct {
					int maxIter;
					double tol;
				} gsm;

				struct {
					int verbosity;
					int iniIter;
					int numIter;
				} gibbs;

				struct {
					int verbosity;
					int numIter;
					int numSamples;
				} ais;

				struct {
					int verbosity;
					int maxMerge;
					int maxIter;
					double threshold;
				} merge;

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

		inline MatrixXd hiddenStates();
		inline void setHiddenStates(const MatrixXd& hiddenStates);

		inline double gaussianity();
		inline void setGaussianity(double gaussianity);

		virtual MatrixXd nullspaceBasis();

		virtual void initialize();
		virtual void initialize(const MatrixXd& data);

		virtual void orthogonalize();

		virtual void train(const MatrixXd& data, Parameters params = Parameters());
		virtual void trainPrior(
			const MatrixXd& states,
			const Parameters& params = Parameters());
		virtual bool trainSGD(
			const MatrixXd& complData,
			const MatrixXd& complBasis,
			const Parameters& params = Parameters());
		virtual bool trainLBFGS(
			const MatrixXd& complData,
			const MatrixXd& complBasis,
			const Parameters& params = Parameters());
		virtual void trainMP(
			const MatrixXd& data,
			const Parameters& params = Parameters());
		virtual MatrixXd mergeSubspaces(MatrixXd states, const Parameters& params = Parameters());

		virtual MatrixXd sample(int numSamples = 1);
		virtual MatrixXd samplePrior(int numSamples = 1);
		virtual MatrixXd sampleScales(const MatrixXd& states);
		virtual MatrixXd samplePosterior(
			const MatrixXd& data,
			const MatrixXd& states,
			const Parameters& params = Parameters());
		virtual pair<MatrixXd, MatrixXd> samplePosteriorAIS(
			const MatrixXd& data,
			const Parameters& params = Parameters());
		virtual MatrixXd samplePosterior(const MatrixXd& data, const Parameters& params = Parameters());
		virtual MatrixXd sampleNullspace(const MatrixXd& data, const Parameters& params = Parameters());
		virtual MatrixXd sampleAIS(const MatrixXd& data, const Parameters& params = Parameters());

		virtual MatrixXd matchingPursuit(const MatrixXd& data, const Parameters& params = Parameters());

		virtual MatrixXd priorLogLikelihood(const MatrixXd& states);
		virtual MatrixXd priorEnergy(const MatrixXd& states);
		virtual MatrixXd priorEnergyGradient(const MatrixXd& states);

		virtual Array<double, 1, Dynamic> logLikelihood(const MatrixXd& data);
		virtual Array<double, 1, Dynamic> logLikelihood(const MatrixXd& data, const Parameters& params);
		virtual Array<double, 1, Dynamic> logLikelihoodISA(const MatrixXd& data, const Parameters& params);
		virtual double evaluate(const MatrixXd& data, const Parameters& params = Parameters());

		virtual Array<double, 1, Dynamic> posteriorWeights(
			const MatrixXd& data,
			const Parameters& params = Parameters());

	protected:
		int mNumVisibles;
		int mNumHiddens;
		MatrixXd mBasis;
		vector<GSM> mSubspaces;
		MatrixXd mHiddenStates;
		double mGaussianity;
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
	if(basis.rows() != numVisibles() && basis.cols() != numHiddens())
		throw Exception("Basis has wrong dimensionality.");

	mBasis = basis;
}



inline MatrixXd ISA::hiddenStates() {
	return mHiddenStates;
}



inline void ISA::setHiddenStates(const MatrixXd& hiddenStates) {
	mHiddenStates = hiddenStates;
}



inline double ISA::gaussianity() {
	return mGaussianity;
}



inline void ISA::setGaussianity(double gaussianity) {
	if(gaussianity < 0. || gaussianity > 1.)
		throw Exception("Gaussianity has to be between 0 and 1."); 

	mGaussianity = gaussianity;
}

#endif
