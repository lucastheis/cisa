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

		inline MatrixXf basis();
		inline void setBasis(const MatrixXf& basis);

		inline MatrixXf hiddenStates();
		inline void setHiddenStates(const MatrixXf& hiddenStates);

		virtual MatrixXf nullspaceBasis();

		virtual void initialize();
		virtual void initialize(const MatrixXf& data);

		virtual void orthogonalize();

		virtual void train(const MatrixXf& data, Parameters params = Parameters());
		virtual void trainPrior(
			const MatrixXf& states,
			const Parameters& params = Parameters());
		virtual bool trainSGD(
			const MatrixXf& complData,
			const MatrixXf& complBasis,
			const Parameters& params = Parameters());
		virtual bool trainLBFGS(
			const MatrixXf& complData,
			const MatrixXf& complBasis,
			const Parameters& params = Parameters());
		virtual void trainMP(
			const MatrixXf& data,
			const Parameters& params = Parameters());
		virtual MatrixXf mergeSubspaces(MatrixXf states, const Parameters& params = Parameters());

		virtual MatrixXf sample(int numSamples = 1);
		virtual MatrixXf samplePrior(int numSamples = 1);
		virtual MatrixXf sampleScales(const MatrixXf& states);
		virtual MatrixXf samplePosterior(
			const MatrixXf& data,
			const MatrixXf& states,
			const Parameters& params = Parameters());
		virtual pair<MatrixXf, MatrixXf> samplePosteriorAIS(
			const MatrixXf& data,
			const Parameters& params = Parameters());
		virtual MatrixXf samplePosterior(const MatrixXf& data, const Parameters& params = Parameters());
		virtual MatrixXf sampleNullspace(const MatrixXf& data, const Parameters& params = Parameters());
		virtual MatrixXf sampleAIS(const MatrixXf& data, const Parameters& params = Parameters());

		virtual MatrixXf matchingPursuit(const MatrixXf& data, const Parameters& params = Parameters());

		virtual MatrixXf priorLogLikelihood(const MatrixXf& states);
		virtual MatrixXf priorEnergy(const MatrixXf& states);
		virtual MatrixXf priorEnergyGradient(const MatrixXf& states);

		virtual Array<float, 1, Dynamic> logLikelihood(const MatrixXf& data);
		virtual Array<float, 1, Dynamic> logLikelihood(const MatrixXf& data, const Parameters& params);
		virtual float evaluate(const MatrixXf& data, const Parameters& params = Parameters());

	protected:
		int mNumVisibles;
		int mNumHiddens;
		MatrixXf mBasis;
		vector<GSM> mSubspaces;
		MatrixXf mHiddenStates;
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



inline MatrixXf ISA::basis() {
	return mBasis;
}



inline void ISA::setBasis(const MatrixXf& basis) {
	if(basis.rows() != numVisibles() || basis.cols() != numHiddens())
		throw Exception("Basis has wrong dimensionality.");

	mBasis = basis;
}



inline MatrixXf ISA::hiddenStates() {
	return mHiddenStates;
}



inline void ISA::setHiddenStates(const MatrixXf& hiddenStates) {
	mHiddenStates = hiddenStates;
}

#endif
