#include "isa.h"
#include "Eigen/LU"
#include "Eigen/SVD"
#include "Eigen/Eigenvalues"
#include "utils.h"
#include "lbfgs.h"
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <cmath>
#include <functional>

using namespace std;

#if LBFGS_FLOAT != 64
#error "liblbfgs needs to be compiled with double precision."
#endif

static lbfgsfloatval_t evaluateLBFGS(void* instance, const lbfgsfloatval_t* x, lbfgsfloatval_t* g, int, double) {
	// unpack user data
	ISA* isa = static_cast<pair<ISA*, MatrixXd*>*>(instance)->first;
	const MatrixXd& data = *static_cast<pair<ISA*, const MatrixXd*>*>(instance)->second;

	// interpret parameters and gradients
	Map<Matrix<lbfgsfloatval_t, Dynamic, Dynamic> > W(const_cast<lbfgsfloatval_t*>(x), isa->numHiddens(), isa->numHiddens());
	Map<Matrix<lbfgsfloatval_t, Dynamic, Dynamic> > dW(g, isa->numHiddens(), isa->numHiddens());

	// compute hidden states
	MatrixXd states = W * data;

	// LU decomposition
	PartialPivLU<MatrixXd> filterLU(W);

	// log-determinant of filter matrix
	double logDet = filterLU.matrixLU().diagonal().array().abs().log().sum();

	// compute gradient
	dW = isa->priorEnergyGradient(states) * data.transpose() / data.cols() - filterLU.inverse().transpose();

	// return objective function value
	return isa->priorEnergy(states).mean() - logDet;
}



ISA::Callback::~Callback() {
}



ISA::Parameters::Parameters() {
	// default parameters
	verbosity = 0;
	trainingMethod = "SGD";
	samplingMethod = "Gibbs";
	maxIter = 10;
	adaptive = true;
	trainPrior = true;
	trainBasis = true;
	mergeSubspaces = false;
	orthogonalize = false;
	callback = 0;
	persistent = true;

	sgd.maxIter = 1;
	sgd.batchSize = 100;
	sgd.stepWidth = 0.005;
	sgd.momentum = 0.8;
	sgd.shuffle = true;
	sgd.pocket = true;

	lbfgs.maxIter = 50;
	lbfgs.numGrad = 10;

	mp.maxIter = 100;
	mp.batchSize = 100;
	mp.stepWidth = 0.01;
	mp.momentum = 0.8;
	mp.numCoeff = 10;
	mp.callback = 0;

	gsm.maxIter = 10;
	gsm.tol = 1e-8;

	gibbs.verbosity = 0;
	gibbs.iniIter = 10;
	gibbs.numIter = 2;

	ais.verbosity = 0;
	ais.numIter = 100;
	ais.numSamples = 10;

	merge.verbosity = 0;
	merge.maxMerge = 100;
	merge.maxIter = 10;
	merge.threshold = 0.;
}



ISA::Parameters::Parameters(const Parameters& params) :
	verbosity(params.verbosity),
	trainingMethod(params.trainingMethod),
	samplingMethod(params.samplingMethod),
	maxIter(params.maxIter),
	adaptive(params.adaptive),
	trainPrior(params.trainPrior),
	trainBasis(params.trainBasis),
	mergeSubspaces(params.mergeSubspaces),
	persistent(params.persistent),
	orthogonalize(params.orthogonalize),
	callback(0),
	sgd(params.sgd),
	lbfgs(params.lbfgs),
	mp(params.mp),
	gsm(params.gsm),
	gibbs(params.gibbs),
	ais(params.ais),
	merge(params.merge)
{
	if(params.callback)
		callback = params.callback->copy();

	if(params.mp.callback)
		mp.callback = params.mp.callback->copy();
}



ISA::Parameters::~Parameters() {
	if(callback)
		delete callback;
	if(mp.callback)
		delete mp.callback;
}



ISA::Parameters& ISA::Parameters::operator=(const Parameters& params) {
	verbosity = params.verbosity;
	trainingMethod = params.trainingMethod;
	samplingMethod = params.samplingMethod;
	maxIter = params.maxIter;
	adaptive = params.adaptive;
	trainPrior = params.trainPrior;
	trainBasis = params.trainBasis;
	mergeSubspaces = params.mergeSubspaces;
	orthogonalize = params.orthogonalize;
	persistent = params.persistent;
	callback = params.callback ? params.callback->copy() : 0;
	sgd = params.sgd;
	lbfgs = params.lbfgs;
	mp = params.mp;
	mp.callback = params.mp.callback ? params.mp.callback->copy() : 0;
	gsm = params.gsm;
	gibbs = params.gibbs;
	ais = params.ais;
	merge = params.merge;

	return *this;
}



ISA::ISA(int numVisibles, int numHiddens, int sSize, int numScales) :
	mNumVisibles(numVisibles), mNumHiddens(numHiddens)
{
	if(mNumHiddens < mNumVisibles)
		mNumHiddens = mNumVisibles;
	mBasis = ArrayXXd::Random(mNumVisibles, mNumHiddens) / 10.;

	for(int i = 0; i < mNumHiddens / sSize; ++i)
		mSubspaces.push_back(GSM(sSize, numScales));

	if(mNumHiddens % sSize)
		mSubspaces.push_back(GSM(mNumHiddens % sSize, numScales));
}



ISA::~ISA() {
}



MatrixXd ISA::nullspaceBasis() {
	// TODO: JacobiSVD is slow, can we replace it with something faster?
	JacobiSVD<MatrixXd> svd(basis(), ComputeFullV);
	return svd.matrixV().rightCols(numHiddens() - numVisibles()).transpose();
}



void ISA::initialize() {
	GSM gaussian;
	GSM gsm(numHiddens() + 1, 1);

	for(int i = 0; i < numSubspaces(); ++i) {
		if(gsm.dim() != mSubspaces[i].dim() || gsm.numScales() != mSubspaces[i].numScales()) {
			gaussian = GSM(mSubspaces[i].dim(), 1);

			// sample radial component from Gamma distribution
			RowVectorXd radial = sampleGamma(1, 10000, mSubspaces[i].dim());

			// sample from unit sphere and scale by radial component
			MatrixXd data = normalize(gaussian.sample(10000)).array().rowwise() * radial.array();

			// fit GSM to multivariate Laplace distribution
			gsm = GSM(mSubspaces[i].dim(), mSubspaces[i].numScales());
			gsm.train(data, 200, 1e-8);
			gsm.normalize();
			mSubspaces[i].setScales(gsm.scales());
		}

		mSubspaces[i].setScales(gsm.scales());
	}
}



void ISA::initialize(const MatrixXd& data) {
	if(data.rows() != numVisibles())
		throw Exception("Data has wrong dimensionality.");

	// whiten data
	SelfAdjointEigenSolver<MatrixXd> eigenSolver1(covariance(data));
	MatrixXd dataWhite = eigenSolver1.operatorInverseSqrt() * data;

	// sort data by norm descending
	VectorXi indices = argsort(dataWhite.colwise().squaredNorm());

	// largest index of largest 20% data points
	int N = data.cols() / 5;
	N = N < numHiddens() ? numHiddens() : N;
	N = N > data.cols() ? data.cols() : N;

	// store N largest data points and normalize
	MatrixXd dataWhiteLarge = MatrixXd::Zero(data.rows(), N);
	for(int i = 0; i < N; ++i)
		dataWhiteLarge.col(i) = dataWhite.col(indices[i]);
	dataWhiteLarge = normalize(dataWhiteLarge);

	// pick first basis vector at random
	mBasis.col(0) = dataWhiteLarge.col(rand() % N);

	MatrixXd innerProd;
	MatrixXd::Index j;

	for(int i = 1; i < min(numHiddens(), N); ++i) {
		// find data point with maximal inner product to other basis vectors
		innerProd = mBasis.leftCols(i).transpose() * dataWhiteLarge;
		innerProd.array().abs().colwise().maxCoeff().minCoeff(&j);
		mBasis.col(i) = dataWhiteLarge.col(j);
	}

	// orthogonalize and unwhiten
	SelfAdjointEigenSolver<MatrixXd> eigenSolver2(mBasis * mBasis.transpose());
	mBasis = eigenSolver1.operatorSqrt() * eigenSolver2.operatorInverseSqrt() * mBasis;
}



void ISA::orthogonalize() {
	// symmetrically orthogonalize basis
	SelfAdjointEigenSolver<MatrixXd> eigenSolver1(mBasis * mBasis.transpose());
	mBasis = eigenSolver1.operatorInverseSqrt() * mBasis;
}



void ISA::train(const MatrixXd& data, Parameters params) {
	if(data.rows() != numVisibles())
		throw Exception("Data has wrong dimensionality.");

	if(params.trainingMethod[0] == 'm' or params.trainingMethod[0] == 'M') {
		if(params.callback && !params.mp.callback)
			params.mp.callback = params.callback->copy();
		ISA::trainMP(data, params);
		return;
	}

	if(params.callback)
		// call callback function once before training
		if(!(*params.callback)(0, *this))
			return;

	if(params.verbosity > 0) {
		cout << setw(5) << "Epoch";
		if(complete())
			cout << setw(14) << "Value";
		if(params.adaptive && (params.trainingMethod[0] == 's' || params.trainingMethod[0] == 'S'))
			cout << setw(14) << "Step width";
		cout << endl;
	}

	if(mHiddenStates.cols() != data.cols() || mHiddenStates.rows() != numHiddens()) {
		Parameters iniParams = params;
		iniParams.gibbs.numIter = iniParams.gibbs.iniIter;

		// initialize hidden states
		mHiddenStates = samplePosterior(data, iniParams);
	}

	for(int i = 0; i < params.maxIter; ++i) {
		MatrixXd complBasis(numHiddens(), numHiddens());
		MatrixXd complData(numHiddens(), data.cols());

		// sample hidden states
		mHiddenStates = params.persistent ?
			samplePosterior(data, mHiddenStates, params) :
			samplePosterior(data, params);

		// completed basis and data
		complBasis << basis(), nullspaceBasis();
		complData << data, nullspaceBasis() * mHiddenStates;

		if(params.trainPrior)
			// optimize marginal distributions
			trainPrior(mHiddenStates, params);

		if(params.mergeSubspaces)
			mHiddenStates = mergeSubspaces(mHiddenStates, params);

		if(params.trainBasis) {
			// optimize basis
			bool improved;

			switch(params.trainingMethod[0]) {
				case 's':
				case 'S':
					improved = trainSGD(complData, complBasis, params);

					if(params.adaptive)
						// adapt step width
						params.sgd.stepWidth *= improved ? 1.1 : 0.5;
					break;

				case 'l':
				case 'L':
					trainLBFGS(complData, complBasis, params);
					break;

				default:
					throw Exception("Unknown training method.");
			}
		}

		if(params.verbosity > 0) {
			// print some information
			cout << setw(5) << i;
			if(complete())
				cout << setw(14) << fixed << setprecision(7) << evaluate(data);
			if(params.adaptive && (params.trainingMethod[0] == 's' || params.trainingMethod[0] == 'S') && params.trainBasis)
				cout << setw(14) << fixed << setprecision(7) << params.sgd.stepWidth;
			cout << endl;
		}

		if(params.trainBasis && params.orthogonalize)
			orthogonalize();

		if(params.callback)
			if(!(*params.callback)(i + 1, *this))
				break;
	}
}



void ISA::trainPrior(const MatrixXd& states, const Parameters& params) {
	int from[numSubspaces()];
	for(int f = 0, i = 0; i < numSubspaces(); f += mSubspaces[i].dim(), ++i)
		from[i] = f;

	#pragma omp parallel for
	for(int i = 0; i < numSubspaces(); ++i) {
		mSubspaces[i].train(
			states.middleRows(from[i], mSubspaces[i].dim()),
			params.gsm.maxIter,
			params.gsm.tol);

		// normalize marginal variance
		mBasis.middleCols(from[i], mSubspaces[i].dim()) *= sqrt(mSubspaces[i].variance());
		mSubspaces[i].normalize();
	}
}



bool ISA::trainSGD(
	const MatrixXd& complData,
	const MatrixXd& complBasis,
	const Parameters& params)
{
	// LU decomposition
	PartialPivLU<MatrixXd> basisLU(complBasis);

	// filter matrix, momentum and batch
	MatrixXd W = basisLU.inverse();
	MatrixXd P = MatrixXd::Zero(W.rows(), W.cols());
	MatrixXd X;

	// compute value of lower bound
	double logDet = basisLU.matrixLU().diagonal().array().abs().log().sum();
	double energy = priorEnergy(W * complData).array().mean() + logDet;

	for(int i = 0; i < params.sgd.maxIter; ++i) {
		for(int j = 0; j + params.sgd.batchSize <= complData.cols(); j += params.sgd.batchSize) {
			X = complData.middleCols(j, params.sgd.batchSize);

			// update momentum with natural gradient
			P = params.sgd.momentum * P + W
				- priorEnergyGradient(W * X) * X.transpose() / params.sgd.batchSize * (W.transpose() * W);

			// update filter matrix
			W += params.sgd.stepWidth * P;
		}
	}

	// compute LU decomposition from filter matrix
	PartialPivLU<MatrixXd> filterLU(W);

	// compute new value of lower bound
	double logDetNew = filterLU.matrixLU().diagonal().array().abs().log().sum();
	double energyNew = priorEnergy(W * complData).array().mean() - logDetNew;

	if(params.sgd.pocket && energy < energyNew)
		// don't update basis
		return false;

	// update basis
	setBasis(filterLU.inverse().topRows(numVisibles()));

	return energyNew < energy;
}



bool ISA::trainLBFGS(
	const MatrixXd& complData,
	const MatrixXd& complBasis,
	const Parameters& params)
{
	// compute initial filter matrix
	MatrixXd W = complBasis.inverse();

	// request memory for LBFGS
	lbfgsfloatval_t* x = lbfgs_malloc(W.size());

	// copy parameters
	for(int i = 0; i < W.size(); ++i)
		x[i] = W.data()[i];

	// optimization parameters
	lbfgs_parameter_t param;
	lbfgs_parameter_init(&param);
	param.max_iterations = params.lbfgs.maxIter;
	param.m = params.lbfgs.numGrad;

	pair<ISA*, const MatrixXd*> instance(this, &complData);

	// start LBFGS optimization
	lbfgs(W.size(), x, 0, &evaluateLBFGS, 0, &instance, &param);

	// copy optimized parameters back
	W = Map<Matrix<lbfgsfloatval_t, Dynamic, Dynamic> >(x, W.rows(), W.cols());

	// free memory used by LBFGS
	lbfgs_free(x);

	// update basis
	setBasis(W.inverse().topRows(numVisibles()));

	return true;
}



void ISA::trainMP(const MatrixXd& data, const Parameters& params) {
	// momentum, hidden and visible states
	MatrixXd P = MatrixXd::Zero(mBasis.rows(), mBasis.cols());
	MatrixXd X, Y;

	mBasis = normalize(mBasis);

	if(params.mp.callback)
		if(!(*params.mp.callback)(0, *this))
			return;

	int from[numSubspaces()];
	for(int f = 0, i = 0; i < numSubspaces(); f += mSubspaces[i].dim(), ++i)
		from[i] = f;

	for(int i = 0; i < params.mp.maxIter; ++i) {
		for(int j = 0; j + params.mp.batchSize <= data.cols(); j += params.mp.batchSize) {
			X = data.middleCols(j, params.mp.batchSize);

			// find coefficients
			Y = matchingPursuit(X, params);

			// update momentum with reconstruction gradient
			P = params.mp.momentum * P + (X - mBasis * Y) * Y.transpose() / params.mp.batchSize;

			// update filter matrix
			mBasis += params.mp.stepWidth * P;
			mBasis = normalize(mBasis);
		}

		if(params.mp.callback)
			if(!(*params.mp.callback)(i + 1, *this))
				break;
	}

	if(numSubspaces() != numHiddens())
		# pragma omp parallel for
		for(int j = 0; j < numSubspaces(); ++j) {
			// orthogonalize subspace
			MatrixXd subsp = mBasis.middleCols(from[j], mSubspaces[j].dim());
			SelfAdjointEigenSolver<MatrixXd> eigenSolver(subsp.transpose() * subsp);
			mBasis.middleCols(from[j], mSubspaces[j].dim()) = subsp * eigenSolver.operatorInverseSqrt();
		}
}



MatrixXd ISA::matchingPursuit(const MatrixXd& data, const Parameters& params) {
	MatrixXd hiddenStates = MatrixXd::Zero(numHiddens(), data.cols());

	// assumes basis is normalized
	MatrixXd responses = mBasis.transpose() * data;
	MatrixXd gramMatrix = mBasis.transpose() * mBasis;

	if(numSubspaces() == numHiddens()) {
		for(int i = 0; i < params.mp.numCoeff; ++i) {
			#pragma omp parallel for
			for(int j = 0; j < data.cols(); ++j) {
				// find maximally active coefficient
				int idx;
				responses.col(j).array().abs().maxCoeff(&idx);

				// update hidden states and filter responses
				double r = responses(idx, j);
				hiddenStates(idx, j) += r;
				responses.col(j) -= r * gramMatrix.col(idx);
			}
		}
	} else {
		// subspace responses
		MatrixXd ssResponses = MatrixXd(numSubspaces(), data.cols());

		int from[numSubspaces()];
		for(int f = 0, i = 0; i < numSubspaces(); f += mSubspaces[i].dim(), ++i)
			from[i] = f;

		for(int i = 0; i < params.mp.numCoeff; ++i) {
			// compute subspace responses
			#pragma omp parallel for
			for(int j = 0; j < numSubspaces(); ++j)
				ssResponses.row(j) = responses.middleRows(from[j], mSubspaces[j].dim()).array().square().colwise().sum();

			#pragma omp parallel for
			for(int j = 0; j < data.cols(); ++j) {
				// find maximally active coefficient
				int idx;
				ssResponses.col(j).maxCoeff(&idx);

				for(int k = 0; k < mSubspaces[idx].dim(); ++k) {
					// update hidden states and filter responses
					double l = from[idx] + k;
					double r = responses(l, j);
					hiddenStates(l, j) += r;
					responses.col(j) -= r * gramMatrix.col(l);
				}
			}
		}
	}

	return hiddenStates;
}



MatrixXd ISA::mergeSubspaces(MatrixXd states, const Parameters& params) {
	if(numSubspaces() > 1) {
		vector<int> from(numSubspaces());
		for(int f = 0, i = 0; i < numSubspaces(); f += mSubspaces[i].dim(), ++i)
			from[i] = f;

		// compute subspace energies
		MatrixXd energies(numSubspaces(), states.cols());

		#pragma omp parallel for
		for(int i = 0; i < numSubspaces(); ++i)
			energies.row(i) = states.middleRows(from[i], mSubspaces[i].dim()).colwise().norm();

		// compute correlations between subspaces
		MatrixXd corr = corrcoef(energies).triangularView<StrictlyLower>();

		for(int i = 0; i < params.merge.maxMerge; ++i) {
			// find the two maximally correlated subspaces
			int row, col;
			corr.colwise().maxCoeff().maxCoeff(&col);
			corr.col(col).maxCoeff(&row);

			if(corr(row, col) <= 0.)
				break;

			if(row == col)
				throw Exception("Something went wrong.");

			// makes sure subspaces aren't selected again
			corr(row, col) = 0.;

			// data corresponding to subspaces
			MatrixXd statesRow = states.middleRows(from[row], mSubspaces[row].dim());
			MatrixXd statesCol = states.middleRows(from[col], mSubspaces[col].dim());
			MatrixXd statesJnt(mSubspaces[row].dim() + mSubspaces[col].dim(), states.cols());

			statesJnt << statesRow, statesCol;

			// train a joint model
			GSM gsm(statesJnt.rows(), mSubspaces[row].numScales());
			gsm.setScales(mSubspaces[row].scales());
			gsm.train(statesJnt, params.merge.maxIter);

			// log-likelihood improvement
			double mi = gsm.logLikelihood(statesJnt).mean()
				- mSubspaces[row].logLikelihood(statesRow).mean()
				- mSubspaces[col].logLikelihood(statesCol).mean();

			if(mi > params.merge.threshold) {
				mSubspaces.push_back(gsm);

				// indices of subspace dimensions
				vector<int> indices;
				for(int i = 0; i < mSubspaces[row].dim(); ++i)
					indices.push_back(from[row] + i);
				for(int i = 0; i < mSubspaces[col].dim(); ++i)
					indices.push_back(from[col] + i);

				// rearrange basis vectors
				MatrixXd basisRow = mBasis.middleCols(from[row], mSubspaces[row].dim());
				MatrixXd basisCol = mBasis.middleCols(from[col], mSubspaces[col].dim());

				MatrixXd basisDel = deleteCols(mBasis, indices);
				mBasis << basisDel, basisRow, basisCol;

				// rearrange hidden states
				MatrixXd statesDel = deleteRows(states, indices);
				states << statesDel, statesJnt;

				// remove subspaces from correlation matrix
				vector<int> rc;
				rc.push_back(row);
				rc.push_back(col);
				corr = deleteRows(corr, rc);
				corr = deleteCols(corr, rc);

				// update subspace indices
				for(unsigned int k = row + 1; k < from.size(); ++k)
					from[k] -= mSubspaces[row].dim();
				for(unsigned int k = col + 1; k < from.size(); ++k)
					from[k] -= mSubspaces[col].dim();

				if(row < col) {
					from.erase(from.begin() + col);
					from.erase(from.begin() + row);
					mSubspaces.erase(mSubspaces.begin() + col);
					mSubspaces.erase(mSubspaces.begin() + row);
				} else {
					from.erase(from.begin() + row);
					from.erase(from.begin() + col);
					mSubspaces.erase(mSubspaces.begin() + row);
					mSubspaces.erase(mSubspaces.begin() + col);
				}

				if(params.merge.verbosity > 0)
					cout << "Merged subspaces." << endl;

				if(corr.rows() < 2)
					// no subspaces left to merge
					break;
			}
		}
	}

	return states;
}



MatrixXd ISA::sample(int numSamples) {
	return basis() * samplePrior(numSamples);
}



MatrixXd ISA::samplePrior(int numSamples) {
	MatrixXd samples = MatrixXd::Zero(numHiddens(), numSamples);

	// TODO: parallelize
	for(int from = 0, i = 0; i < numSubspaces(); from += mSubspaces[i].dim(), ++i)
		samples.middleRows(from, mSubspaces[i].dim()) =
			mSubspaces[i].sample(numSamples);

	return samples;
}



MatrixXd ISA::sampleScales(const MatrixXd& states) {
	if(states.rows() != numHiddens())
		throw Exception("Hidden states have wrong dimensionality.");

	MatrixXd scales = MatrixXd::Zero(states.rows(), states.cols());

	int from[numSubspaces()];
	for(int f = 0, i = 0; i < numSubspaces(); f += mSubspaces[i].dim(), ++i)
		from[i] = f;

	#pragma omp parallel for
	for(int i = 0; i < numSubspaces(); ++i)
		scales.middleRows(from[i], mSubspaces[i].dim()).rowwise() =
			mSubspaces[i].samplePosterior(states.middleRows(from[i], mSubspaces[i].dim())).matrix();

	return scales;
}



MatrixXd ISA::samplePosterior(const MatrixXd& data, const Parameters& params) {
	return samplePosterior(data, samplePrior(data.cols()), params);
}



MatrixXd ISA::samplePosterior(const MatrixXd& data, const MatrixXd& states, const Parameters& params) {
	if(data.rows() != numVisibles())
		throw Exception("Data has wrong dimensionality.");

	if(complete())
		return basis().inverse() * data;

	if(data.cols() != states.cols())
		throw Exception("The number of hidden states and the number of data points should be equal.");

	// scales, variances, and visible states
	MatrixXd S, v, X;

	// basis and nullspace basis
	MatrixXd& A = mBasis;
	MatrixXd B = nullspaceBasis();
	MatrixXd At = A.transpose();
	MatrixXd Bt = B.transpose();

	// nullspace projection matrix
	MatrixXd Q = Bt * B;

	// part of the hidden representation
	MatrixXd WX = At * (A * At).llt().solve(data);

	// initialize Markov chain
	MatrixXd Y = WX + Q * states;

	for(int i = 0; i < params.gibbs.numIter; ++i) {
		// sample scales
		S = sampleScales(Y);
		v = S.array().square();

		// sample source variables
		Y = sampleNormal(numHiddens(), data.cols()) * S.array();
		X = data - A * Y;

		#pragma omp parallel for
		for(int j = 0; j < data.cols(); ++j) {
			MatrixXd vAt = v.col(j).asDiagonal() * At;
			Y.col(j) = WX.col(j) + Q * (Y.col(j) + vAt * (A * vAt).llt().solve(X.col(j)));
		}

		if(params.gibbs.verbosity > 0)
			cout << setw(10) << i << setw(12) << fixed << setprecision(4) << priorEnergy(Y).mean() << endl;
	}

	return Y;
}



pair<MatrixXd, MatrixXd> ISA::samplePosteriorAIS(const MatrixXd& data, const Parameters& params) {
	VectorXd annealingWeights = VectorXd::LinSpaced(params.ais.numIter + 1, 0.0, 1.0).bottomRows(params.ais.numIter);

	// initialize proposal distribution to be Gaussian
	ISA isa = *this;

	for(int j = 0; j < isa.numSubspaces(); ++j)
		isa.mSubspaces[j].setScales(VectorXd::Ones(isa.mSubspaces[j].numScales()));

	// scales, variances, and visible states
	MatrixXd S, v, X;

	// basis and nullspace basis
	MatrixXd& A = mBasis;
	MatrixXd B = nullspaceBasis();
	MatrixXd At = A.transpose();
	MatrixXd Bt = B.transpose();

	// nullspace projection matrix
	MatrixXd Q = Bt * B;

	// part of the hidden representation
	MatrixXd WX = At * (A * At).llt().solve(data);

	// initialize hidden states
	MatrixXd Y = WX + Q * isa.samplePrior(data.cols());

	// importance weights
	MatrixXd logWeights = (B * Y).colwise().squaredNorm().array() / 2.
		+ (numHiddens() - numVisibles()) * log(2. * PI) / 2. - logDetPD(A * At) / 2.;

	for(int i = 0; i < params.ais.numIter; ++i) {
		// adjust proposal distribution
		for(int j = 0; j < isa.numSubspaces(); ++j)
			isa.mSubspaces[j].setScales(
				annealingWeights[i] * mSubspaces[j].scales() + (1. - annealingWeights[i]));

		logWeights -= isa.priorEnergy(Y);

		// sample scales
		S = isa.sampleScales(Y);
		v = S.array().square();

		// sample source variables
		Y = sampleNormal(numHiddens(), data.cols()) * S.array();
		X = data - A * Y;

		#pragma omp parallel for
		for(int j = 0; j < data.cols(); ++j) {
			MatrixXd vAt = v.col(j).asDiagonal() * At;
			Y.col(j) = WX.col(j) + Q * (Y.col(j) + vAt * (A * vAt).llt().solve(X.col(j)));
		}

		logWeights += isa.priorEnergy(Y);

		if(params.ais.verbosity > 0)
			cout << setw(10) << i << setw(12) << fixed << setprecision(4) << priorEnergy(Y).mean() << endl;
	}

	logWeights += priorLogLikelihood(Y);

	return pair<MatrixXd, MatrixXd>(Y, logWeights);
}



MatrixXd ISA::sampleNullspace(const MatrixXd& data, const Parameters& params) {
	return nullspaceBasis() * samplePosterior(data, params);
}



MatrixXd ISA::priorLogLikelihood(const MatrixXd& states) {
	MatrixXd logLik = MatrixXd::Zero(numSubspaces(), states.cols());

	int from[numSubspaces()];
	for(int f = 0, i = 0; i < numSubspaces(); f += mSubspaces[i].dim(), ++i)
		from[i] = f;

	#pragma omp parallel for
	for(int i = 0; i < numSubspaces(); ++i)
		logLik.row(i) = mSubspaces[i].logLikelihood(
			states.middleRows(from[i], mSubspaces[i].dim()));

	return logLik.colwise().sum();
}



MatrixXd ISA::priorEnergy(const MatrixXd& states) {
	MatrixXd energy = MatrixXd::Zero(numSubspaces(), states.cols());

	int from[numSubspaces()];
	for(int f = 0, i = 0; i < numSubspaces(); f += mSubspaces[i].dim(), ++i)
		from[i] = f;

	#pragma omp parallel for
	for(int i = 0; i < numSubspaces(); ++i)
		energy.row(i) = mSubspaces[i].energy(
			states.middleRows(from[i], mSubspaces[i].dim()));

	return energy.colwise().sum();
}



MatrixXd ISA::priorEnergyGradient(const MatrixXd& states) {
	MatrixXd gradient = MatrixXd::Zero(states.rows(), states.cols());

	int from[numSubspaces()];
	for(int f = 0, i = 0; i < numSubspaces(); f += mSubspaces[i].dim(), ++i)
		from[i] = f;

	#pragma omp parallel for
	for(int i = 0; i < numSubspaces(); ++i)
		gradient.middleRows(from[i], mSubspaces[i].dim()) =
			mSubspaces[i].energyGradient(states.middleRows(from[i], mSubspaces[i].dim()));

	return gradient;
}



Array<double, 1, Dynamic> ISA::logLikelihood(const MatrixXd& data) {
	return logLikelihood(data, Parameters());
}



Array<double, 1, Dynamic> ISA::logLikelihood(const MatrixXd& data, const Parameters& params) {
	if(data.rows() != numVisibles())
		throw Exception("Data has wrong dimensionality.");

	if(complete()) {
		// LU decomposition
		PartialPivLU<MatrixXd> basisLU(mBasis);

		// compute log-determinant of basis
		double logDet = basisLU.matrixLU().diagonal().array().abs().log().sum();

		return priorLogLikelihood(basisLU.inverse() * data).array() - logDet;
	} else {
		return logmeanexp(sampleAIS(data, params));
	}
}



MatrixXd ISA::sampleAIS(const MatrixXd& data, const Parameters& params) {
	MatrixXd logWeights(params.ais.numSamples, data.cols());

	#pragma omp parallel for
	for(int i = 0; i < params.ais.numSamples; ++i)
		logWeights.row(i) = samplePosteriorAIS(data, params).second;

	return logWeights;
}



double ISA::evaluate(const MatrixXd& data, const Parameters& params) {
	return -logLikelihood(data, params).mean() / log(2.) / dim();
}
