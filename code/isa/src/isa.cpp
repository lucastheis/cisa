#include "isa.h"
#include "Eigen/LU"
#include "Eigen/SVD"
#include "Eigen/Eigenvalues"
#include "utils.h"
#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <cmath>
#include <functional>

using namespace std;

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
	callback = 0;
	persistent = true;

	sgd.maxIter = 1;
	sgd.batchSize = 100;
	sgd.stepWidth = 0.005;
	sgd.momentum = 0.8;
	sgd.shuffle = true;
	sgd.pocket = true;

	gsm.maxIter = 10;
	gsm.tol = 1e-8;

	gibbs.verbosity = 0;
	gibbs.iniIter = 10;
	gibbs.numIter = 2;

	ais.verbosity = 0;
	ais.numIter = 100;
	ais.numSamples = 10;
}



ISA::Parameters::Parameters(const Parameters& params) :
	verbosity(params.verbosity),
	trainingMethod(params.trainingMethod),
	samplingMethod(params.samplingMethod),
	maxIter(params.maxIter),
	adaptive(params.adaptive),
	trainPrior(params.trainPrior),
	persistent(params.persistent),
	callback(0),
	sgd(params.sgd),
	gsm(params.gsm),
	gibbs(params.gibbs),
	ais(params.ais)
{
	if(params.callback)
		callback = params.callback->copy();
}



ISA::Parameters::~Parameters() {
	if(callback)
		delete callback;
}



ISA::Parameters& ISA::Parameters::operator=(const Parameters& params) {
	verbosity = params.verbosity;
	trainingMethod = params.trainingMethod;
	samplingMethod = params.samplingMethod;
	maxIter = params.maxIter;
	adaptive = params.adaptive;
	trainPrior = params.trainPrior;
	persistent = params.persistent;
	callback = params.callback ? params.callback->copy() : 0;
	sgd = params.sgd;
	gsm = params.gsm;
	gibbs = params.gibbs;
	ais = params.ais;

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

			// sample from Laplace with unit variance
			RowVectorXd radial = (ArrayXXd::Random(1, 10000) + 1.) / 2.;
			radial = radial.array().log() / sqrt(2.);

			// sample from unit sphere and scale by radial component
			MatrixXd data = gaussian.sample(10000);
			data = normalize(data).array().rowwise() * radial.array();

			// fit GSM to multivariate Laplace distribution
			gsm = GSM(mSubspaces[i].dim(), mSubspaces[i].numScales());
			gsm.train(data, 200, 1e-8);
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
	N = N > data.rows() ? data.rows() : N;

	// store N largest data points and normalize
	MatrixXd dataWhiteLarge = MatrixXd::Zero(data.rows(), N);
	for(int i = 0; i < N; ++i)
		dataWhiteLarge.col(i) = dataWhite.col(indices[i]);
	dataWhiteLarge = normalize(dataWhiteLarge);

	// pick first basis vector at random
	mBasis.col(0) = dataWhiteLarge.col(rand() % N);

	MatrixXd innerProd;
	MatrixXd::Index j;

	for(int i = 1; i < numHiddens(); ++i) {
		// find data point with maximal inner product to other basis vectors
		innerProd = mBasis.leftCols(i).transpose() * dataWhiteLarge;
		innerProd.array().abs().colwise().maxCoeff().minCoeff(&j);
		mBasis.col(i) = dataWhiteLarge.col(j);
	}

	// orthogonalize and unwhiten
	SelfAdjointEigenSolver<MatrixXd> eigenSolver2(mBasis * mBasis.transpose());
	mBasis = eigenSolver1.operatorSqrt() * eigenSolver2.operatorInverseSqrt() * mBasis;
}



void ISA::train(const MatrixXd& data, Parameters params) {
	if(data.rows() != numVisibles())
		throw Exception("Data has wrong dimensionality.");

	if(params.callback)
		// call callback function once before training
		if(!(*params.callback)(0, *this))
			return;

	if(params.verbosity > 0) {
		cout << setw(5) << "Epoch";
		if(complete())
			cout << setw(14) << "Value";
		if(params.adaptive)
			cout << setw(14) << "Step width";
		cout << endl;
	}

	// initialize hidden states
	Parameters iniParams = params;
	iniParams.gibbs.numIter = iniParams.gibbs.iniIter;

	mHiddenStates = samplePosterior(data, iniParams);

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

		// optimize basis
		bool improved = trainSGD(complData, complBasis, params);

		if(params.callback)
			if(!(*params.callback)(i + 1, *this))
				break;

		if(params.verbosity > 0) {
			// print some information
			cout << setw(5) << i;
			if(complete())
				cout << setw(14) << fixed << setprecision(7) << evaluate(data);
			if(params.adaptive)
				cout << setw(14) << fixed << setprecision(7) << params.sgd.stepWidth;
			cout << endl;
		}

		if(params.adaptive)
			// adapt step width
			params.sgd.stepWidth *= improved ? 1.1 : 0.5;
	}
}



void ISA::trainPrior(const MatrixXd& states, const Parameters params) {
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
	const Parameters params)
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



MatrixXd ISA::samplePosterior(const MatrixXd& data, const Parameters params) {
	return samplePosterior(data, samplePrior(data.cols()), params);
}



MatrixXd ISA::samplePosterior(const MatrixXd& data, const MatrixXd& states, const Parameters params) {
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



pair<MatrixXd, MatrixXd> ISA::samplePosteriorAIS(const MatrixXd& data, const Parameters params) {
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
				annealingWeights[i] * isa.mSubspaces[j].scales() + (1. - annealingWeights[i]));

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



MatrixXd ISA::sampleNullspace(const MatrixXd& data, const Parameters params) {
	return nullspaceBasis() * samplePosterior(data, params);
}



MatrixXd ISA::priorLogLikelihood(const MatrixXd& states) {
	MatrixXd logLik = MatrixXd::Zero(states.rows(), states.cols());

	int from[numSubspaces()];
	for(int f = 0, i = 0; i < numSubspaces(); f += mSubspaces[i].dim(), ++i)
		from[i] = f;

	#pragma omp parallel for
	for(int i = 0; i < numSubspaces(); ++i)
		logLik.middleRows(from[i], mSubspaces[i].dim()) =
			mSubspaces[i].logLikelihood(states.middleRows(from[i], mSubspaces[i].dim()));

	return logLik.colwise().sum();
}



MatrixXd ISA::priorEnergy(const MatrixXd& states) {
	MatrixXd energy = MatrixXd::Zero(states.rows(), states.cols());

	int from[numSubspaces()];
	for(int f = 0, i = 0; i < numSubspaces(); f += mSubspaces[i].dim(), ++i)
		from[i] = f;

	#pragma omp parallel for
	for(int i = 0; i < numSubspaces(); ++i)
		energy.middleRows(from[i], mSubspaces[i].dim()) =
			mSubspaces[i].energy(states.middleRows(from[i], mSubspaces[i].dim()));

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



Array<double, 1, Dynamic> ISA::logLikelihood(const MatrixXd& data, const Parameters params) {
	if(data.rows() != numVisibles())
		throw Exception("Data has wrong dimensionality.");

	if(complete()) {
		// LU decomposition
		PartialPivLU<MatrixXd> basisLU(mBasis);

		// compute log-determinant of basis
		double logDet = basisLU.matrixLU().diagonal().array().abs().log().sum();

		// multiplication is faster than solve()
		MatrixXd states = basisLU.inverse() * data;
		MatrixXd logLik = MatrixXd::Zero(states.rows(), states.cols());

		int from[numSubspaces()];
		for(int f = 0, i = 0; i < numSubspaces(); f += mSubspaces[i].dim(), ++i)
			from[i] = f;

		#pragma omp parallel for
		for(int i = 0; i < numSubspaces(); ++i)
			logLik.middleRows(from[i], mSubspaces[i].dim()) =
				mSubspaces[i].logLikelihood(states.middleRows(from[i], mSubspaces[i].dim()));

		return logLik.colwise().sum().array() - logDet;
	} else {
		MatrixXd logWeights(params.ais.numSamples, data.cols());

		#pragma omp parallel for
		for(int i = 0; i < params.ais.numSamples; ++i)
			logWeights.row(i) = samplePosteriorAIS(data, params).second;

		return logmeanexp(logWeights);
	}
}



double ISA::evaluate(const MatrixXd& data, const Parameters params) {
	return -logLikelihood(data, params).mean() / log(2.) / dim();
}
