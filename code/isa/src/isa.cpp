#include "isa.h"
#include "Eigen/LU"
#include "Eigen/SVD"
#include "Eigen/Eigenvalues"
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <cstdlib>
#include <cmath>
#include <functional>

using namespace std;

VectorXi argsort(const VectorXd& data) {
	// create pairs of values and indices
	vector<pair<double, int> > pairs(data.size());
	for(int i = 0; i < data.size(); ++i) {
		pairs[i].first = data[i];
		pairs[i].second = i;
	}

	// sort values in descending order
	sort(pairs.begin(), pairs.end(), greater<pair<double, int> >());

	// store indices
	VectorXi indices(data.size());
	for(int i = 0; i < data.size(); ++i)
		indices[pairs[i].second] = i;

	return indices;
}



MatrixXd covariance(const MatrixXd& data) {
	MatrixXd data_centered = data.colwise() - data.rowwise().mean().eval();
	return data_centered * data_centered.transpose() / data.cols();
}



MatrixXd normalize(const MatrixXd& matrix) {
	return matrix.array().rowwise() / matrix.colwise().norm().eval().array();
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



ISA::Parameters::Parameters(const Parameters& params) :
	verbosity(params.verbosity),
	trainingMethod(params.trainingMethod),
	samplingMethod(params.samplingMethod),
	maxIter(params.maxIter),
	adaptive(params.adaptive),
	trainPrior(params.trainPrior),
	callback(0),
	SGD(params.SGD),
	GSM(params.GSM)
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
	callback = params.callback ? params.callback->copy() : 0;
	SGD = params.SGD;
	GSM = params.GSM;

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
	if(!complete())
		throw Exception("Training of overcomplete models not implemented yet.");

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

	for(int i = 0; i < params.maxIter; ++i) {
		// optimize basis
		bool improved = trainSGD(data, basis(), params);

		if(params.trainPrior)
			// optimize marginal distributions
			trainPrior(basis().inverse() * data, params);

		if(params.callback)
			if(!(*params.callback)(i + 1, *this))
				break;

		if(params.verbosity > 0) {
			// print some information
			cout << setw(5) << i;
			if(complete())
				cout << setw(14) << fixed << setprecision(7) << evaluate(data);
			if(params.adaptive)
				cout << setw(14) << fixed << setprecision(7) << params.SGD.stepWidth;
			cout << endl;
		}

		if(params.adaptive)
			// adapt step width
			params.SGD.stepWidth *= improved ? 1.1 : 0.5;
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
			params.GSM.maxIter,
			params.GSM.tol);

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

	for(int i = 0; i < params.SGD.maxIter; ++i) {
		for(int j = 0; j + params.SGD.batchSize <= complData.cols(); j += params.SGD.batchSize) {
			X = complData.middleCols(j, params.SGD.batchSize);

			// update momentum with natural gradient
			P = params.SGD.momentum * P + W
				- priorEnergyGradient(W * X) * X.transpose() / params.SGD.batchSize * (W.transpose() * W);

			// update filter matrix
			W += params.SGD.stepWidth * P;
		}
	}

	// compute LU decomposition from filter matrix
	PartialPivLU<MatrixXd> filterLU(W);

	// compute new value of lower bound
	double logDetNew = filterLU.matrixLU().diagonal().array().abs().log().sum();
	double energyNew = priorEnergy(W * complData).array().mean() - logDetNew;

	if(params.SGD.pocket && energy < energyNew)
		// don't update basis
		return false;

	// update basis
	setBasis(filterLU.inverse().leftCols(numHiddens()));

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



MatrixXd ISA::sampleNullspace(const MatrixXd& data, const Parameters params) {
	if(data.rows() != numVisibles())
		throw Exception("Data has wrong dimensionality.");

	// TODO: implement Gibbs sampling
	return nullspaceBasis() * samplePrior(data.cols());
}



MatrixXd ISA::samplePosterior(const MatrixXd& data, const Parameters params) {
	MatrixXd complData(numHiddens(), data.cols());
	MatrixXd complBasis(numHiddens(), numHiddens());

	complData << data, sampleNullspace(data);
	complBasis << basis(), nullspaceBasis();

	return complBasis.inverse() * complData;
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
	if(data.rows() != numVisibles())
		throw Exception("Data has wrong dimensionality.");

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
}
