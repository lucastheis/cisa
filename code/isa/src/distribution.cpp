#include "distribution.h"
#include <cmath>

using std::log;

Distribution::~Distribution() {
}



float Distribution::evaluate(const MatrixXf& data) {
	return -logLikelihood(data).mean() / log(2.) / dim();
}
