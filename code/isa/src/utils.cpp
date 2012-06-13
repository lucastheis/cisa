#include "utils.h"

Array<double, 1, Dynamic> logsumexp(const ArrayXXd& array) {
	Array<double, 1, Dynamic> array_max = array.colwise().maxCoeff();
	return array_max + (array.rowwise() - array_max).exp().colwise().sum().log();
}



Array<double, 1, Dynamic> logmeanexp(const ArrayXXd& array) {
	Array<double, 1, Dynamic> array_max = array.colwise().maxCoeff();
	return array_max + (array.rowwise() - array_max).exp().colwise().mean().log();
}



ArrayXXd sampleNormal(int m, int n) {
	ArrayXXd U1 = ArrayXXd::Random(m, n) / 2. + 0.5;
	ArrayXXd U2 = ArrayXXd::Random(m, n) / 2. + 0.5;
	// Box-Muller transform
	return (-2. * U1.log()).sqrt() * (2. * PI * U2).cos();
}
