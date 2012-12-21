#ifndef DISTRIBUTION_H
#define DISTRIBUTION_H

#include "Eigen/Core"

using namespace Eigen;

class Distribution {
	public:
		virtual ~Distribution();

		virtual int dim() = 0;
		virtual Array<float, 1, Dynamic> logLikelihood(const MatrixXf& data) = 0;
		virtual float evaluate(const MatrixXf& data);
};

#endif
