import sys
import unittest

sys.path.append('./code')
sys.path.append('./build/lib.macosx-10.6-intel-2.7')
sys.path.append('./build/lib.linux-x86_64-2.7')

from isa import GSM
from numpy import asarray, isnan, any, sqrt, sum, square, std
from numpy.random import randn, rand
from scipy.stats import kstest, norm, laplace, cauchy
from scipy.optimize import check_grad
from pickle import dump, load
from tempfile import mkstemp

class Tests(unittest.TestCase):
	def test_sample(self):
		gsm = GSM(2, 1)
		gsm.scales = [2]

		samples = gsm.sample(100)

		# simple sanity checks
		self.assertEqual(samples.shape[0], 2)
		self.assertEqual(samples.shape[1], 100)

		p = kstest(gsm.sample(10000).flatten(), lambda x: norm.cdf(x, scale=2.))[1]

		# Gaussianity test
		self.assertTrue(p > 0.0001)



	def test_train(self):
		gsm = GSM(1, 10)
		gsm.train(laplace.rvs(size=[1, 10000]), max_iter=100, tol=-1)

		p = kstest(gsm.sample(10000).flatten(), laplace.cdf)[1]

		# test whether GSM faithfully reproduces Laplace samples
		self.assertTrue(p > 0.0001)

		gsm = GSM(1, 6)
		gsm.train(cauchy.rvs(size=[1, 10000]), max_iter=100, tol=-1)

		# test for stability of training
		self.assertTrue(not any(isnan(gsm.scales)))



	def test_posterior(self):
		gsm = GSM(1, 10)
		gsm.train(laplace.rvs(size=[1, 10000]), max_iter=100, tol=-1)

		samples = gsm.sample(100)
		posterior = gsm.posterior(samples)

		# simple sanity checks
		self.assertEqual(posterior.shape[0], gsm.scales.shape[0])
		self.assertEqual(posterior.shape[1], samples.shape[1])



	def test_energy_gradient(self):
		gsm = GSM(2, 10)
		gsm.train(randn(2, 10000) * laplace.rvs(size=[1, 10000]), max_iter=100, tol=-1)

		samples = gsm.sample(100)
		gradient = gsm.energy_gradient(samples)

		# simple sanity checks
		self.assertEqual(gradient.shape[0], samples.shape[0])
		self.assertEqual(gradient.shape[1], samples.shape[1])

		f = lambda x: gsm.energy(x.reshape(-1, 1)).flatten()
		df = lambda x: gsm.energy_gradient(x.reshape(-1, 1)).flatten()

		for i in range(samples.shape[1]):
			relative_error = check_grad(f, df, samples[:, i]) / sqrt(sum(square(df(samples[:, i]))))

			# comparison with numerical gradient
			self.assertLess(relative_error, 0.001)



	def test_normalize(self):
		gsm = GSM(1, 5)
		gsm.scales = rand(gsm.num_scales) + 1.
		gsm.normalize()

		# variance should be 1 after normalization
		self.assertLess(abs(gsm.variance() - 1.), 1e-8)
		self.assertLess(abs(std(gsm.sample(10**6), ddof=1) - 1.), 1e-2)



	def test_pickle(self):
		gsm0 = GSM(3, 11)

		tmp_file = mkstemp()[1]

		# store model
		with open(tmp_file, 'w') as handle:
			dump({'gsm': gsm0}, handle)

		# load model
		with open(tmp_file) as handle:
			gsm1 = load(handle)['gsm']

		# make sure parameters haven't changed
		self.assertEqual(gsm0.dim, gsm1.dim)
		self.assertEqual(gsm0.num_scales, gsm1.num_scales)
		self.assertLess(max(abs(gsm0.scales - gsm1.scales)), 1e-20)



if __name__ == '__main__':
	unittest.main()
