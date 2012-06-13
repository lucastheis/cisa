import sys
import unittest

sys.path.append('./code')
sys.path.append('./build/lib.macosx-10.6-intel-2.7')
sys.path.append('./build/lib.linux-x86_64-2.7')

from isa import ISA
from numpy import sqrt, sum, square, dot, var, eye, cov, diag, std, max, asarray, mean
from numpy.linalg import inv, eig
from numpy.random import randn
from scipy.optimize import check_grad
from scipy.stats import kstest, laplace, ks_2samp

class Tests(unittest.TestCase):
#	def test_default_parameters(self):
#		isa = ISA(2, 4)
#		params = isa.default_parameters()
#
#		# simple sanity checks
#		self.assertTrue(isinstance(params, dict))
#		self.assertEqual(sys.getrefcount(params) - 1, 1)
#
#
#
#	def test_nullspace_basis(self):
#		isa = ISA(2, 5)
#		B = isa.nullspace_basis()
#
#		# simple sanity checks
#		self.assertTrue(B.shape[0], 3)
#		self.assertTrue(B.shape[1], 5)
#
#		# B should be orthogonal to A and orthonormal
#		self.assertLess(max(abs(dot(isa.A, B.T).flatten())), 1e-10)
#		self.assertLess(max(abs((dot(B, B.T) - eye(3)).flatten())), 1e-10)
#
#		self.assertEqual(sys.getrefcount(B) - 1, 1)
#
#
#
#	def test_initialize(self):
#		def sqrtmi(mat):
#			"""
#			Compute matrix inverse square root.
#
#			@type  mat: array_like
#			@param mat: matrix for which to compute inverse square root
#			"""
#
#			# find eigenvectors
#			eigvals, eigvecs = eig(mat)
#
#			# eliminate eigenvectors whose eigenvalues are zero
#			eigvecs = eigvecs[:, eigvals > 0]
#			eigvals = eigvals[eigvals > 0]
#
#			# inverse square root
#			return dot(eigvecs, dot(diag(1. / sqrt(eigvals)), eigvecs.T))
#
#		# white data
#		data = randn(5, 1000)
#		data = dot(sqrtmi(cov(data)), data)
#
#		isa = ISA(5, 10)
#		isa.initialize(data)
#
#		# rows of A should be roughly orthogonal
#		self.assertTrue(sum(square(dot(isa.A, isa.A.T) - eye(5)).flatten()) < 1e-3)
#
#		p = kstest(
#			isa.sample_prior(100).flatten(),
#			lambda x: laplace.cdf(x, scale=1. / sqrt(2.)))[1]
#
#		# prior marginals should be roughly Laplace
#		self.assertTrue(p > 0.0001)
#
#		# test initialization with larger subspaces
#		isa = ISA(5, 10, ssize=2)
#		isa.initialize(data)
#
#
#
#	def test_subspaces(self):
#		isa = ISA(2, 4, 2)
#
#		# simple sanity checks
#		self.assertEqual(isa.subspaces()[0].dim, 2)
#		self.assertEqual(isa.subspaces()[1].dim, 2)
#
#		self.assertEqual(sys.getrefcount(isa.subspaces), 1)
#
#
#
#	def test_train(self):
#		# make sure train() doesn't throw any errors
#		isa = ISA(2)
#		params = isa.default_parameters()
#		params['verbosity'] = 0
#		params['max_iter'] = 2
#		params['training_method'] = 'SGD'
#		params['sgd']['max_iter'] = 1
#		params['sgd']['batch_size'] = 57
#
#		isa.initialize(randn(2, 1000))
#		isa.train(randn(2, 1000), params)
#
#		isa = ISA(4, ssize=2)
#		isa.initialize(randn(4, 1000))
#		isa.train(randn(4, 1000), params)
#
#
#
#	def test_sample_prior(self):
#		isa = ISA(5, 10)
#		samples = isa.sample_prior(20)
#
#		# simple sanity checks
#		self.assertEqual(samples.shape[0], 10)
#		self.assertEqual(samples.shape[1], 20)
#
#
#
#	def test_sample(self):
#		isa = ISA(3, 4)
#
#		samples = isa.sample(100)
#		samples_prior = isa.sample_prior(100)
#
#		# simple sanity checks
#		self.assertEqual(samples.shape[0], isa.num_visibles)
#		self.assertEqual(samples.shape[1], 100)
#		self.assertEqual(samples_prior.shape[0], isa.num_hiddens)
#		self.assertEqual(samples_prior.shape[1], 100)
#
#
#
#	def test_prior_energy_gradient(self):
#		isa = ISA(4)
#
#		samples = isa.sample_prior(100)
#		grad = isa.prior_energy_gradient(samples)
#
#		# simple sanity checks
#		self.assertEqual(grad.shape[0], samples.shape[0])
#		self.assertEqual(grad.shape[1], samples.shape[1])
#
#		f = lambda x: isa.prior_energy(x.reshape(-1, 1)).flatten()
#		df = lambda x: isa.prior_energy_gradient(x.reshape(-1, 1)).flatten()
#
#		for i in range(samples.shape[1]):
#			relative_error = check_grad(f, df, samples[:, i]) / sqrt(sum(square(df(samples[:, i]))))
#
#			# comparison with numerical gradient
#			self.assertLess(relative_error, 0.001)
#
#
#
#	def test_loglikelihood(self):
#		isa = ISA(7)
#
#		samples = isa.sample(100)
#
#		energy = isa.prior_energy(dot(inv(isa.A), samples))
#		loglik = isa.loglikelihood(samples)
#
#		# difference between loglik and -energy should be const
#		self.assertTrue(var(loglik + energy) < 1e-10)
#
#
#
#	def test_callback(self):
#		isa = ISA(2)
#
#		# callback function
#		def callback(i, isa_):
#			callback.count += 1
#			self.assertTrue(isa == isa_)
#		callback.count = 0
#
#		# set callback function
#		parameters = {
#				'verbosity': 0,
#				'max_iter': 7,
#				'callback': callback,
#				'sgd': {'max_iter': 0}
#			}
#
#		isa.train(randn(2, 1000), parameters=parameters)
#
#		# test how often callback function was called
#		self.assertEqual(callback.count, parameters['max_iter'] + 1)
#
#		def callback(i, isa_):
#			if i == 5:
#				return False
#			callback.count += 1
#		callback.count = 0
#
#		parameters['callback'] = callback
#
#		isa.train(randn(2, 1000), parameters=parameters)
#
#		# test how often callback function was called
#		self.assertEqual(callback.count, 5)
#
#		# make sure referece counts stay stable
#		self.assertEqual(sys.getrefcount(isa) - 1, 1)
#		self.assertEqual(sys.getrefcount(callback) - 1, 2)
#
#
#
#	def test_sample_scales(self):
#		isa = ISA(2, 5, num_scales=4)
#
#		# get a copy of subspaces
#		subspaces = isa.subspaces()
#
#		# replace scales
#		for gsm in subspaces:
#			gsm.scales = asarray([1., 2., 3., 4.])
#
#		isa.set_subspaces(subspaces)
#
#		samples = isa.sample_prior(100000)
#		scales = isa.sample_scales(samples)
#
#		# simple sanity checks
#		self.assertEqual(scales.shape[0], isa.num_hiddens)
#		self.assertEqual(scales.shape[1], samples.shape[1])
#
#		priors = mean(abs(scales.flatten() - asarray([[1., 2., 3., 4.]]).T) < 0.5, 1)
#
#		# prior probabilities of scales should be equal and sum up to one
#		self.assertLess(max(abs(priors - 1. / subspaces[0].num_scales)), 0.01)
#		self.assertLess(abs(sum(priors) - 1.), 1e-10)

	
	def test_sample_posterior(self):
		isa = ISA(2, 4, num_scales=5)

		isa.initialize()

		params = isa.default_parameters()
		params['gibbs']['verbosity'] = 1
		params['gibbs']['num_iter'] = 1000

		states_post = isa.sample_posterior(isa.sample(1000), params)
		states_prio = isa.sample_prior(states_post.shape[1])

		from matplotlib.pyplot import figure, plot, show

		figure()
		plot(states_post[0], states_post[1], '.')
		figure()
		plot(states_prio[0], states_prio[1], '.')
		show()

		p = ks_2samp(states_post.flatten(), states_prio.flatten())[1]
   	
		print p



if __name__ == '__main__':
	unittest.main()
