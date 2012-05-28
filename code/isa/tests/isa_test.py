import sys
import unittest

sys.path.append('./code')

from isa import ISA
from numpy.random import randn

class Tests(unittest.TestCase):
	def test_default_parameters(self):
		# make sure default_parameters() works as expected
		isa = ISA(2, 4)
		params = isa.default_parameters()
		self.assertTrue(isinstance(params, dict))



	def test_train(self):
		# make sure train() doesn't throw any errors
		isa = ISA(2)
		params = isa.default_parameters()
		params['training_method'] = 'SGD'
		params['SGD']['max_iter'] = 1
		params['SGD']['batch_size'] = 57
		isa.train(randn(2, 1000), params)


if __name__ == '__main__':
	unittest.main()
