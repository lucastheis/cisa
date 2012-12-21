# ISA

A C++ implementation of overcomplete independent subspace analysis.

This code implements an efficient blocked Gibbs sampler for inference and maximum likelihood
learning in overcomplete linear models with sparse source distributions.

## Requirements

* Python >= 2.6.0
* NumPy >= 1.6.2
* automake >= 1.11.0
* libtool >= 2.4.0

I have tested the code with the above versions, but older versions might also work.

## Installation

### Linux

Make sure autoconf, automake and libtool are installed.

	apt-get install autoconf automake libtool

Go to `./code/liblbfgs` and execute the following:

	./autogen.sh
	./configure --enable-sse2
	make CFLAGS="-fPIC"

Once the L-BFGS library is compiled, go back to the root directory and execute:

	python setup.py build
	python setup.py install

### Mac OS X

First, make sure you have recent versions of automake and libtool installed. The versions that come
with Xcode 4.3 didn't work for me. You can use [Homebrew](http://mxcl.github.com/homebrew/) to install
newer ones.

	brew install autoconf automake libtool
	brew link autoconf automake libtool

Next, go to `./code/liblbfgs` and execute the following:

	./autogen.sh
	./configure --disable-dependency-tracking --enable-sse2
	make CFLAGS="-arch x86_64 -arch i386"

Once the L-BFGS library is compiled, go back to the root directory and execute:

	python setup.py build
	python setup.py install

### Building with the Intel compiler and MKL

To get even better performance, you might want to try compiling the module with Intel's compiler and
the MKL libraries. This probably requires some changes of the paths in `setup.py`. After that, use
the following line to compile the code

	python setup.py build --compiler=intelem

on 64-bit systems and

	python setup.py build --compiler=intel

on 32-bit systems.

## Example

```python
from isa import ISA

# create overcomplete model with two-dimensional subspaces
isa = ISA(num_visibles=16, num_hiddens=32, ssize=2)

# initialize filters and source distributions
isa.initialize(data)

# data should be stored in a 16xN NumPy array
data = load('data.npz')['data']

# will be called in every iteration
def callback(i, isa):
	print i

# optimize basis using matching pursuit
isa.train(data, parameters={
	'training_method': 'mp',
	'mp': {
		'max_iter': 50,
		'step_width': 0.01,
		'batch_size': 100,
		'num_coeff': 10},
	'callback': callback})

# optimize model using persistent EM
isa.train(data, parameters={
	'max_iter': 100, # number of EM iterations
	'training_method': 'lbfgs',
	'lbfgs': {
		'max_iter': 100, # number of iterations in each M-step
	},
	'sampling_method': 'gibbs',
	'gibbs': {
		'ini_iter': 10, # initialize persistent Markov chain before training
		'num_iter': 1 # number of iterations in each E-step
	},
	'callback': callback})

# gives you a list of all available parameters
parameters = isa.default_parameters()
```

## Reference

L. Theis, J. Sohl-Dickstein, and M. Bethge, *Training sparse natural image models with a fast Gibbs
sampler of an extended state space*, Advances in Neural Information Processing Systems 25, 2012
