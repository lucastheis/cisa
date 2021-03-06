#include "isainterface.h"
#include "exception.h"
#include "pyutils.h"
#include "Eigen/Core"
#include "callbacktrain.h"
#include "gsminterface.h"
#include <iostream>

using namespace Eigen;

const char* ISA_doc =
	"An implementation of the probabilistic model underlying overcomplete independent\n"
	"subspace analysis.\n"
	"\n"
	"B{References:}\n"
	"\t- L. Theis, J. Sohl-Dickstein, and M. Bethge, I{Training sparse natural image\n"
	"\tmodels with a fast Gibbs sampler of an extended state space}, NIPS 2012.";

ISA::Parameters PyObject_ToParameters(ISAObject* self, PyObject* parameters) {
	ISA::Parameters params;

	// read parameters from dictionary
	if(parameters) {
		if(!PyDict_Check(parameters))
			throw Exception("Parameters should be stored in a dictionary.");

		PyObject* verbosity = PyDict_GetItemString(parameters, "verbosity");
		if(verbosity)
			if(PyInt_Check(verbosity))
				params.verbosity = PyInt_AsLong(verbosity);
			else
				throw Exception("verbosity should be of type `int`.");

		PyObject* training_method = PyDict_GetItemString(parameters, "training_method");
		if(training_method)
			if(PyString_Check(training_method))
				params.trainingMethod = PyString_AsString(training_method);
			else
				throw Exception("training_method should be of type `string`.");

		PyObject* sampling_method = PyDict_GetItemString(parameters, "sampling_method");
		if(sampling_method)
			if(PyString_Check(sampling_method))
				params.samplingMethod = PyString_AsString(sampling_method);
			else
				throw Exception("sampling_method should be of type `string`.");

		PyObject* max_iter = PyDict_GetItemString(parameters, "max_iter");
		if(max_iter)
			if(PyInt_Check(max_iter))
				params.maxIter = PyInt_AsLong(max_iter);
			else if(PyFloat_Check(max_iter))
				params.maxIter = static_cast<int>(PyFloat_AsDouble(max_iter));
			else
				throw Exception("max_iter should be of type `int`.");

		PyObject* adaptive = PyDict_GetItemString(parameters, "adaptive");
		if(adaptive)
			if(PyBool_Check(adaptive))
				params.adaptive = (adaptive == Py_True);
			else
				throw Exception("adaptive should be of type `bool`.");

		PyObject* merge_subspaces = PyDict_GetItemString(parameters, "merge_subspaces");
		if(merge_subspaces)
			if(PyBool_Check(merge_subspaces))
				params.mergeSubspaces = (merge_subspaces == Py_True);
			else
				throw Exception("merge_subspaces should be of type `bool`.");

		PyObject* train_prior = PyDict_GetItemString(parameters, "train_prior");
		if(train_prior)
			if(PyBool_Check(train_prior))
				params.trainPrior = (train_prior == Py_True);
			else
				throw Exception("train_prior should be of type `bool`.");

		PyObject* train_basis = PyDict_GetItemString(parameters, "train_basis");
		if(train_basis)
			if(PyBool_Check(train_basis))
				params.trainBasis = (train_basis == Py_True);
			else
				throw Exception("train_basis should be of type `bool`.");

		PyObject* orthogonalize = PyDict_GetItemString(parameters, "orthogonalize");
		if(orthogonalize)
			if(PyBool_Check(orthogonalize))
				params.orthogonalize = (orthogonalize == Py_True);
			else
				throw Exception("orthogonalize should be of type `bool`.");

		PyObject* callback = PyDict_GetItemString(parameters, "callback");
		if(callback)
			if(PyCallable_Check(callback))
				params.callback = new CallbackTrain(self, callback);
			else if(callback != Py_None)
				throw Exception("callback should be a function or callable object.");

		PyObject* sgd = PyDict_GetItemString(parameters, "sgd");

		if(!sgd)
			sgd = PyDict_GetItemString(parameters, "SGD");

		if(sgd && PyDict_Check(sgd)) {
			PyObject* max_iter = PyDict_GetItemString(sgd, "max_iter");
			if(max_iter)
				if(PyInt_Check(max_iter))
					params.sgd.maxIter = PyInt_AsLong(max_iter);
				else if(PyFloat_Check(max_iter))
					params.sgd.maxIter = static_cast<int>(PyFloat_AsDouble(max_iter));
				else
					throw Exception("sgd.max_iter should be of type `int`.");

			PyObject* batch_size = PyDict_GetItemString(sgd, "batch_size");
			if(batch_size)
				if(PyInt_Check(batch_size))
					params.sgd.batchSize = PyInt_AsLong(batch_size);
				else if(PyFloat_Check(batch_size))
					params.sgd.batchSize = static_cast<int>(PyFloat_AsDouble(batch_size));
				else
					throw Exception("sgd.batch_size should be of type `int`.");

			PyObject* step_width = PyDict_GetItemString(sgd, "step_width");
			if(step_width)
				if(PyFloat_Check(step_width))
					params.sgd.stepWidth = PyFloat_AsDouble(step_width);
				else if(PyInt_Check(step_width))
					params.sgd.stepWidth = static_cast<double>(PyInt_AsLong(step_width));
				else
					throw Exception("sgd.step_width should be of type `float`.");

			PyObject* momentum = PyDict_GetItemString(sgd, "momentum");
			if(momentum)
				if(PyFloat_Check(momentum))
					params.sgd.momentum = PyFloat_AsDouble(momentum);
				else if(PyInt_Check(momentum))
					params.sgd.momentum = static_cast<double>(PyInt_AsLong(momentum));
				else
					throw Exception("sgd.momentum should be of type `float`.");

			PyObject* shuffle = PyDict_GetItemString(sgd, "shuffle");
			if(shuffle)
				if(PyBool_Check(shuffle))
					params.sgd.shuffle = (shuffle == Py_True);
				else
					throw Exception("sgd.shuffle should be of type `bool`.");

			PyObject* pocket = PyDict_GetItemString(sgd, "pocket");
			if(pocket)
				if(PyBool_Check(pocket))
					params.sgd.pocket = (pocket == Py_True);
				else
					throw Exception("sgd.pocket should be of type `bool`.");
		}

		PyObject* lbfgs = PyDict_GetItemString(parameters, "lbfgs");

		if(!lbfgs)
			lbfgs = PyDict_GetItemString(parameters, "LBFGS");

		if(lbfgs && PyDict_Check(lbfgs)) {
			PyObject* max_iter = PyDict_GetItemString(lbfgs, "max_iter");
			if(max_iter)
				if(PyInt_Check(max_iter))
					params.lbfgs.maxIter = PyInt_AsLong(max_iter);
				else if(PyFloat_Check(max_iter))
					params.lbfgs.maxIter = static_cast<int>(PyFloat_AsDouble(max_iter));
				else
					throw Exception("lbfgs.max_iter should be of type `int`.");

			PyObject* num_grad = PyDict_GetItemString(lbfgs, "num_grad");
			if(num_grad)
				if(PyInt_Check(num_grad))
					params.lbfgs.numGrad = PyInt_AsLong(num_grad);
				else if(PyFloat_Check(num_grad))
					params.lbfgs.numGrad = static_cast<int>(PyFloat_AsDouble(num_grad));
				else
					throw Exception("lbfgs.num_grad should be of type `int`.");
		}

		PyObject* mp = PyDict_GetItemString(parameters, "MP");

		if(!mp)
			mp = PyDict_GetItemString(parameters, "mp");

		if(mp && PyDict_Check(mp)) {
 			PyObject* max_iter = PyDict_GetItemString(mp, "max_iter");
 			if(max_iter)
 				if(PyInt_Check(max_iter))
 					params.mp.maxIter = PyInt_AsLong(max_iter);
 				else if(PyFloat_Check(max_iter))
 					params.mp.maxIter = static_cast<int>(PyFloat_AsDouble(max_iter));
 				else
 					throw Exception("mp.max_iter should be of type `int`.");

			PyObject* batch_size = PyDict_GetItemString(mp, "batch_size");
			if(batch_size)
				if(PyInt_Check(batch_size))
					params.mp.batchSize = PyInt_AsLong(batch_size);
				else if(PyFloat_Check(batch_size))
					params.mp.batchSize = static_cast<int>(PyFloat_AsDouble(batch_size));
				else
					throw Exception("mp.batch_size should be of type `int`.");

			PyObject* step_width = PyDict_GetItemString(mp, "step_width");
			if(step_width)
				if(PyFloat_Check(step_width))
					params.mp.stepWidth = PyFloat_AsDouble(step_width);
				else if(PyInt_Check(step_width))
					params.mp.stepWidth = static_cast<double>(PyInt_AsLong(step_width));
				else
					throw Exception("mp.step_width should be of type `float`.");

			PyObject* momentum = PyDict_GetItemString(mp, "momentum");
			if(momentum)
				if(PyFloat_Check(momentum))
					params.mp.momentum = PyFloat_AsDouble(momentum);
				else if(PyInt_Check(momentum))
					params.mp.momentum = static_cast<double>(PyInt_AsLong(momentum));
				else
					throw Exception("mp.momentum should be of type `float`.");

			PyObject* num_coeff = PyDict_GetItemString(mp, "num_coeff");
			if(num_coeff)
				if(PyInt_Check(num_coeff))
					params.mp.numCoeff = PyInt_AsLong(num_coeff);
				else if(PyFloat_Check(num_coeff))
					params.mp.numCoeff = static_cast<int>(PyFloat_AsDouble(num_coeff));
				else
					throw Exception("mp.num_coeff should be of type `int`.");
		}

		PyObject* gsm = PyDict_GetItemString(parameters, "gsm");

		if(!gsm)
			gsm = PyDict_GetItemString(parameters, "GSM");

		if(gsm && PyDict_Check(gsm)) {
			PyObject* max_iter = PyDict_GetItemString(gsm, "max_iter");
			if(max_iter)
				if(PyInt_Check(max_iter))
					params.gsm.maxIter = PyInt_AsLong(max_iter);
				else if(PyFloat_Check(max_iter))
					params.gsm.maxIter = static_cast<int>(PyFloat_AsDouble(max_iter));
				else
					throw Exception("gsm.max_iter should be of type `int`.");

			PyObject* tol = PyDict_GetItemString(gsm, "tol");
			if(tol)
				if(PyFloat_Check(tol))
					params.gsm.tol = PyFloat_AsDouble(tol);
				else if(PyInt_Check(tol))
					params.gsm.tol = static_cast<double>(PyInt_AsLong(tol));
				else
					throw Exception("gsm.tol should be of type `float`.");
		}

		PyObject* gibbs = PyDict_GetItemString(parameters, "gibbs");

		if(!gibbs)
			gibbs = PyDict_GetItemString(parameters, "Gibbs");

		if(!gibbs)
			gibbs = PyDict_GetItemString(parameters, "GIBBS");

		if(gibbs && PyDict_Check(gibbs)) {
			PyObject* verbosity = PyDict_GetItemString(gibbs, "verbosity");
			if(verbosity)
				if(PyInt_Check(verbosity))
					params.gibbs.verbosity = PyInt_AsLong(verbosity);
				else
					throw Exception("gibbs.verbosity should be of type `int`.");

			PyObject* ini_iter = PyDict_GetItemString(gibbs, "ini_iter");
			if(ini_iter)
				if(PyInt_Check(ini_iter))
					params.gibbs.iniIter = PyInt_AsLong(ini_iter);
				else
					throw Exception("gibbs.ini_iter should be of type `int`.");

			if(PyDict_GetItemString(gibbs, "num_steps"))
				throw Exception("No parameter gibbs.num_steps. Did you mean gibbs.num_iter?");

			PyObject* num_iter = PyDict_GetItemString(gibbs, "num_iter");
			if(num_iter)
				if(PyInt_Check(num_iter))
					params.gibbs.numIter = PyInt_AsLong(num_iter);
				else
					throw Exception("gibbs.num_iter should be of type `int`.");
		}

		PyObject* ais = PyDict_GetItemString(parameters, "ais");

		if(!ais)
			ais = PyDict_GetItemString(parameters, "AIS");

		if(ais && PyDict_Check(ais)) {
			PyObject* verbosity = PyDict_GetItemString(ais, "verbosity");
			if(verbosity)
				if(PyInt_Check(verbosity))
					params.ais.verbosity = PyInt_AsLong(verbosity);
				else
					throw Exception("ais.verbosity should be of type `int`.");

			PyObject* num_iter = PyDict_GetItemString(ais, "num_iter");
			if(num_iter)
				if(PyInt_Check(num_iter))
					params.ais.numIter = PyInt_AsLong(num_iter);
				else
					throw Exception("ais.num_iter should be of type `int`.");

			PyObject* num_samples = PyDict_GetItemString(ais, "num_samples");
			if(num_samples)
				if(PyInt_Check(num_samples))
					params.ais.numSamples = PyInt_AsLong(num_samples);
				else
					throw Exception("ais.num_samples should be of type `int`.");
		}

		PyObject* merge = PyDict_GetItemString(parameters, "merge");

		if(!merge)
			merge = PyDict_GetItemString(parameters, "MERGE");


		if(merge && PyDict_Check(merge)) {
			PyObject* verbosity = PyDict_GetItemString(merge, "verbosity");
			if(verbosity)
				if(PyInt_Check(verbosity))
					params.merge.verbosity = PyInt_AsLong(verbosity);
				else
					throw Exception("merge.verbosity should be of type `int`.");

			PyObject* max_merge = PyDict_GetItemString(merge, "max_merge");
			if(max_merge)
				if(PyInt_Check(max_merge))
					params.merge.maxMerge = PyInt_AsLong(max_merge);
				else
					throw Exception("merge.max_merge should be of type `int`.");

			PyObject* max_iter = PyDict_GetItemString(merge, "max_iter");
			if(max_iter)
				if(PyInt_Check(max_iter))
					params.merge.maxIter = PyInt_AsLong(max_iter);
				else
					throw Exception("merge.max_iter should be of type `int`.");

			PyObject* threshold = PyDict_GetItemString(merge, "threshold");
			if(threshold)
				if(PyFloat_Check(threshold))
					params.merge.threshold = PyFloat_AsDouble(threshold);
				else if(PyInt_Check(threshold))
					params.merge.threshold = static_cast<double>(PyInt_AsLong(threshold));
				else
					throw Exception("merge.threshold should be of type `float`.");
		}
	}

	return params;
}



PyObject* ISA_new(PyTypeObject* type, PyObject*, PyObject*) {
	PyObject* self = type->tp_alloc(type, 0);

	if(self)
		reinterpret_cast<ISAObject*>(self)->isa = 0;

	return self;
}



int ISA_init(ISAObject* self, PyObject* args, PyObject* kwds) {
	const char* kwlist[] = {"num_visibles", "num_hiddens", "ssize", "num_scales", 0};
	int num_visibles;
	int num_hiddens = -1;
	int ssize = 1;
	int num_scales = 10;

	// read arguments
	if(!PyArg_ParseTupleAndKeywords(args, kwds, "i|iii", const_cast<char**>(kwlist),
		&num_visibles, &num_hiddens, &ssize, &num_scales))
		return -1;

	// create actual ISA instance
	self->isa = new ISA(num_visibles, num_hiddens, ssize, num_scales);

	return 0;
}



void ISA_dealloc(ISAObject* self) {
	// delete actual ISA instance
	delete self->isa;

	// delete ISA object
	self->ob_type->tp_free(reinterpret_cast<PyObject*>(self));
}



PyObject* ISA_dim(ISAObject* self, PyObject*, void*) {
	return PyInt_FromLong(self->isa->numVisibles());
}



PyObject* ISA_num_visibles(ISAObject* self, PyObject*, void*) {
	return PyInt_FromLong(self->isa->numVisibles());
}



PyObject* ISA_num_hiddens(ISAObject* self, PyObject*, void*) {
	return PyInt_FromLong(self->isa->numHiddens());
}



PyObject* ISA_A(ISAObject* self, PyObject*, void*) {
	PyObject* array = PyArray_FromMatrixXd(self->isa->basis());

	// make array immutable
	reinterpret_cast<PyArrayObject*>(array)->flags &= ~NPY_WRITEABLE;

	return array;
}



int ISA_set_A(ISAObject* self, PyObject* value, void*) {
	if(!PyArray_Check(value)) {
		PyErr_SetString(PyExc_TypeError, "Basis should be of type `ndarray`.");
		return -1;
	}

	try {
		self->isa->setBasis(PyArray_ToMatrixXd(value));

	} catch(Exception exception) {
		PyErr_SetString(PyExc_RuntimeError, exception.message());
		return -1;
	}

	return 0;
}



const char* ISA_basis_doc =
	"Returns the current basis of the model. Each column corresponds to one basis\n"
	"vector and one hidden unit.\n"
	"\n"
	"@rtype: C{ndarray}\n"
	"@return: the current basis";

PyObject* ISA_basis(ISAObject* self, PyObject*, PyObject*) {
	try {
		return PyArray_FromMatrixXd(self->isa->basis());

	} catch(Exception exception) {
		PyErr_SetString(PyExc_RuntimeError, exception.message());
		return 0;
	}

	return 0;
}


const char* ISA_set_basis_doc =
	"Replaces the basis vectors of the model. The number of columns should correspond\n"
	"to the number of hidden units, the number of rows to the number of visible\n"
	"units.\n"
	"\n"
	"@type  basis: C{ndarray}\n"
	"@param basis: the basis vectors stored in columns";

PyObject* ISA_set_basis(ISAObject* self, PyObject* args, PyObject* kwds) {
	const char* kwlist[] = {"basis", 0};

	PyObject* basis = 0;

	// read arguments
	if(!PyArg_ParseTupleAndKeywords(args, kwds, "O", const_cast<char**>(kwlist), &basis))
		return 0;

	if(!PyArray_Check(basis)) {
		PyErr_SetString(PyExc_TypeError, "Basis should be of type `ndarray`.");
		return 0;
	}

	try {
		self->isa->setBasis(PyArray_ToMatrixXd(basis));

	} catch(Exception exception) {
		PyErr_SetString(PyExc_RuntimeError, exception.message());
		return 0;
	}

	Py_INCREF(Py_None);
	return Py_None;
}



const char* ISA_nullspace_basis_doc =
	"Computes a basis spanning the nullspace of the basis matrix. That is, the column\n"
	"vectors of the returned matrix will be orthogonal to the basis vectors.\n"
	"\n"
	"@rtype: C{ndarray}\n"
	"@return: a basis for the nullspace of the basis matrix";

PyObject* ISA_nullspace_basis(ISAObject* self, PyObject* args, PyObject* kwds) {
	try {
		return PyArray_FromMatrixXd(self->isa->nullspaceBasis());

	} catch(Exception exception) {
		PyErr_SetString(PyExc_RuntimeError, exception.message());
		return 0;
	}

	return 0;
}


const char* ISA_hidden_states_doc =
	"Returns the current state of the persistent Markov chain used for training. The\n"
	"number of columns of the returned matrix corresponds to the number of data points\n"
	"used during the last training run with persistent EM. The number of rows corresponds\n"
	"to the number of hidden units.\n"
	"\n"
	"@rtype: C{ndarray}\n"
	"@return: the states of the persistent Markov chain";

PyObject* ISA_hidden_states(ISAObject* self, PyObject*, PyObject*) {
	try {
		return PyArray_FromMatrixXd(self->isa->hiddenStates());

	} catch(Exception exception) {
		PyErr_SetString(PyExc_RuntimeError, exception.message());
		return 0;
	}

	return 0;
}



const char* ISA_set_hidden_states_doc =
	"Can be used to set the state of the persistent Markov chain. The number of columns\n"
	"should correspond to the number of data points that will be used for training, the.\n"
	"number of rows should correspond to the number of hidden units.\n"
	"\n"
	"@type  states: C{ndarray}\n"
	"@param states: new states for the persistent Markov chain over hidden units";

PyObject* ISA_set_hidden_states(ISAObject* self, PyObject* args, PyObject* kwds) {
	const char* kwlist[] = {"states", 0};

	PyObject* states = 0;

	// read arguments
	if(!PyArg_ParseTupleAndKeywords(args, kwds, "O", const_cast<char**>(kwlist), &states))
		return 0;

	if(!PyArray_Check(states)) {
		PyErr_SetString(PyExc_TypeError, "Hidden states should be of type `ndarray`.");
		return 0;
	}

	try {
		self->isa->setHiddenStates(PyArray_ToMatrixXd(states));

	} catch(Exception exception) {
		PyErr_SetString(PyExc_RuntimeError, exception.message());
		return 0;
	}

	Py_INCREF(Py_None);
	return Py_None;
}



const char* ISA_subspaces_doc =
	"Returns a list of L{GSM} objects which model the distributions over hidden units\n"
	"within each subspace.\n"
	"\n"
	"rtype: C{list}\n"
	"return: a list of Gaussian scale mixture distributions";

PyObject* ISA_subspaces(ISAObject* self, PyObject*, PyObject*) {
	vector<GSM> subspaces = self->isa->subspaces();

	PyObject* list = PyList_New(subspaces.size());

 	for(unsigned int i = 0; i < subspaces.size(); ++i) {
 		// create Python object representing GSM
 		PyObject* gsmObj = _PyObject_New(&GSM_type);
 		reinterpret_cast<GSMObject*>(gsmObj)->gsm = new GSM(subspaces[i]);
 
 		PyList_SetItem(list, i, gsmObj);
 	}

	return list;
}



const char* ISA_set_subspaces_doc =
	"Can be used to modify the distribution over hidden units. The given list should\n"
	"contain one L{GSM} for each subspace. The dimensionality of each subspace can be\n"
	"chosen arbitrarily, but the dimensionalities of all subspaces should add up to\n"
	"the number of hidden units.\n"
	"\n"
	"type  subspaces: C{list}\n"
	"param subspaces: a list of Gaussian scale mixture distributions";

PyObject* ISA_set_subspaces(ISAObject* self, PyObject* args, PyObject* kwds) {
	const char* kwlist[] = {"subspaces", 0};

	PyObject* list = 0;

	// read arguments
	if(!PyArg_ParseTupleAndKeywords(args, kwds, "O", const_cast<char**>(kwlist), &list))
		return 0;

	if(!PyList_Check(list)) {
		PyErr_SetString(PyExc_TypeError, "Subspace GSMs should be stored in a list.");
		return 0;
	}

	try {
		vector<GSM> subspaces;

		for(Py_ssize_t i = 0; i < PyList_Size(list); ++i) {
 			PyObject* gsmObj = PyList_GetItem(list, i);
 
 			if(!PyObject_IsInstance(gsmObj, reinterpret_cast<PyObject*>(&GSM_type))) {
 				PyErr_SetString(PyExc_TypeError, "Subspaces should be modeled by GSMs.");
 				return 0;
 			}

			subspaces.push_back(*reinterpret_cast<GSMObject*>(gsmObj)->gsm);
		}

		self->isa->setSubspaces(subspaces);

	} catch(Exception exception) {
		PyErr_SetString(PyExc_TypeError, exception.message());
		return 0;
	}

	Py_INCREF(Py_None);
	return Py_None;
}


const char* ISA_default_parameters_doc =
	"Returns a dictionary of default parameters.\n"
	"\n"
	"@rtype: C{dict}\n"
	"@return: default parameters";

PyObject* ISA_default_parameters(ISAObject* self) {
	ISA::Parameters params;
	PyObject* parameters = PyDict_New();
	PyObject* sgd = PyDict_New();
	PyObject* lbfgs = PyDict_New();
	PyObject* mp = PyDict_New();
	PyObject* gsm = PyDict_New();
	PyObject* gibbs = PyDict_New();
	PyObject* ais = PyDict_New();
	PyObject* merge = PyDict_New();

	PyDict_SetItemString(parameters, "verbosity", PyInt_FromLong(params.verbosity));
	PyDict_SetItemString(parameters, "training_method",
		PyString_FromString(params.trainingMethod.c_str()));
	PyDict_SetItemString(parameters, "sampling_method",
		PyString_FromString(params.samplingMethod.c_str()));
	PyDict_SetItemString(parameters, "max_iter", PyInt_FromLong(params.maxIter));
	PyDict_SetItemString(parameters, "callback", Py_None);
	Py_INCREF(Py_None);

	if(params.adaptive) {
		PyDict_SetItemString(parameters, "adaptive", Py_True);
		Py_INCREF(Py_True);
	} else {
		PyDict_SetItemString(parameters, "adaptive", Py_False);
		Py_INCREF(Py_False);
	}

	if(params.trainBasis) {
		PyDict_SetItemString(parameters, "train_basis", Py_True);
		Py_INCREF(Py_True);
	} else {
		PyDict_SetItemString(parameters, "train_basis", Py_False);
		Py_INCREF(Py_False);
	}

	if(params.trainPrior) {
		PyDict_SetItemString(parameters, "train_prior", Py_True);
		Py_INCREF(Py_True);
	} else {
		PyDict_SetItemString(parameters, "train_prior", Py_False);
		Py_INCREF(Py_False);
	}

	if(params.mergeSubspaces) {
		PyDict_SetItemString(parameters, "merge_subspaces", Py_True);
		Py_INCREF(Py_True);
	} else {
		PyDict_SetItemString(parameters, "merge_subspaces", Py_False);
		Py_INCREF(Py_False);
	}

	if(params.orthogonalize) {
		PyDict_SetItemString(parameters, "orthogonalize", Py_True);
		Py_INCREF(Py_True);
	} else {
		PyDict_SetItemString(parameters, "orthogonalize", Py_False);
		Py_INCREF(Py_False);
	}

	PyDict_SetItemString(sgd, "max_iter", PyInt_FromLong(params.sgd.maxIter));
	PyDict_SetItemString(sgd, "batch_size", PyInt_FromLong(params.sgd.batchSize));
	PyDict_SetItemString(sgd, "step_width", PyFloat_FromDouble(params.sgd.stepWidth));
	PyDict_SetItemString(sgd, "momentum", PyFloat_FromDouble(params.sgd.momentum));

	if(params.sgd.shuffle) {
		PyDict_SetItemString(sgd, "shuffle", Py_True);
		Py_INCREF(Py_True);
	} else {
		PyDict_SetItemString(sgd, "shuffle", Py_False);
		Py_INCREF(Py_False);
	}

	if(params.sgd.pocket) {
		PyDict_SetItemString(sgd, "pocket", Py_True);
		Py_INCREF(Py_True);
	} else {
		PyDict_SetItemString(sgd, "pocket", Py_False);
		Py_INCREF(Py_False);
	}

	PyDict_SetItemString(lbfgs, "max_iter", PyInt_FromLong(params.lbfgs.maxIter));
	PyDict_SetItemString(lbfgs, "num_grad", PyInt_FromLong(params.lbfgs.numGrad));

	PyDict_SetItemString(mp, "max_iter", PyInt_FromLong(params.mp.maxIter));
	PyDict_SetItemString(mp, "batch_size", PyInt_FromLong(params.mp.batchSize));
	PyDict_SetItemString(mp, "step_width", PyFloat_FromDouble(params.mp.stepWidth));
	PyDict_SetItemString(mp, "momentum", PyFloat_FromDouble(params.mp.momentum));
	PyDict_SetItemString(mp, "num_coeff", PyInt_FromLong(params.mp.numCoeff));

	PyDict_SetItemString(gsm, "max_iter", PyInt_FromLong(params.gsm.maxIter));
	PyDict_SetItemString(gsm, "tol", PyFloat_FromDouble(params.gsm.tol));

	PyDict_SetItemString(gibbs, "verbosity", PyInt_FromLong(params.gibbs.verbosity));
	PyDict_SetItemString(gibbs, "ini_iter", PyInt_FromLong(params.gibbs.iniIter));
	PyDict_SetItemString(gibbs, "num_iter", PyInt_FromLong(params.gibbs.numIter));

	PyDict_SetItemString(ais, "verbosity", PyInt_FromLong(params.ais.verbosity));
	PyDict_SetItemString(ais, "num_iter", PyInt_FromLong(params.ais.numIter));
	PyDict_SetItemString(ais, "num_samples", PyInt_FromLong(params.ais.numSamples));

	PyDict_SetItemString(merge, "verbosity", PyInt_FromLong(params.merge.verbosity));
	PyDict_SetItemString(merge, "max_merge", PyInt_FromLong(params.merge.maxMerge));
	PyDict_SetItemString(merge, "max_iter", PyInt_FromLong(params.merge.maxIter));
	PyDict_SetItemString(merge, "threhold", PyFloat_FromDouble(params.merge.threshold));

	PyDict_SetItemString(parameters, "sgd", sgd);
	PyDict_SetItemString(parameters, "lbfgs", lbfgs);
	PyDict_SetItemString(parameters, "mp", mp);
	PyDict_SetItemString(parameters, "gsm", gsm);
	PyDict_SetItemString(parameters, "gibbs", gibbs);
	PyDict_SetItemString(parameters, "ais", ais);
	PyDict_SetItemString(parameters, "merge", merge);

	Py_DECREF(sgd);
	Py_DECREF(lbfgs);
	Py_DECREF(mp);
	Py_DECREF(gsm);
	Py_DECREF(gibbs);
	Py_DECREF(ais);
	Py_DECREF(merge);

	return parameters;
}



const char* ISA_initialize_doc =
	"Initializes the parameters of the model. The distributions over hidden units\n"
	"are initialized to approximate the Laplace distribution if the subspaces are\n"
	"one-dimensional. If data points are given, the basis vectors are additionally\n"
	"using a heuristic.\n"
	"\n"
	"@type  data: C{ndarray}\n"
	"@param data: a set of data points (optional)";

PyObject* ISA_initialize(ISAObject* self, PyObject* args, PyObject* kwds) {
	const char* kwlist[] = {"data", 0};

	PyObject* data = 0;

	// read arguments
	if(!PyArg_ParseTupleAndKeywords(args, kwds, "|O", const_cast<char**>(kwlist), &data))
		return 0;

	// make sure data is stored in NumPy array
	if(data && !PyArray_Check(data)) {
		PyErr_SetString(PyExc_TypeError, "Data has to be stored in a NumPy array.");
		return 0;
	}

	try {
		self->isa->initialize();
		if(data)
			self->isa->initialize(PyArray_ToMatrixXd(data));
	} catch(Exception exception) {
		PyErr_SetString(PyExc_RuntimeError, exception.message());
		return 0;
	}

	Py_INCREF(Py_None);
	return Py_None;
}



const char* ISA_orthogonalize_doc =
	"Symmetrically orthogonalizes the basis vectors.\n";

PyObject* ISA_orthogonalize(ISAObject* self, PyObject* args, PyObject* kwds) {
	try {
		self->isa->orthogonalize();
	} catch(Exception exception) {
		PyErr_SetString(PyExc_RuntimeError, exception.message());
		return 0;
	}

	Py_INCREF(Py_None);
	return Py_None;
}



const char* ISA_train_doc =
	"Trains the parameters of the model.\n"
	"\n"
	"By default, the model will be trained using a Monte Carlo variant of expectation\n"
	"maximization with either stochastic gradient descent (SGD) or limited-memory BFGS\n"
	"(LBFGS) in each M-step. If C{train_prior} is C{True}, also the distributions over\n"
	"hidden units will be adjusted. Alternatively, C{matching_pursuit} can be used to\n"
	"optimize the basis vectors.\n"
	"\n"
	"Which method is used is determined by the C{training_method} entry of the dictionary\n"
	"C{parameters} (either 'MP', 'SGD' or 'LBFGS').\n"
	"\n"
	"@type  data: C{ndarray}\n"
	"@param data: data points stored in columns\n"
	"\n"
	"@type  parameters: C{dict}\n"
	"@param parameters: parameters controlling the training method (optional)";

PyObject* ISA_train(ISAObject* self, PyObject* args, PyObject* kwds) {
	const char* kwlist[] = {"data", "parameters", 0};

	PyObject* data;
	PyObject* parameters = 0;

	// read arguments
	if(!PyArg_ParseTupleAndKeywords(args, kwds, "O|O", const_cast<char**>(kwlist), &data, &parameters))
		return 0;

	data = PyArray_FROM_OTF(data, NPY_DOUBLE, NPY_F_CONTIGUOUS | NPY_ALIGNED);

	// make sure data is stored in NumPy array
	if(!data) {
		PyErr_SetString(PyExc_TypeError, "Data has to be stored in a NumPy array.");
		return 0;
	}

	try {
		ISA::Parameters params = PyObject_ToParameters(self, parameters);
			
		// fit model to training data
		self->isa->train(PyArray_ToMatrixXd(data), params);
	} catch(Exception exception) {
		PyErr_SetString(PyExc_RuntimeError, exception.message());
		Py_DECREF(data);
		return 0;
	}

	Py_DECREF(data);
	Py_INCREF(Py_None);
	return Py_None;
}



const char* ISA_sample_doc =
	"Draws samples from the model.\n"
	"\n"
	"@type  num_samples: C{int}\n"
	"@param num_samples: the number of samples to draw\n"
	"\n"
	"@rtype: C{ndarray}\n"
	"@return: samples from the model";

PyObject* ISA_sample(ISAObject* self, PyObject* args, PyObject* kwds) {
	const char* kwlist[] = {"num_samples", 0};

	int num_samples = 1;

	if(!PyArg_ParseTupleAndKeywords(args, kwds, "|i", const_cast<char**>(kwlist), &num_samples))
		return 0;

	try {
		return PyArray_FromMatrixXd(self->isa->sample(num_samples));
	} catch(Exception exception) {
		PyErr_SetString(PyExc_RuntimeError, exception.message());
		return 0;
	}

	return 0;
}



const char* ISA_sample_prior_doc =
	"Draws samples from the prior distribution over hidden units.\n"
	"\n"
	"@type  num_samples: C{int}\n"
	"@param num_samples: the number of samples to draw\n"
	"\n"
	"@rtype: C{ndarray}\n"
	"@return: samples from the prior over hidden units";

PyObject* ISA_sample_prior(ISAObject* self, PyObject* args, PyObject* kwds) {
	const char* kwlist[] = {"num_samples", 0};

	int num_samples = 1;

	if(!PyArg_ParseTupleAndKeywords(args, kwds, "|i", const_cast<char**>(kwlist), &num_samples))
		return 0;

	try {
 		return PyArray_FromMatrixXd(self->isa->samplePrior(num_samples));
	} catch(Exception exception) {
		PyErr_SetString(PyExc_RuntimeError, exception.message());
		return 0;
	}

	return 0;
}



const char* ISA_sample_nullspace_doc =
	"Draws samples from the posterior distribution over the nullspace representation\n"
	"of the hidden states using Gibbs sampling or some other method.\n"
	"\n"
	"@type  data: C{ndarray}\n"
	"@param data: states of the visible units\n"
	"\n"
	"@type  parameters: C{dict}\n"
	"@param parameters: parameters controlling the sampling method (optional)\n"
	"\n"
	"@rtype: C{ndarray}\n"
	"@return: samples from the posterior over nullspace states";

PyObject* ISA_sample_nullspace(ISAObject* self, PyObject* args, PyObject* kwds) {
	const char* kwlist[] = {"data", "parameters", 0};

	PyObject* data;
	PyObject* parameters = 0;

	// read arguments
	if(!PyArg_ParseTupleAndKeywords(args, kwds, "O|O", const_cast<char**>(kwlist), &data, &parameters))
		return 0;

	data = PyArray_FROM_OTF(data, NPY_DOUBLE, NPY_F_CONTIGUOUS | NPY_ALIGNED);

	// make sure data is stored in NumPy array
	if(!data) {
		PyErr_SetString(PyExc_TypeError, "Data has to be stored in a NumPy array.");
		return 0;
	}

	try {
		PyObject* samples = PyArray_FromMatrixXd(self->isa->sampleNullspace(
			PyArray_ToMatrixXd(data),
			PyObject_ToParameters(self, parameters)));
		Py_DECREF(data);
		return samples;
	} catch(Exception exception) {
		PyErr_SetString(PyExc_RuntimeError, exception.message());
		Py_DECREF(data);
		return 0;
	}

	Py_DECREF(data);
	return 0;
}


const char* ISA_sample_posterior_doc =
	"Draws samples from the posterior distribution over hidden units using Gibbs\n"
	"sampling or some other method. For each data point, one sample is generated.\n"
	"\n"
	"@type  data: C{ndarray}\n"
	"@param data: states of the visible units\n"
	"\n"
	"@type  parameters: C{dict}\n"
	"@param parameters: parameters controlling the sampling method (optional)\n"
	"\n"
	"@type  hidden_states: C{ndarray}\n"
	"@param hidden_states: initial states for the Markov chain of the sampler (optional)\n"
	"\n"
	"@rtype: C{ndarray}\n"
	"@return: samples from the posterior distribution over hidden units";

PyObject* ISA_sample_posterior(ISAObject* self, PyObject* args, PyObject* kwds) {
	const char* kwlist[] = {"data", "parameters", "hidden_states", 0};

	PyObject* data;
	PyObject* parameters = 0;
	PyObject* hidden_states = 0;

	// read arguments
	if(!PyArg_ParseTupleAndKeywords(args, kwds, "O|OO", const_cast<char**>(kwlist), &data, &parameters, &hidden_states))
		return 0;

	data = PyArray_FROM_OTF(data, NPY_DOUBLE, NPY_F_CONTIGUOUS | NPY_ALIGNED);

	// make sure data is stored in NumPy array
	if(!data) {
		PyErr_SetString(PyExc_TypeError, "Data has to be stored in a NumPy array.");
		return 0;
	}

	if(hidden_states) {
		hidden_states = PyArray_FROM_OTF(hidden_states, NPY_DOUBLE, NPY_F_CONTIGUOUS | NPY_ALIGNED);

		if(!hidden_states) {
			PyErr_SetString(PyExc_TypeError, "Hidden states have to be stored in a NumPy array.");
			Py_DECREF(data);
			return 0;
		}
	}

	try {
		PyObject* samples;
		if(hidden_states)
			samples = PyArray_FromMatrixXd(self->isa->samplePosterior(
				PyArray_ToMatrixXd(data),
				PyArray_ToMatrixXd(hidden_states),
				PyObject_ToParameters(self, parameters)));
		else
			samples = PyArray_FromMatrixXd(self->isa->samplePosterior(
				PyArray_ToMatrixXd(data),
				PyObject_ToParameters(self, parameters)));
		Py_DECREF(data);
		Py_XDECREF(hidden_states);
		return samples;
	} catch(Exception exception) {
		PyErr_SetString(PyExc_RuntimeError, exception.message());
		Py_DECREF(data);
		Py_XDECREF(hidden_states);
		return 0;
	}

	Py_DECREF(data);
	Py_XDECREF(hidden_states);
	return 0;
}



const char* ISA_sample_posterior_ais_doc =
	"Draws samples from the posterior and generates corresponding importance weights\n"
	"using annealed importance sampling (AIS).\n"
	"\n"
	"@type  data: C{ndarray}\n"
	"@param data: states of the visible units\n"
	"\n"
	"@type  parameters: C{dict}\n"
	"@param parameters: parameters controlling the sampling method (optional)\n"
	"\n"
	"@rtype: C{tuple}\n"
	"@return: samples and log importance weights";

PyObject* ISA_sample_posterior_ais(ISAObject* self, PyObject* args, PyObject* kwds) {
	const char* kwlist[] = {"data", "parameters", 0};

	PyObject* data;
	PyObject* parameters = 0;

	// read arguments
	if(!PyArg_ParseTupleAndKeywords(args, kwds, "O|O", const_cast<char**>(kwlist), &data, &parameters))
		return 0;

	// make sure data is stored in contiguous NumPy array
	data = PyArray_FROM_OTF(data, NPY_DOUBLE, NPY_F_CONTIGUOUS | NPY_ALIGNED);

	if(!data) {
		PyErr_SetString(PyExc_TypeError, "Data has to be stored in a NumPy array.");
		return 0;
	}

	try {
		ISA::Parameters params = PyObject_ToParameters(self, parameters);

		pair<MatrixXd, MatrixXd> result = self->isa->samplePosteriorAIS(PyArray_ToMatrixXd(data), params);

		PyObject* samples = PyArray_FromMatrixXd(result.first);
		PyObject* logWeights = PyArray_FromMatrixXd(result.second);

		PyObject* tuple = Py_BuildValue("(OO)", samples, logWeights);

		Py_DECREF(data);
		Py_DECREF(samples);
		Py_DECREF(logWeights);

		return tuple;
	} catch(Exception exception) {
		PyErr_SetString(PyExc_RuntimeError, exception.message());
		Py_DECREF(data);
		return 0;
	}

	Py_DECREF(data);
	return 0;
}



const char* ISA_sample_ais_doc =
	"Like L{sample_posterior_ais}, but only returns the logarithm of the importance\n"
	"weights and not the sampled hidden units.\n"
	"\n"
	"@type  data: C{ndarray}\n"
	"@param data: states of the visible units\n"
	"\n"
	"@type  parameters: C{dict}\n"
	"@param parameters: parameters controlling the sampling method (optional)\n"
	"\n"
	"@rtype: C{ndarray}\n"
	"@return: log importance weights";

PyObject* ISA_sample_ais(ISAObject* self, PyObject* args, PyObject* kwds) {
	const char* kwlist[] = {"data", "parameters", 0};

	PyObject* data;
	PyObject* parameters = 0;

	// read arguments
	if(!PyArg_ParseTupleAndKeywords(args, kwds, "O|O", const_cast<char**>(kwlist), &data, &parameters))
		return 0;

	// make sure data is stored in contiguous NumPy array
	data = PyArray_FROM_OTF(data, NPY_DOUBLE, NPY_F_CONTIGUOUS | NPY_ALIGNED);

	if(!data) {
		PyErr_SetString(PyExc_TypeError, "Data has to be stored in a NumPy array.");
		return 0;
	}

	try {
		PyObject* samples = PyArray_FromMatrixXd(self->isa->sampleAIS(
			PyArray_ToMatrixXd(data),
			PyObject_ToParameters(self, parameters)));
		Py_DECREF(data);
		return samples;
	} catch(Exception exception) {
		PyErr_SetString(PyExc_RuntimeError, exception.message());
		Py_DECREF(data);
		return 0;
	}

	return 0;
}



const char* ISA_sample_scales_doc =
	"Samples standard deviations from the posterior distribution of the Gaussian scale\n"
	"mixtures given states for the hidden units.\n"
	"\n"
	"@type  states: C{ndarray}\n"
	"@param states: states of the hidden units\n"
	"\n"
	"@rtype: C{ndarray}\n"
	"@return: standard deviations";

PyObject* ISA_sample_scales(ISAObject* self, PyObject* args, PyObject* kwds) {
	const char* kwlist[] = {"states", 0};

	PyObject* states;

	// read arguments
	if(!PyArg_ParseTupleAndKeywords(args, kwds, "O", const_cast<char**>(kwlist), &states))
		return 0;

	// make sure data is stored in NumPy array
	if(!PyArray_Check(states)) {
		PyErr_SetString(PyExc_TypeError, "Hidden states have to be stored in a NumPy array.");
		return 0;
	}

	try {
		return PyArray_FromMatrixXd(self->isa->sampleScales(PyArray_ToMatrixXd(states)));
	} catch(Exception exception) {
		PyErr_SetString(PyExc_RuntimeError, exception.message());
		return 0;
	}

	return 0;
}



const char* ISA_matching_pursuit_doc =
	"Tries to infer the state of the hidden units using matching pursuit. Here, the\n"
	"number of active coefficients is fixed and can be controlled by setting\n"
	"C{num_coeff} in the parameters.\n"
	"\n"
	"@type  data: C{ndarray}\n"
	"@param data: states of the visible units\n"
	"\n"
	"@type  parameters: C{dict}\n"
	"@param parameters: parameters controlling the number of active coefficients (optional)\n"
	"\n"
	"@rtype: C{ndarray}\n"
	"@return: inferred states of the hidden units";

PyObject* ISA_matching_pursuit(ISAObject* self, PyObject* args, PyObject* kwds) {
	const char* kwlist[] = {"data", "parameters", 0};

	PyObject* data;
	PyObject* parameters = 0;

	// read arguments
	if(!PyArg_ParseTupleAndKeywords(args, kwds, "O|O", const_cast<char**>(kwlist), &data, &parameters))
		return 0;

	// make sure data is stored in NumPy array
	if(!PyArray_Check(data)) {
		PyErr_SetString(PyExc_TypeError, "Data has to be stored in a NumPy array.");
		return 0;
	}

	try {
		return PyArray_FromMatrixXd(self->isa->matchingPursuit(
			PyArray_ToMatrixXd(data),
			PyObject_ToParameters(self, parameters)));
	} catch(Exception exception) {
		PyErr_SetString(PyExc_RuntimeError, exception.message());
		return 0;
	}

	return 0;
}


const char* ISA_prior_energy_doc =
	"Computes the negative logarithm of the unnormalized density of hidden states.\n"
	"\n"
	"@type  states: C{ndarray}\n"
	"@param states: states of the hidden units\n"
	"\n"
	"@rtype: C{ndarray}\n"
	"@return: energies of the hidden unit states";

PyObject* ISA_prior_energy(ISAObject* self, PyObject* args, PyObject* kwds) {
	const char* kwlist[] = {"states", 0};

	PyObject* states;

	// read arguments
	if(!PyArg_ParseTupleAndKeywords(args, kwds, "O", const_cast<char**>(kwlist), &states))
		return 0;

	// make sure data is stored in NumPy array
	if(!PyArray_Check(states)) {
		PyErr_SetString(PyExc_TypeError, "Hidden states have to be stored in a NumPy array.");
		return 0;
	}

	try {
		return PyArray_FromMatrixXd(self->isa->priorEnergy(PyArray_ToMatrixXd(states)));
	} catch(Exception exception) {
		PyErr_SetString(PyExc_RuntimeError, exception.message());
		return 0;
	}

	return 0;
}


const char* ISA_prior_energy_gradient_doc =
	"Computes the gradient of the energy of hidden states.\n"
	"\n"
	"@type  states: C{ndarray}\n"
	"@param states: states of the hidden units\n"
	"\n"
	"@rtype: C{ndarray}\n"
	"@return: energy gradients of the hidden unit states";

PyObject* ISA_prior_energy_gradient(ISAObject* self, PyObject* args, PyObject* kwds) {
	const char* kwlist[] = {"states", 0};

	PyObject* states;

	// read arguments
	if(!PyArg_ParseTupleAndKeywords(args, kwds, "O", const_cast<char**>(kwlist), &states))
		return 0;

	// make sure data is stored in NumPy array
	if(!PyArray_Check(states)) {
		PyErr_SetString(PyExc_TypeError, "Hidden states have to be stored in a NumPy array.");
		return 0;
	}

	try {
		return PyArray_FromMatrixXd(self->isa->priorEnergyGradient(PyArray_ToMatrixXd(states)));
	} catch(Exception exception) {
		PyErr_SetString(PyExc_RuntimeError, exception.message());
		return 0;
	}

	return 0;
}



const char* ISA_prior_loglikelihood_doc =
	"Computes the logarithm of the normalized density of hidden states.\n"
	"\n"
	"@type  states: C{ndarray}\n"
	"@param states: states of the hidden units\n"
	"\n"
	"@rtype: C{ndarray}\n"
	"@return: density evaluated at the given states for the hidden units";

PyObject* ISA_prior_loglikelihood(ISAObject* self, PyObject* args, PyObject* kwds) {
	const char* kwlist[] = {"states", 0};

	PyObject* states;

	// read arguments
	if(!PyArg_ParseTupleAndKeywords(args, kwds, "O", const_cast<char**>(kwlist), &states))
		return 0;

	// make sure data is stored in NumPy array
	if(!PyArray_Check(states)) {
		PyErr_SetString(PyExc_TypeError, "Hidden states have to be stored in a NumPy array.");
		return 0;
	}

	try {
		return PyArray_FromMatrixXd(self->isa->priorLogLikelihood(PyArray_ToMatrixXd(states)));
	} catch(Exception exception) {
		PyErr_SetString(PyExc_RuntimeError, exception.message());
		return 0;
	}

	return 0;
}



const char* ISA_loglikelihood_doc =
	"Estimates the density of data points under the model using annealed importance\n"
	"sampling (AIS). This estimator will tend to underestimate the log-likelihood if\n"
	"the parameters are not chosen well enough.\n"
	"\n"
	"If C{return_all} is C{True}, all importance weights are returned instead of averaging them.\n"
	"\n"
	"@type  data: C{ndarray}\n"
	"@param data: states of the visible units\n"
	"\n"
	"@type  parameters: C{dict}\n"
	"@param parameters: parameters controlling AIS (optional)\n"
	"\n"
	"@type  return_all: C{bool}\n"
	"@param return_all: return one estimate for each AIS sample (default: False)\n"
	"\n"
	"@rtype: C{ndarray}\n"
	"@return: natural logarithm of the estimated density of the given data points";

PyObject* ISA_loglikelihood(ISAObject* self, PyObject* args, PyObject* kwds) {
	const char* kwlist[] = {"data", "parameters", "return_all", 0};

	PyObject* data;
	PyObject* parameters = 0;
	int return_all = 0;

	// read arguments
	if(!PyArg_ParseTupleAndKeywords(args, kwds, "O|Oi", const_cast<char**>(kwlist), &data, &parameters, &return_all))
		return 0;

	// make sure data is stored in NumPy array
	if(!PyArray_Check(data)) {
		PyErr_SetString(PyExc_TypeError, "Data has to be stored in a NumPy array.");
		return 0;
	}

	try {
		if(!self->isa->complete() && return_all)
			return PyArray_FromMatrixXd(self->isa->sampleAIS(
				PyArray_ToMatrixXd(data),
				PyObject_ToParameters(self, parameters)));
		else
			return PyArray_FromMatrixXd(self->isa->logLikelihood(
				PyArray_ToMatrixXd(data),
				PyObject_ToParameters(self, parameters)));

	} catch(Exception exception) {
		PyErr_SetString(PyExc_RuntimeError, exception.message());
		return 0;
	}

	return 0;
}


const char* ISA_evaluate_doc =
	"Estimates the negative average log-likelihood in bits per component for a given\n"
	"set of data points using annealed importance sampling (AIS). This estimator\n"
	"will tend to underestimate the log-likelihood if the parameters are not chosen\n"
	"well enough.\n"
	"\n"
	"@type  data: C{ndarray}\n"
	"@param data: states of the visible units\n"
	"\n"
	"@type  parameters: C{dict}\n"
	"@param parameters: parameters controlling AIS (optional)\n"
	"\n"
	"@rtype: C{float}\n"
	"@return: the estimated negative average log-likelihood in bits per component";

PyObject* ISA_evaluate(ISAObject* self, PyObject* args, PyObject* kwds) {
	const char* kwlist[] = {"data", "parameters", 0};

	PyObject* data;
	PyObject* parameters = 0;

	// read arguments
	if(!PyArg_ParseTupleAndKeywords(args, kwds, "O|O", const_cast<char**>(kwlist), &data, &parameters))
		return 0;

	// make sure data is stored in NumPy array
	if(!PyArray_Check(data)) {
		PyErr_SetString(PyExc_TypeError, "Data has to be stored in a NumPy array.");
		return 0;
	}

	try {
		return PyFloat_FromDouble(self->isa->evaluate(
			PyArray_ToMatrixXd(data),
			PyObject_ToParameters(self, parameters)));
	} catch(Exception exception) {
		PyErr_SetString(PyExc_RuntimeError, exception.message());
		return 0;
	}

	return 0;
}



PyObject* ISA_reduce(ISAObject* self, PyObject*, PyObject*) {
	PyObject* args = Py_BuildValue("(ii)", self->isa->numVisibles(), self->isa->numHiddens());

	PyObject* basis = ISA_basis(self, 0, 0);
	PyObject* hidden_states = ISA_hidden_states(self, 0, 0);
	PyObject* subspaces = ISA_subspaces(self, 0, 0);
	PyObject* state = Py_BuildValue("(OOO)", basis, subspaces, hidden_states);
	Py_DECREF(basis);
	Py_DECREF(hidden_states);
	Py_DECREF(subspaces);

	PyObject* result = Py_BuildValue("OOO", self->ob_type, args, state);
	Py_DECREF(args);
	Py_DECREF(state);

	return result;
}



PyObject* ISA_setstate(ISAObject* self, PyObject* state, PyObject*) {
	PyObject* basis;
	PyObject* subspaces;
	PyObject* hidden_states;

	if(!PyArg_ParseTuple(state, "(OOO)", &basis, &subspaces, &hidden_states))
		return 0;

	PyObject* args;
	PyObject* kwds = PyDict_New();

	args = Py_BuildValue("(O)", basis);
	ISA_set_basis(self, args, kwds);
	Py_DECREF(args);

	args = Py_BuildValue("(O)", subspaces);
	ISA_set_subspaces(self, args, kwds);
	Py_DECREF(args);

	args = Py_BuildValue("(O)", hidden_states);
	ISA_set_hidden_states(self, args, kwds);
	Py_DECREF(args);

	Py_DECREF(kwds);

	Py_INCREF(Py_None);
	return Py_None;
}
