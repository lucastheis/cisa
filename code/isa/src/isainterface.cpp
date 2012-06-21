#include "isainterface.h"
#include "exception.h"
#include "pyutils.h"
#include "Eigen/Core"
#include "callbacktrain.h"
#include "gsminterface.h"
#include <iostream>

using namespace Eigen;

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
			PyObject* maxIter = PyDict_GetItemString(sgd, "max_iter");
			if(maxIter)
				if(PyInt_Check(maxIter))
					params.sgd.maxIter = PyInt_AsLong(maxIter);
				else if(PyFloat_Check(maxIter))
					params.sgd.maxIter = static_cast<int>(PyFloat_AsDouble(maxIter));
				else
					throw Exception("sgd.max_iter should be of type `int`.");

			PyObject* batchSize = PyDict_GetItemString(sgd, "batch_size");
			if(batchSize)
				if(PyInt_Check(batchSize))
					params.sgd.batchSize = PyInt_AsLong(batchSize);
				else if(PyFloat_Check(batchSize))
					params.sgd.batchSize = static_cast<int>(PyFloat_AsDouble(batchSize));
				else
					throw Exception("sgd.batch_size should be of type `int`.");

			PyObject* stepWidth = PyDict_GetItemString(sgd, "step_width");
			if(stepWidth)
				if(PyFloat_Check(stepWidth))
					params.sgd.stepWidth = PyFloat_AsDouble(stepWidth);
				else if(PyInt_Check(stepWidth))
					params.sgd.stepWidth = static_cast<double>(PyFloat_AsDouble(stepWidth));
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
			PyObject* maxIter = PyDict_GetItemString(lbfgs, "max_iter");
			if(maxIter)
				if(PyInt_Check(maxIter))
					params.lbfgs.maxIter = PyInt_AsLong(maxIter);
				else if(PyFloat_Check(maxIter))
					params.lbfgs.maxIter = static_cast<int>(PyFloat_AsDouble(maxIter));
				else
					throw Exception("lbfgs.max_iter should be of type `int`.");
		}

		PyObject* mp = PyDict_GetItemString(parameters, "MP");

		if(!mp)
			mp = PyDict_GetItemString(parameters, "mp");

		if(mp && PyDict_Check(mp)) {
			PyObject* maxIter = PyDict_GetItemString(mp, "max_iter");
			if(maxIter)
				if(PyInt_Check(maxIter))
					params.mp.maxIter = PyInt_AsLong(maxIter);
				else if(PyFloat_Check(maxIter))
					params.mp.maxIter = static_cast<int>(PyFloat_AsDouble(maxIter));
				else
					throw Exception("mp.max_iter should be of type `int`.");

			PyObject* batchSize = PyDict_GetItemString(mp, "batch_size");
			if(batchSize)
				if(PyInt_Check(batchSize))
					params.mp.batchSize = PyInt_AsLong(batchSize);
				else if(PyFloat_Check(batchSize))
					params.mp.batchSize = static_cast<int>(PyFloat_AsDouble(batchSize));
				else
					throw Exception("mp.batch_size should be of type `int`.");

			PyObject* stepWidth = PyDict_GetItemString(mp, "step_width");
			if(stepWidth)
				if(PyFloat_Check(stepWidth))
					params.mp.stepWidth = PyFloat_AsDouble(stepWidth);
				else if(PyInt_Check(stepWidth))
					params.mp.stepWidth = static_cast<double>(PyFloat_AsDouble(stepWidth));
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

			PyObject* numCoeff = PyDict_GetItemString(mp, "num_coeff");
			if(maxIter)
				if(PyInt_Check(numCoeff))
					params.mp.numCoeff = PyInt_AsLong(numCoeff);
				else if(PyFloat_Check(numCoeff))
					params.mp.numCoeff = static_cast<int>(PyFloat_AsDouble(maxIter));
				else
					throw Exception("mp.num_coeff should be of type `int`.");
		}

		PyObject* gsm = PyDict_GetItemString(parameters, "gsm");

		if(!gsm)
			gsm = PyDict_GetItemString(parameters, "GSM");

		if(gsm && PyDict_Check(gsm)) {
			PyObject* maxIter = PyDict_GetItemString(gsm, "max_iter");
			if(maxIter)
				if(PyInt_Check(maxIter))
					params.gsm.maxIter = PyInt_AsLong(maxIter);
				else if(PyFloat_Check(maxIter))
					params.gsm.maxIter = static_cast<int>(PyFloat_AsDouble(maxIter));
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
	char* kwlist[] = {"num_visibles", "num_hiddens", "ssize", "num_scales", 0};
	int num_visibles;
	int num_hiddens = -1;
	int ssize = 1;
	int num_scales = 10;

	// read arguments
	if(!PyArg_ParseTupleAndKeywords(args, kwds, "i|iii", kwlist,
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
	return PyArray_FromMatrixXd(self->isa->basis());
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



PyObject* ISA_basis(ISAObject* self, PyObject*, PyObject*) {
	try {
		return PyArray_FromMatrixXd(self->isa->basis());

	} catch(Exception exception) {
		PyErr_SetString(PyExc_RuntimeError, exception.message());
		return 0;
	}

	return 0;
}



PyObject* ISA_set_basis(ISAObject* self, PyObject* args, PyObject* kwds) {
	char* kwlist[] = {"basis", 0};

	PyObject* basis = 0;

	// read arguments
	if(!PyArg_ParseTupleAndKeywords(args, kwds, "O", kwlist, &basis))
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



PyObject* ISA_nullspace_basis(ISAObject* self, PyObject* args, PyObject* kwds) {
	try {
		return PyArray_FromMatrixXd(self->isa->nullspaceBasis());

	} catch(Exception exception) {
		PyErr_SetString(PyExc_RuntimeError, exception.message());
		return 0;
	}

	return 0;
}



PyObject* ISA_hidden_states(ISAObject* self, PyObject*, PyObject*) {
	try {
		return PyArray_FromMatrixXd(self->isa->hiddenStates());

	} catch(Exception exception) {
		PyErr_SetString(PyExc_RuntimeError, exception.message());
		return 0;
	}

	return 0;
}



PyObject* ISA_set_hidden_states(ISAObject* self, PyObject* args, PyObject* kwds) {
	char* kwlist[] = {"states", 0};

	PyObject* states = 0;

	// read arguments
	if(!PyArg_ParseTupleAndKeywords(args, kwds, "O", kwlist, &states))
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



PyObject* ISA_set_subspaces(ISAObject* self, PyObject* args, PyObject* kwds) {
	char* kwlist[] = {"subspaces", 0};

	PyObject* list = 0;

	// read arguments
	if(!PyArg_ParseTupleAndKeywords(args, kwds, "O", kwlist, &list))
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



PyObject* ISA_initialize(ISAObject* self, PyObject* args, PyObject* kwds) {
	char* kwlist[] = {"data", 0};

	PyObject* data = 0;

	// read arguments
	if(!PyArg_ParseTupleAndKeywords(args, kwds, "|O", kwlist, &data))
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



PyObject* ISA_train(ISAObject* self, PyObject* args, PyObject* kwds) {
	char* kwlist[] = {"data", "parameters", 0};

	PyObject* data;
	PyObject* parameters = 0;

	// read arguments
	if(!PyArg_ParseTupleAndKeywords(args, kwds, "O|O", kwlist, &data, &parameters))
		return 0;

	// make sure data is stored in NumPy array
	if(!PyArray_Check(data)) {
		PyErr_SetString(PyExc_TypeError, "Data has to be stored in a NumPy array.");
		return 0;
	}

	try {
		ISA::Parameters params = PyObject_ToParameters(self, parameters);
			
		// fit model to training data
		self->isa->train(PyArray_ToMatrixXd(data), params);
	} catch(Exception exception) {
		PyErr_SetString(PyExc_RuntimeError, exception.message());
		return 0;
	}

	Py_INCREF(Py_None);
	return Py_None;
}



PyObject* ISA_sample(ISAObject* self, PyObject* args, PyObject* kwds) {
	char* kwlist[] = {"num_samples", 0};

	int num_samples = 1;

	if(!PyArg_ParseTupleAndKeywords(args, kwds, "|i", kwlist, &num_samples))
		return 0;

	try {
		return PyArray_FromMatrixXd(self->isa->sample(num_samples));
	} catch(Exception exception) {
		PyErr_SetString(PyExc_RuntimeError, exception.message());
		return 0;
	}

	return 0;
}



PyObject* ISA_sample_prior(ISAObject* self, PyObject* args, PyObject* kwds) {
	char* kwlist[] = {"num_samples", 0};

	int num_samples = 1;

	if(!PyArg_ParseTupleAndKeywords(args, kwds, "|i", kwlist, &num_samples))
		return 0;

	try {
 		return PyArray_FromMatrixXd(self->isa->samplePrior(num_samples));
	} catch(Exception exception) {
		PyErr_SetString(PyExc_RuntimeError, exception.message());
		return 0;
	}

	return 0;
}



PyObject* ISA_sample_nullspace(ISAObject* self, PyObject* args, PyObject* kwds) {
	char* kwlist[] = {"data", "parameters", 0};

	PyObject* data;
	PyObject* parameters = 0;

	// read arguments
	if(!PyArg_ParseTupleAndKeywords(args, kwds, "O|O", kwlist, &data, &parameters))
		return 0;

	// make sure data is stored in NumPy array
	if(!PyArray_Check(data)) {
		PyErr_SetString(PyExc_TypeError, "Data has to be stored in a NumPy array.");
		return 0;
	}

	try {
		return PyArray_FromMatrixXd(self->isa->sampleNullspace(
			PyArray_ToMatrixXd(data),
			PyObject_ToParameters(self, parameters)));
	} catch(Exception exception) {
		PyErr_SetString(PyExc_RuntimeError, exception.message());
		return 0;
	}

	return 0;
}



PyObject* ISA_sample_posterior(ISAObject* self, PyObject* args, PyObject* kwds) {
	char* kwlist[] = {"data", "parameters", 0};

	PyObject* data;
	PyObject* parameters = 0;

	// read arguments
	if(!PyArg_ParseTupleAndKeywords(args, kwds, "O|O", kwlist, &data, &parameters))
		return 0;

	// make sure data is stored in NumPy array
	if(!PyArray_Check(data)) {
		PyErr_SetString(PyExc_TypeError, "Data has to be stored in a NumPy array.");
		return 0;
	}

	try {
		return PyArray_FromMatrixXd(self->isa->samplePosterior(
			PyArray_ToMatrixXd(data),
			PyObject_ToParameters(self, parameters)));
	} catch(Exception exception) {
		PyErr_SetString(PyExc_RuntimeError, exception.message());
		return 0;
	}

	return 0;
}



PyObject* ISA_sample_posterior_ais(ISAObject* self, PyObject* args, PyObject* kwds) {
	char* kwlist[] = {"data", "parameters", 0};

	PyObject* data;
	PyObject* parameters = 0;

	// read arguments
	if(!PyArg_ParseTupleAndKeywords(args, kwds, "O|O", kwlist, &data, &parameters))
		return 0;

	// make sure data is stored in NumPy array
	if(!PyArray_Check(data)) {
		PyErr_SetString(PyExc_TypeError, "Data has to be stored in a NumPy array.");
		return 0;
	}

	try {
		ISA::Parameters params = PyObject_ToParameters(self, parameters);

		pair<MatrixXd, MatrixXd> result = self->isa->samplePosteriorAIS(PyArray_ToMatrixXd(data), params);

		PyObject* samples = PyArray_FromMatrixXd(result.first);
		PyObject* logWeights = PyArray_FromMatrixXd(result.second);

		PyObject* tuple = Py_BuildValue("(OO)", samples, logWeights);

		Py_DECREF(samples);
		Py_DECREF(logWeights);

		return tuple;
	} catch(Exception exception) {
		PyErr_SetString(PyExc_RuntimeError, exception.message());
		return 0;
	}

	return 0;
}



PyObject* ISA_sample_ais(ISAObject* self, PyObject* args, PyObject* kwds) {
	char* kwlist[] = {"data", "parameters", 0};

	PyObject* data;
	PyObject* parameters = 0;

	// read arguments
	if(!PyArg_ParseTupleAndKeywords(args, kwds, "O|O", kwlist, &data, &parameters))
		return 0;

	// make sure data is stored in NumPy array
	if(!PyArray_Check(data)) {
		PyErr_SetString(PyExc_TypeError, "Data has to be stored in a NumPy array.");
		return 0;
	}

	try {
		return PyArray_FromMatrixXd(self->isa->sampleAIS(
			PyArray_ToMatrixXd(data),
			PyObject_ToParameters(self, parameters)));
	} catch(Exception exception) {
		PyErr_SetString(PyExc_RuntimeError, exception.message());
		return 0;
	}

	return 0;
}



PyObject* ISA_sample_scales(ISAObject* self, PyObject* args, PyObject* kwds) {
	char* kwlist[] = {"states", 0};

	PyObject* states;

	// read arguments
	if(!PyArg_ParseTupleAndKeywords(args, kwds, "O", kwlist, &states))
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



PyObject* ISA_matching_pursuit(ISAObject* self, PyObject* args, PyObject* kwds) {
	char* kwlist[] = {"data", "parameters", 0};

	PyObject* data;
	PyObject* parameters = 0;

	// read arguments
	if(!PyArg_ParseTupleAndKeywords(args, kwds, "O|O", kwlist, &data, &parameters))
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



PyObject* ISA_prior_energy(ISAObject* self, PyObject* args, PyObject* kwds) {
	char* kwlist[] = {"states", 0};

	PyObject* states;

	// read arguments
	if(!PyArg_ParseTupleAndKeywords(args, kwds, "O", kwlist, &states))
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



PyObject* ISA_prior_energy_gradient(ISAObject* self, PyObject* args, PyObject* kwds) {
	char* kwlist[] = {"states", 0};

	PyObject* states;

	// read arguments
	if(!PyArg_ParseTupleAndKeywords(args, kwds, "O", kwlist, &states))
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



PyObject* ISA_loglikelihood(ISAObject* self, PyObject* args, PyObject* kwds) {
	char* kwlist[] = {"data", "parameters", "return_all", 0};

	PyObject* data;
	PyObject* parameters = 0;
	int return_all = 0;

	// read arguments
	if(!PyArg_ParseTupleAndKeywords(args, kwds, "O|Oi", kwlist, &data, &parameters, &return_all))
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



PyObject* ISA_evaluate(ISAObject* self, PyObject* args, PyObject* kwds) {
	char* kwlist[] = {"data", "parameters", 0};

	PyObject* data;
	PyObject* parameters = 0;

	// read arguments
	if(!PyArg_ParseTupleAndKeywords(args, kwds, "O|O", kwlist, &data, &parameters))
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
