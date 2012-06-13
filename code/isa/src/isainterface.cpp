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

		PyObject* trainingMethod = PyDict_GetItemString(parameters, "training_method");
		if(trainingMethod)
			if(PyString_Check(trainingMethod))
				params.trainingMethod = PyString_AsString(trainingMethod);
			else
				throw Exception("training_method should be of type `string`.");

		PyObject* samplingMethod = PyDict_GetItemString(parameters, "sampling_method");
		if(samplingMethod)
			if(PyString_Check(samplingMethod))
				params.trainingMethod = PyString_AsString(samplingMethod);
			else
				throw Exception("sampling_method should be of type `string`.");

		PyObject* maxIter = PyDict_GetItemString(parameters, "max_iter");
		if(maxIter)
			if(PyInt_Check(maxIter))
				params.maxIter = PyInt_AsLong(maxIter);
			else if(PyFloat_Check(maxIter))
				params.maxIter = static_cast<int>(PyFloat_AsDouble(maxIter));
			else
				throw Exception("max_iter should be of type `int`.");

		PyObject* adaptive = PyDict_GetItemString(parameters, "adaptive");
		if(adaptive)
			if(PyBool_Check(adaptive))
				params.adaptive = (adaptive == Py_True);
			else
				throw Exception("adaptive should be of type `bool`.");

		PyObject* trainPrior = PyDict_GetItemString(parameters, "train_prior");
		if(trainPrior)
			if(PyBool_Check(trainPrior))
				params.trainPrior = (trainPrior == Py_True);
			else
				throw Exception("train_prior should be of type `bool`.");

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
	}

	return params;
}



PyObject* ISA_new(PyTypeObject* type, PyObject* args, PyObject* kwds) {
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
	PyObject* gsm = PyDict_New();
	PyObject* gibbs = PyDict_New();

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

	if(params.trainPrior) {
		PyDict_SetItemString(parameters, "train_prior", Py_True);
		Py_INCREF(Py_True);
	} else {
		PyDict_SetItemString(parameters, "train_prior", Py_False);
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
		Py_DECREF(Py_False);
	}

	if(params.sgd.pocket) {
		PyDict_SetItemString(sgd, "pocket", Py_True);
		Py_INCREF(Py_True);
	} else {
		PyDict_SetItemString(sgd, "pocket", Py_False);
		Py_DECREF(Py_False);
	}

	PyDict_SetItemString(gsm, "max_iter", PyInt_FromLong(params.gsm.maxIter));
	PyDict_SetItemString(gsm, "tol", PyFloat_FromDouble(params.gsm.tol));

	PyDict_SetItemString(gibbs, "verbosity", PyInt_FromLong(params.gibbs.verbosity));
	PyDict_SetItemString(gibbs, "ini_iter", PyInt_FromLong(params.gibbs.iniIter));
	PyDict_SetItemString(gibbs, "num_iter", PyInt_FromLong(params.gibbs.numIter));

	PyDict_SetItemString(parameters, "sgd", sgd);
	PyDict_SetItemString(parameters, "gsm", gsm);
	PyDict_SetItemString(parameters, "gibbs", gibbs);

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
		ISA::Parameters params = PyObject_ToParameters(self, parameters);
		return PyArray_FromMatrixXd(self->isa->sampleNullspace(PyArray_ToMatrixXd(data), params));
	} catch(Exception exception) {
		PyErr_SetString(PyExc_RuntimeError, exception.message());
		return 0;
	}
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
		ISA::Parameters params = PyObject_ToParameters(self, parameters);
		return PyArray_FromMatrixXd(self->isa->samplePosterior(PyArray_ToMatrixXd(data), params));
	} catch(Exception exception) {
		PyErr_SetString(PyExc_RuntimeError, exception.message());
		return 0;
	}
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
}



PyObject* ISA_loglikelihood(ISAObject* self, PyObject* args, PyObject* kwds) {
	char* kwlist[] = {"data", 0};

	PyObject* data;

	// read arguments
	if(!PyArg_ParseTupleAndKeywords(args, kwds, "O", kwlist, &data))
		return 0;

	// make sure data is stored in NumPy array
	if(!PyArray_Check(data)) {
		PyErr_SetString(PyExc_TypeError, "Data has to be stored in a NumPy array.");
		return 0;
	}

	try {
		return PyArray_FromMatrixXd(self->isa->logLikelihood(PyArray_ToMatrixXd(data)));
	} catch(Exception exception) {
		PyErr_SetString(PyExc_RuntimeError, exception.message());
		return 0;
	}
}
