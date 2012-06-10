#ifndef ISAINTERFACE_H
#define ISAINTERFACE_H

#include "isa.h"
#include "exception.h"
#include "pyutils.h"
#include "Eigen/Core"
#include "gsminterface.h"
#include <iostream>

using namespace Eigen;

struct ISAObject {
	PyObject_HEAD
	ISA* isa;
};



class CallbackTrain : public ISA::Callback {
	public:
		CallbackTrain(ISAObject* isa, PyObject* callback);
		CallbackTrain(const CallbackTrain& callbackTrain);
		virtual ~CallbackTrain();
		virtual CallbackTrain& operator=(const CallbackTrain& callbackTrain);
		virtual CallbackTrain* copy();
		virtual bool operator()(int iter, const ISA&);

	private:
		ISAObject* mIsa;
		PyObject* mCallback;
};



CallbackTrain::CallbackTrain(ISAObject* isa, PyObject* callback) : 
	mIsa(isa), 
	mCallback(callback) 
{
	Py_INCREF(mIsa);
	Py_INCREF(mCallback);
}



CallbackTrain::CallbackTrain(const CallbackTrain& callbackTrain) :
	mIsa(callbackTrain.mIsa),
	mCallback(callbackTrain.mCallback)
{
	Py_INCREF(mIsa);
	Py_INCREF(mCallback);
}



CallbackTrain::~CallbackTrain() {
	Py_DECREF(mIsa);
	Py_DECREF(mCallback);
}



CallbackTrain& CallbackTrain::operator=(const CallbackTrain& callbackTrain) {
	Py_DECREF(mIsa);
	Py_DECREF(mCallback);

	mIsa = callbackTrain.mIsa;
	mCallback = callbackTrain.mCallback;

	Py_INCREF(mIsa);
	Py_INCREF(mCallback);
}



CallbackTrain* CallbackTrain::copy() {
	return new CallbackTrain(*this);
}



bool CallbackTrain::operator()(int iter, const ISA&) {
	// call Python object
	PyObject* args = Py_BuildValue("(iO)", iter, mIsa);
	PyObject* result = PyObject_CallObject(mCallback, args);

	Py_DECREF(args);

	// if cont is false, training will be aborted
	bool cont = true;
	if(result) {
		if(PyBool_Check(result))
			cont = (result == Py_True);
		Py_DECREF(result);
	} else {
		throw Exception("Some error occured during callback().");
	}

	return cont;
}



/**
 * Extract parameters from Python object.
 */
ISA::Parameters PyObject_ToParameters(ISAObject* self, PyObject* parameters) {
	ISA::Parameters params;

	// read parameters from dictionary
	if(parameters) {
		if(!PyDict_Check(parameters))
			throw Exception("Parameters should be stored in a dictionary.");

		PyObject* trainingMethod = PyDict_GetItemString(parameters, "training_method");
		if(trainingMethod)
			if(PyString_Check(trainingMethod))
				params.trainingMethod = PyString_AsString(trainingMethod);
			else
				throw Exception("training_method should be of type string.");

		PyObject* samplingMethod = PyDict_GetItemString(parameters, "sampling_method");
		if(samplingMethod)
			if(PyString_Check(samplingMethod))
				params.trainingMethod = PyString_AsString(samplingMethod);
			else
				throw Exception("sampling_method should be of type string.");

		PyObject* maxIter = PyDict_GetItemString(parameters, "max_iter");
		if(maxIter)
			if(PyInt_Check(maxIter))
				params.maxIter = PyInt_AsLong(maxIter);
			else if(PyFloat_Check(maxIter))
				params.maxIter = static_cast<int>(PyFloat_AsDouble(maxIter));
			else
				throw Exception("max_iter should be of type int.");

		PyObject* adaptive = PyDict_GetItemString(parameters, "adaptive");
		if(adaptive)
			if(PyBool_Check(adaptive))
				params.adaptive = (adaptive == Py_True);
			else
				throw Exception("adaptive should be of type bool.");

		PyObject* trainPrior = PyDict_GetItemString(parameters, "train_prior");
		if(trainPrior)
			if(PyBool_Check(trainPrior))
				params.trainPrior = (trainPrior == Py_True);
			else
				throw Exception("train_prior should be of type bool.");

		PyObject* callback = PyDict_GetItemString(parameters, "callback");
		if(callback)
			if(PyCallable_Check(callback))
				params.callback = new CallbackTrain(self, callback);
			else if(callback != Py_None)
				throw Exception("callback should be a function or callable object.");

		PyObject* SGD = PyDict_GetItemString(parameters, "SGD");

		if(!SGD)
			SGD = PyDict_GetItemString(parameters, "sgd");

		if(SGD && PyDict_Check(SGD)) {
			PyObject* maxIter = PyDict_GetItemString(SGD, "max_iter");
			if(maxIter)
				if(PyInt_Check(maxIter))
					params.SGD.maxIter = PyInt_AsLong(maxIter);
				else if(PyFloat_Check(maxIter))
					params.SGD.maxIter = static_cast<int>(PyFloat_AsDouble(maxIter));
				else
					throw Exception("SGD.max_iter should be of type int.");

			PyObject* batchSize = PyDict_GetItemString(SGD, "batch_size");
			if(batchSize)
				if(PyInt_Check(batchSize))
					params.SGD.batchSize = PyInt_AsLong(batchSize);
				else if(PyFloat_Check(batchSize))
					params.SGD.batchSize = static_cast<int>(PyFloat_AsDouble(batchSize));
				else
					throw Exception("SGD.batch_size should be of type int.");

			PyObject* stepWidth = PyDict_GetItemString(SGD, "step_width");
			if(stepWidth)
				if(PyFloat_Check(stepWidth))
					params.SGD.stepWidth = PyFloat_AsDouble(stepWidth);
				else if(PyInt_Check(stepWidth))
					params.SGD.stepWidth = static_cast<double>(PyFloat_AsDouble(stepWidth));
				else
					throw Exception("SGD.step_width should be of type float.");

			PyObject* momentum = PyDict_GetItemString(SGD, "momentum");
			if(momentum)
				if(PyFloat_Check(momentum))
					params.SGD.momentum = PyFloat_AsDouble(momentum);
				else if(PyInt_Check(momentum))
					params.SGD.momentum = static_cast<double>(PyInt_AsLong(momentum));
				else
					throw Exception("SGD.momentum should be of type float.");

			PyObject* shuffle = PyDict_GetItemString(SGD, "shuffle");
			if(shuffle)
				if(PyBool_Check(shuffle))
					params.SGD.shuffle = (shuffle == Py_True);
				else
					throw Exception("SGD.shuffle should be of type bool.");

			PyObject* pocket = PyDict_GetItemString(SGD, "pocket");
			if(pocket)
				if(PyBool_Check(pocket))
					params.SGD.pocket = (pocket == Py_True);
				else
					throw Exception("SGD.pocket should be of type bool.");
		}

		PyObject* GSM = PyDict_GetItemString(parameters, "GSM");

		if(!GSM)
			GSM = PyDict_GetItemString(parameters, "gsm");

		if(GSM && PyDict_Check(GSM)) {
			PyObject* maxIter = PyDict_GetItemString(GSM, "max_iter");
			if(maxIter)
				if(PyInt_Check(maxIter))
					params.GSM.maxIter = PyInt_AsLong(maxIter);
				else if(PyFloat_Check(maxIter))
					params.GSM.maxIter = static_cast<int>(PyFloat_AsDouble(maxIter));
				else
					throw Exception("GSM.max_iter should be of type int.");

			PyObject* tol = PyDict_GetItemString(GSM, "tol");
			if(tol)
				if(PyFloat_Check(tol))
					params.GSM.tol = PyFloat_AsDouble(tol);
				else if(PyInt_Check(tol))
					params.GSM.tol = static_cast<double>(PyInt_AsLong(tol));
				else
					throw Exception("GSM.tol should be of type float.");
		}
	}

	return params;
}



/**
 * Create a new ISA object.
 */
static PyObject* ISA_new(PyTypeObject* type, PyObject* args, PyObject* kwds) {
	PyObject* self = type->tp_alloc(type, 0);

	if(self)
		reinterpret_cast<ISAObject*>(self)->isa = 0;

	return self;
}



/**
 * Initialize ISA object.
 */
static int ISA_init(ISAObject* self, PyObject* args, PyObject* kwds) {
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



/**
 * Delete ISA object.
 */
static void ISA_dealloc(ISAObject* self) {
	// delete actual ISA instance
	delete self->isa;

	// delete ISA object
	self->ob_type->tp_free(reinterpret_cast<PyObject*>(self));
}



/**
 * Return number of visible units.
 */
static PyObject* ISA_dim(ISAObject* self, PyObject*, void*) {
	return PyInt_FromLong(self->isa->numVisibles());
}



/**
 * Return number of visible units.
 */
static PyObject* ISA_num_visibles(ISAObject* self, PyObject*, void*) {
	return PyInt_FromLong(self->isa->numVisibles());
}



/**
 * Return number of hidden units.
 */
static PyObject* ISA_num_hiddens(ISAObject* self, PyObject*, void*) {
	return PyInt_FromLong(self->isa->numHiddens());
}



/**
 * Return copy of linear basis.
 */
static PyObject* ISA_A(ISAObject* self, PyObject*, void*) {
	return PyArray_FromMatrixXd(self->isa->basis());
}



/**
 * Replace linear basis.
 */
static int ISA_set_A(ISAObject* self, PyObject* value, void*) {
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



/**
 * Return copy of subspace GSMs.
 */
static PyObject* ISA_subspaces(ISAObject* self, PyObject*, void*) {
	vector<GSM> subspaces = self->isa->subspaces();

	PyObject* list = PyList_New(subspaces.size());

	for(int i = 0; i < subspaces.size(); ++i) {
		// create Python object representing GSM
		PyObject* gsmObj = _PyObject_New(&GSM_type);
		reinterpret_cast<GSMObject*>(gsmObj)->gsm = new GSM(subspaces[i]);

		PyList_SetItem(list, i, gsmObj);
	}

	return list;
}



static PyObject* ISA_default_parameters(ISAObject* self) {
	ISA::Parameters params;
	PyObject* parameters = PyDict_New();
	PyObject* SGD = PyDict_New();
	PyObject* GSM = PyDict_New();

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

	PyDict_SetItemString(SGD, "max_iter", PyInt_FromLong(params.SGD.maxIter));
	PyDict_SetItemString(SGD, "batch_size", PyInt_FromLong(params.SGD.batchSize));
	PyDict_SetItemString(SGD, "step_width", PyFloat_FromDouble(params.SGD.stepWidth));
	PyDict_SetItemString(SGD, "momentum", PyFloat_FromDouble(params.SGD.momentum));

	if(params.SGD.shuffle) {
		PyDict_SetItemString(SGD, "shuffle", Py_True);
		Py_INCREF(Py_True);
	} else {
		PyDict_SetItemString(SGD, "shuffle", Py_False);
		Py_DECREF(Py_False);
	}

	if(params.SGD.pocket) {
		PyDict_SetItemString(SGD, "pocket", Py_True);
		Py_INCREF(Py_True);
	} else {
		PyDict_SetItemString(SGD, "pocket", Py_False);
		Py_DECREF(Py_False);
	}

	PyDict_SetItemString(GSM, "max_iter", PyInt_FromLong(params.GSM.maxIter));
	PyDict_SetItemString(GSM, "tol", PyFloat_FromDouble(params.GSM.tol));

	PyDict_SetItemString(parameters, "SGD", SGD);
	PyDict_SetItemString(parameters, "GSM", GSM);

	return parameters;
}



static PyObject* ISA_nullspace_basis(ISAObject* self, PyObject* args, PyObject* kwds) {
	try {
		return PyArray_FromMatrixXd(self->isa->nullspaceBasis());
	} catch(Exception exception) {
		PyErr_SetString(PyExc_RuntimeError, exception.message());
		return 0;
	}
}



static PyObject* ISA_initialize(ISAObject* self, PyObject* args, PyObject* kwds) {
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



static PyObject* ISA_train(ISAObject* self, PyObject* args, PyObject* kwds) {
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



static PyObject* ISA_sample(ISAObject* self, PyObject* args, PyObject* kwds) {
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



static PyObject* ISA_sample_prior(ISAObject* self, PyObject* args, PyObject* kwds) {
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



static PyObject* ISA_sample_nullspace(ISAObject* self, PyObject* args, PyObject* kwds) {
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



static PyObject* ISA_sample_posterior(ISAObject* self, PyObject* args, PyObject* kwds) {
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



static PyObject* ISA_prior_energy(ISAObject* self, PyObject* args, PyObject* kwds) {
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


static PyObject* ISA_prior_energy_gradient(ISAObject* self, PyObject* args, PyObject* kwds) {
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



static PyObject* ISA_loglikelihood(ISAObject* self, PyObject* args, PyObject* kwds) {
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



static PyGetSetDef ISA_getset[] = {
	{"dim", (getter)ISA_dim, 0, 0},
	{"num_visibles", (getter)ISA_num_visibles, 0, 0},
	{"num_hiddens", (getter)ISA_num_hiddens, 0, 0},
	{"A", (getter)ISA_A, (setter)ISA_set_A, 0},
	{"subspaces", (getter)ISA_subspaces, 0, 0},
	{0}
};



static PyMethodDef ISA_methods[] = {
	{"default_parameters", (PyCFunction)ISA_default_parameters, METH_VARARGS, 0},
	{"nullspace_basis", (PyCFunction)ISA_nullspace_basis, METH_NOARGS, 0},
	{"initialize", (PyCFunction)ISA_initialize, METH_VARARGS|METH_KEYWORDS, 0},
	{"train", (PyCFunction)ISA_train, METH_VARARGS|METH_KEYWORDS, 0},
	{"sample", (PyCFunction)ISA_sample, METH_VARARGS|METH_KEYWORDS, 0},
	{"sample_prior", (PyCFunction)ISA_sample_prior, METH_VARARGS|METH_KEYWORDS, 0},
	{"sample_nullspace", (PyCFunction)ISA_sample_nullspace, METH_VARARGS|METH_KEYWORDS, 0},
	{"sample_posterior", (PyCFunction)ISA_sample_posterior, METH_VARARGS|METH_KEYWORDS, 0},
	{"prior_energy", (PyCFunction)ISA_prior_energy, METH_VARARGS|METH_KEYWORDS, 0},
	{"prior_energy_gradient", (PyCFunction)ISA_prior_energy_gradient, METH_VARARGS|METH_KEYWORDS, 0},
	{"loglikelihood", (PyCFunction)ISA_loglikelihood, METH_VARARGS|METH_KEYWORDS, 0},
	{0}
};



static PyTypeObject ISA_type = {
	PyObject_HEAD_INIT(0)
	0,                         /*ob_size*/
	"isa.ISA",                 /*tp_name*/
	sizeof(ISAObject),         /*tp_basicsize*/
	0,                         /*tp_itemsize*/
	(destructor)ISA_dealloc,   /*tp_dealloc*/
	0,                         /*tp_print*/
	0,                         /*tp_getattr*/
	0,                         /*tp_setattr*/
	0,                         /*tp_compare*/
	0,                         /*tp_repr*/
	0,                         /*tp_as_number*/
	0,                         /*tp_as_sequence*/
	0,                         /*tp_as_mapping*/
	0,                         /*tp_hash */
	0,                         /*tp_call*/
	0,                         /*tp_str*/
	0,                         /*tp_getattro*/
	0,                         /*tp_setattro*/
	0,                         /*tp_as_buffer*/
	Py_TPFLAGS_DEFAULT,        /*tp_flags*/
	0,                         /*tp_doc*/
	0,                         /*tp_traverse*/
	0,                         /*tp_clear*/
	0,                         /*tp_richcompare*/
	0,                         /*tp_weaklistoffset*/
	0,                         /*tp_iter*/
	0,                         /*tp_iternext*/
	ISA_methods,               /*tp_methods*/
	0,                         /*tp_members*/
	ISA_getset,                /*tp_getset*/
	0,                         /*tp_base*/
	0,                         /*tp_dict*/
	0,                         /*tp_descr_get*/
	0,                         /*tp_descr_set*/
	0,                         /*tp_dictoffset*/
	(initproc)ISA_init,        /*tp_init*/
	0,                         /*tp_alloc*/
	ISA_new,                   /*tp_new*/
};

#endif
