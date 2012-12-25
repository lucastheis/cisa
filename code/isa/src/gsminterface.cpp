#include "gsminterface.h"
#include "Eigen/Core"
#include "exception.h"

using namespace Eigen;

const char* GSM_doc =
	"A finite Gaussian scale mixture distribution.";



PyObject* GSM_new(PyTypeObject* type, PyObject* args, PyObject* kwds) {
	PyObject* self = type->tp_alloc(type, 0);

	if(self)
		reinterpret_cast<GSMObject*>(self)->gsm = 0;

	return self;
}



int GSM_init(GSMObject* self, PyObject* args, PyObject* kwds) {
	const char* kwlist[] = {"dim", "num_scales", 0};
	int dim;
	int num_scales = 10;

	// read arguments
	if(!PyArg_ParseTupleAndKeywords(args, kwds, "i|i", const_cast<char**>(kwlist),
		&dim, &num_scales))
		return -1;

	// create actual GSM instance
	self->gsm = new GSM(dim, num_scales);

	return 0;
}



void GSM_dealloc(GSMObject* self) {
	// delete actual GSM instance
	delete self->gsm;

	// delete GSM object
	self->ob_type->tp_free(reinterpret_cast<PyObject*>(self));
}



PyObject* GSM_dim(GSMObject* self, PyObject*, void*) {
	return PyInt_FromLong(self->gsm->dim());
}



PyObject* GSM_num_scales(GSMObject* self, PyObject*, void*) {
	return PyInt_FromLong(self->gsm->numScales());
}



PyObject* GSM_scales(GSMObject* self, PyObject*, void*) {
	PyObject* array = PyArray_FromMatrixXd(self->gsm->scales());

	// make array immutable
	reinterpret_cast<PyArrayObject*>(array)->flags &= ~NPY_WRITEABLE;

	return array;
}



int GSM_set_scales(GSMObject* self, PyObject* value, void*) {
	PyObject* array = PyArray_FROM_OTF(value, NPY_DOUBLE, NPY_IN_ARRAY);

	if(!array) {
		PyErr_SetString(PyExc_TypeError, "Scales should be of type `ndarray`.");
		return -1;
	}

	try {
		self->gsm->setScales(PyArray_ToMatrixXd(array));

	} catch(Exception exception) {
		Py_DECREF(array);
		PyErr_SetString(PyExc_RuntimeError, exception.message());
		return -1;
	}

	Py_DECREF(array);

	return 0;
}



const char* GSM_variance_doc =
	"Computes the variance of the distribution.\n"
	"\n"
	"@rtype: C{float}\n"
	"@return: the variance of the distribution";

PyObject* GSM_variance(GSMObject* self, PyObject*, PyObject*) {
	try {
		return PyFloat_FromDouble(self->gsm->variance());
	} catch(Exception exception) {
		PyErr_SetString(PyExc_RuntimeError, exception.message());
		return 0;
	}
}



const char* GSM_normalize_doc =
	"Normalizes the variance without changing the shape of the distribution.";

PyObject* GSM_normalize(GSMObject* self, PyObject*, PyObject*) {
	try {
		self->gsm->normalize();
	} catch(Exception exception) {
		PyErr_SetString(PyExc_RuntimeError, exception.message());
		return 0;
	}

	Py_INCREF(Py_None);
	return Py_None;
}



const char* GSM_train_doc =
	"Optimizes the parameters of the distribution using expectation maximization.\n"
	"\n"
	"@type  data: C{ndarray}\n"
	"@param data: data points stored in columns\n"
	"\n"
	"@type  max_iter: C{int}\n"
	"@param max_iter: maximum number of EM iterations (default: 100)\n"
	"\n"
	"@type  tol: C{float}\n"
	"@param tol: stop if improvement in log-likelihood is smaller than C{tol} (default: 1e-5)";

PyObject* GSM_train(GSMObject* self, PyObject* args, PyObject* kwds) {
	const char* kwlist[] = {"data", "max_iter", "tol", 0};

	PyObject* data;
	int max_iter = 100;
	double tol = 1e-5;

	// read arguments
	if(!PyArg_ParseTupleAndKeywords(args, kwds, "O|id", const_cast<char**>(kwlist), &data, &max_iter, &tol))
		return 0;

	// make sure data is stored in NumPy array
	if(!PyArray_Check(data)) {
		PyErr_SetString(PyExc_TypeError, "Data has to be stored in a NumPy array.");
		return 0;
	}

	try {
		if(self->gsm->train(PyArray_ToMatrixXd(data), max_iter, tol)) {
			Py_INCREF(Py_True);
			return Py_True;
		} else {
			Py_INCREF(Py_False);
			return Py_False;
		}
	} catch(Exception exception) {
		PyErr_SetString(PyExc_RuntimeError, exception.message());
		return 0;
	}

	return 0;
}



const char* GSM_posterior_doc =
	"Computes the posterior over standard deviations.\n"
	"\n"
	"@type  data: C{ndarray}\n"
	"@param data: data points stored in columns";

PyObject* GSM_posterior(GSMObject* self, PyObject* args, PyObject* kwds) {
	const char* kwlist[] = {"data", 0};

	PyObject* data;

	// read arguments
	if(!PyArg_ParseTupleAndKeywords(args, kwds, "O", const_cast<char**>(kwlist), &data))
		return 0;

	// make sure data is stored in NumPy array
	if(!PyArray_Check(data)) {
		PyErr_SetString(PyExc_TypeError, "Data has to be stored in a NumPy array.");
		return 0;
	}

	try {
		return PyArray_FromMatrixXd(self->gsm->posterior(PyArray_ToMatrixXd(data)));
	} catch(Exception exception) {
		PyErr_SetString(PyExc_RuntimeError, exception.message());
		return 0;
	}
}



const char* GSM_sample_doc =
	"Generates samples from the distribution.\n"
	"\n"
	"@type  num_samples: C{int}\n"
	"@param num_samples: number of samples to generate";

PyObject* GSM_sample(GSMObject* self, PyObject* args, PyObject* kwds) {
	const char* kwlist[] = {"num_samples", 0};

	int num_samples = 1;

	if(!PyArg_ParseTupleAndKeywords(args, kwds, "|i", const_cast<char**>(kwlist), &num_samples))
		return 0;

	try {
		return PyArray_FromMatrixXd(self->gsm->sample(num_samples));
	} catch(Exception exception) {
		PyErr_SetString(PyExc_RuntimeError, exception.message());
		return 0;
	}
}



const char* GSM_sample_posterior_doc =
	"Generates samples from the the posterior distribution over standard deviations.\n"
	"\n"
	"@type  data: C{ndarray}\n"
	"@param data: data points stored in columns";

PyObject* GSM_sample_posterior(GSMObject* self, PyObject* args, PyObject* kwds) {
	const char* kwlist[] = {"data", 0};

	PyObject* data;

	// read arguments
	if(!PyArg_ParseTupleAndKeywords(args, kwds, "O", const_cast<char**>(kwlist), &data))
		return 0;

	// make sure data is stored in NumPy array
	if(!PyArray_Check(data)) {
		PyErr_SetString(PyExc_TypeError, "Data has to be stored in a NumPy array.");
		return 0;
	}

	try {
		return PyArray_FromMatrixXd(self->gsm->samplePosterior(PyArray_ToMatrixXd(data)));
	} catch(Exception exception) {
		PyErr_SetString(PyExc_RuntimeError, exception.message());
		return 0;
	}
}



const char* GSM_loglikelihood_doc =
	"Computes the log-density of data points.\n"
	"\n"
	"@type  data: C{ndarray}\n"
	"@param data: data points stored in columns";

PyObject* GSM_loglikelihood(GSMObject* self, PyObject* args, PyObject* kwds) {
	const char* kwlist[] = {"data", 0};

	PyObject* data;

	// read arguments
	if(!PyArg_ParseTupleAndKeywords(args, kwds, "O", const_cast<char**>(kwlist), &data))
		return 0;

	// make sure data is stored in NumPy array
	if(!PyArray_Check(data)) {
		PyErr_SetString(PyExc_TypeError, "Data has to be stored in a NumPy array.");
		return 0;
	}

	try {
		return PyArray_FromMatrixXd(self->gsm->logLikelihood(PyArray_ToMatrixXd(data)));
	} catch(Exception exception) {
		PyErr_SetString(PyExc_RuntimeError, exception.message());
		return 0;
	}
}



const char* GSM_energy_doc =
	"Computes a negative unnormalized log-density of data points.\n"
	"\n"
	"@type  data: C{ndarray}\n"
	"@param data: data points stored in columns";

PyObject* GSM_energy(GSMObject* self, PyObject* args, PyObject* kwds) {
	const char* kwlist[] = {"data", 0};

	PyObject* data;

	// read arguments
	if(!PyArg_ParseTupleAndKeywords(args, kwds, "O", const_cast<char**>(kwlist), &data))
		return 0;

	// make sure data is stored in NumPy array
	if(!PyArray_Check(data)) {
		PyErr_SetString(PyExc_TypeError, "Data has to be stored in a NumPy array.");
		return 0;
	}

	try {
		return PyArray_FromMatrixXd(self->gsm->energy(PyArray_ToMatrixXd(data)));
	} catch(Exception exception) {
		PyErr_SetString(PyExc_RuntimeError, exception.message());
		return 0;
	}
}



const char* GSM_energy_gradient_doc =
	"Computes the gradient of the energy.\n"
	"\n"
	"@type  data: C{ndarray}\n"
	"@param data: data points stored in columns";

PyObject* GSM_energy_gradient(GSMObject* self, PyObject* args, PyObject* kwds) {
	const char* kwlist[] = {"data", 0};

	PyObject* data;

	// read arguments
	if(!PyArg_ParseTupleAndKeywords(args, kwds, "O", const_cast<char**>(kwlist), &data))
		return 0;

	// make sure data is stored in NumPy array
	if(!PyArray_Check(data)) {
		PyErr_SetString(PyExc_TypeError, "Data has to be stored in a NumPy array.");
		return 0;
	}

	try {
		return PyArray_FromMatrixXd(self->gsm->energyGradient(PyArray_ToMatrixXd(data)));
	} catch(Exception exception) {
		PyErr_SetString(PyExc_RuntimeError, exception.message());
		return 0;
	}
}



PyObject* GSM_reduce(GSMObject* self, PyObject*, PyObject*) {
	PyObject* args = Py_BuildValue("(ii)", self->gsm->dim(), self->gsm->numScales());

	PyObject* scales = GSM_scales(self, 0, 0);
	PyObject* state = Py_BuildValue("(O)", scales);
	Py_DECREF(scales);

	PyObject* result = Py_BuildValue("OOO", self->ob_type, args, state);
	Py_DECREF(args);
	Py_DECREF(state);

	return result;
}



PyObject* GSM_setstate(GSMObject* self, PyObject* state, PyObject*) {
	PyObject* scales;

	if(!PyArg_ParseTuple(state, "(O)", &scales))
		return 0;

	try {
		self->gsm->setScales(PyArray_ToMatrixXd(scales));
	} catch(Exception exception) {
		PyErr_SetString(PyExc_RuntimeError, exception.message());
		return 0;
	}

	Py_INCREF(Py_None);
	return Py_None;
}
