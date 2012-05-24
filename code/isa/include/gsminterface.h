#ifndef GSMINTERFACE_H
#define GSMINTERFACE_H

#include "gsm.h"
#include "exception.h"
#include "pyutils.h"
#include "Eigen/Core"
#include <iostream>

using namespace Eigen;

struct GSMObject {
	PyObject_HEAD
	GSM* gsm;
};



/**
 * Create a new GSM object.
 */
static PyObject* GSM_new(PyTypeObject* type, PyObject* args, PyObject* kwds) {
	PyObject* self = type->tp_alloc(type, 0);

	if(self)
		reinterpret_cast<GSMObject*>(self)->gsm = 0;

	return self;
}



/**
 * Initialize GSM object.
 */
static int GSM_init(GSMObject* self, PyObject* args, PyObject* kwds) {
	char* kwlist[] = {"dim", "num_scales", 0};
	int dim;
	int num_scales = 10;

	// read arguments
	if(!PyArg_ParseTupleAndKeywords(args, kwds, "i|i", kwlist,
		&dim, &num_scales))
		return -1;

	// create actual GSM instance
	self->gsm = new GSM(dim, num_scales);

	return 0;
}



/**
 * Delete GSM object.
 */
static void GSM_dealloc(GSMObject* self) {
	// delete actual GSM instance
	delete self->gsm;

	// delete GSM object
	self->ob_type->tp_free(reinterpret_cast<PyObject*>(self));
}



/**
 * Return number of visible units.
 */
static PyObject* GSM_dim(GSMObject* self, PyObject*, void*) {
	return PyInt_FromLong(self->gsm->dim());
}



/**
 * Return number of hidden units.
 */
static PyObject* GSM_num_scales(GSMObject* self, PyObject*, void*) {
	return PyInt_FromLong(self->gsm->numScales());
}



/**
 * Return copy of linear basis.
 */
static PyObject* GSM_scales(GSMObject* self, PyObject*, void*) {
	return PyArray_FromMatrixXd(self->gsm->scales());
}



/**
 * Replace linear basis.
 */
static int GSM_set_scales(GSMObject* self, PyObject* value, void*) {
	if(!PyArray_Check(value)) {
		PyErr_SetString(PyExc_TypeError, "Scales should be of type `ndarray`.");
		return -1;
	}

	try {
		self->gsm->setScales(PyArray_ToMatrixXd(value));

	} catch(Exception exception) {
		PyErr_SetString(PyExc_RuntimeError, exception.message());
		return -1;
	}

	return 0;
}



static PyObject* GSM_train(GSMObject* self, PyObject* args, PyObject* kwds) {
	char* kwlist[] = {"data", "max_iter", "tol"};

	PyObject* data;
	int max_iter = 100;
	double tol = 1e-5;

	// read arguments
	if(!PyArg_ParseTupleAndKeywords(args, kwds, "O|id", kwlist, &data, &max_iter, &tol))
		return 0;

	// make sure data is stored in NumPy array
	if(!PyArray_Check(data)) {
		PyErr_SetString(PyExc_TypeError, "Data has to be stored in a NumPy array.");
		return 0;
	}

	try {
		self->gsm->train(PyArray_ToMatrixXd(data), max_iter, tol);

	} catch(Exception exception) {
		PyErr_SetString(PyExc_RuntimeError, exception.message());
		return 0;
	}

	Py_INCREF(Py_None);
	return Py_None;
}



static PyGetSetDef GSM_getset[] = {
	{"dim", (getter)GSM_dim, 0, 0},
	{"num_scales", (getter)GSM_num_scales, 0, 0},
	{"scales", (getter)GSM_scales, (setter)GSM_set_scales, 0},
	{0}
};



static PyMethodDef GSM_methods[] = {
	{"train", (PyCFunction)GSM_train, METH_KEYWORDS, 0},
	{0}
};



static PyTypeObject GSM_type = {
	PyObject_HEAD_INIT(0)
	0,                         /*ob_size*/
	"isa.GSM",                 /*tp_name*/
	sizeof(GSMObject),         /*tp_basicsize*/
	0,                         /*tp_itemsize*/
	(destructor)GSM_dealloc,   /*tp_dealloc*/
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
	GSM_methods,               /*tp_methods*/
	0,                         /*tp_members*/
	GSM_getset,                /*tp_getset*/
	0,                         /*tp_base*/
	0,                         /*tp_dict*/
	0,                         /*tp_descr_get*/
	0,                         /*tp_descr_set*/
	0,                         /*tp_dictoffset*/
	(initproc)GSM_init,        /*tp_init*/
	0,                         /*tp_alloc*/
	GSM_new,                   /*tp_new*/
};

#endif
