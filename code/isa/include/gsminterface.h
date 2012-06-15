#ifndef GSMINTERFACE_H
#define GSMINTERFACE_H

#define PY_ARRAY_UNIQUE_SYMBOL ISA_ARRAY_API
#define NO_IMPORT_ARRAY

#include <Python.h>
#include <arrayobject.h>
#include "gsm.h"
#include "pyutils.h"

struct GSMObject {
	PyObject_HEAD
	GSM* gsm;
};

extern PyTypeObject GSM_type;

PyObject* GSM_new(PyTypeObject*, PyObject*, PyObject*);
int GSM_init(GSMObject*, PyObject*, PyObject*);
void GSM_dealloc(GSMObject*);

PyObject* GSM_dim(GSMObject*, PyObject*, void*);
PyObject* GSM_num_scales(GSMObject*, PyObject*, void*);

PyObject* GSM_scales(GSMObject*, PyObject*, void*);
int GSM_set_scales(GSMObject*, PyObject*, void*);

PyObject* GSM_variance(GSMObject*, PyObject*, PyObject*);
PyObject* GSM_normalize(GSMObject*, PyObject*, PyObject*);

PyObject* GSM_train(GSMObject*, PyObject*, PyObject*);

PyObject* GSM_posterior(GSMObject*, PyObject*, PyObject*);

PyObject* GSM_sample(GSMObject*, PyObject*, PyObject*);
PyObject* GSM_sample_posterior(GSMObject*, PyObject*, PyObject*);

PyObject* GSM_loglikelihood(GSMObject*, PyObject*, PyObject*);
PyObject* GSM_energy(GSMObject*, PyObject*, PyObject*);
PyObject* GSM_energy_gradient(GSMObject*, PyObject*, PyObject*);

PyObject* GSM_reduce(GSMObject*, PyObject*, PyObject*);
PyObject* GSM_setstate(GSMObject*, PyObject*, PyObject*);

#endif
