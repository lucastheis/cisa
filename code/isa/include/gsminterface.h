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

PyObject* GSM_new(PyTypeObject* type, PyObject* args, PyObject* kwds);
int GSM_init(GSMObject* self, PyObject* args, PyObject* kwds);
void GSM_dealloc(GSMObject* self);

PyObject* GSM_dim(GSMObject* self, PyObject*, void*);
PyObject* GSM_num_scales(GSMObject* self, PyObject*, void*);

PyObject* GSM_scales(GSMObject* self, PyObject*, void*);
int GSM_set_scales(GSMObject* self, PyObject* value, void*);

PyObject* GSM_variance(GSMObject* self, PyObject*, PyObject*);
PyObject* GSM_normalize(GSMObject* self, PyObject*, PyObject*);

PyObject* GSM_train(GSMObject* self, PyObject* args, PyObject* kwds);

PyObject* GSM_posterior(GSMObject* self, PyObject* args, PyObject* kwds);

PyObject* GSM_sample(GSMObject* self, PyObject* args, PyObject* kwds);
PyObject* GSM_sample_posterior(GSMObject* self, PyObject* args, PyObject* kwds);

PyObject* GSM_loglikelihood(GSMObject* self, PyObject* args, PyObject* kwds);
PyObject* GSM_energy(GSMObject* self, PyObject* args, PyObject* kwds);
PyObject* GSM_energy_gradient(GSMObject* self, PyObject* args, PyObject* kwds);

#endif
