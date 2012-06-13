#ifndef ISAINTERFACE_H
#define ISAINTERFACE_H

#define PY_ARRAY_UNIQUE_SYMBOL ISA_ARRAY_API
#define NO_IMPORT_ARRAY

#include <Python.h>
#include <arrayobject.h>
#include "isa.h"

struct ISAObject {
	PyObject_HEAD
	ISA* isa;
};

ISA::Parameters PyObject_ToParameters(ISAObject* self, PyObject* parameters);

PyObject* ISA_new(PyTypeObject* type, PyObject* args, PyObject* kwds);
int ISA_init(ISAObject* self, PyObject* args, PyObject* kwds);
void ISA_dealloc(ISAObject* self);

PyObject* ISA_dim(ISAObject* self, PyObject*, void*);
PyObject* ISA_num_visibles(ISAObject* self, PyObject*, void*);
PyObject* ISA_num_hiddens(ISAObject* self, PyObject*, void*);

PyObject* ISA_A(ISAObject* self, PyObject*, void*);
int ISA_set_A(ISAObject* self, PyObject* value, void*);

PyObject* ISA_basis(ISAObject* self, PyObject*, PyObject*);
PyObject* ISA_set_basis(ISAObject* self, PyObject* args, PyObject* kwds);
PyObject* ISA_nullspace_basis(ISAObject* self, PyObject* args, PyObject* kwds);

PyObject* ISA_subspaces(ISAObject* self, PyObject*, PyObject*);
PyObject* ISA_set_subspaces(ISAObject* self, PyObject* args, PyObject* kwds);

PyObject* ISA_default_parameters(ISAObject* self);

PyObject* ISA_initialize(ISAObject* self, PyObject* args, PyObject* kwds);
PyObject* ISA_train(ISAObject* self, PyObject* args, PyObject* kwds);

PyObject* ISA_sample(ISAObject* self, PyObject* args, PyObject* kwds);
PyObject* ISA_sample_prior(ISAObject* self, PyObject* args, PyObject* kwds);
PyObject* ISA_sample_nullspace(ISAObject* self, PyObject* args, PyObject* kwds);
PyObject* ISA_sample_posterior(ISAObject* self, PyObject* args, PyObject* kwds);
PyObject* ISA_sample_scales(ISAObject* self, PyObject* args, PyObject* kwds);

PyObject* ISA_prior_energy(ISAObject* self, PyObject* args, PyObject* kwds);
PyObject* ISA_prior_energy_gradient(ISAObject* self, PyObject* args, PyObject* kwds);
PyObject* ISA_loglikelihood(ISAObject* self, PyObject* args, PyObject* kwds);

#endif
