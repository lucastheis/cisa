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

extern PyTypeObject ISA_type;

ISA::Parameters PyObject_ToParameters(ISAObject*, PyObject* parameters);

PyObject* ISA_new(PyTypeObject* type, PyObject*, PyObject*);
int ISA_init(ISAObject*, PyObject*, PyObject*);
void ISA_dealloc(ISAObject*);

PyObject* ISA_dim(ISAObject*, PyObject*, void*);
PyObject* ISA_num_visibles(ISAObject*, PyObject*, void*);
PyObject* ISA_num_hiddens(ISAObject*, PyObject*, void*);

PyObject* ISA_A(ISAObject*, PyObject*, void*);
int ISA_set_A(ISAObject*, PyObject* value, void*);

PyObject* ISA_basis(ISAObject*, PyObject*, PyObject*);
PyObject* ISA_set_basis(ISAObject*, PyObject*, PyObject*);
PyObject* ISA_nullspace_basis(ISAObject*, PyObject*, PyObject*);

PyObject* ISA_hidden_states(ISAObject*, PyObject*, PyObject*);
PyObject* ISA_set_hidden_states(ISAObject*, PyObject*, PyObject*);

PyObject* ISA_subspaces(ISAObject*, PyObject*, PyObject*);
PyObject* ISA_set_subspaces(ISAObject*, PyObject*, PyObject*);

PyObject* ISA_default_parameters(ISAObject*);

PyObject* ISA_initialize(ISAObject*, PyObject*, PyObject*);
PyObject* ISA_orthogonalize(ISAObject*, PyObject*, PyObject*);
PyObject* ISA_train(ISAObject*, PyObject*, PyObject*);

PyObject* ISA_sample(ISAObject*, PyObject*, PyObject*);
PyObject* ISA_sample_prior(ISAObject*, PyObject*, PyObject*);
PyObject* ISA_sample_nullspace(ISAObject*, PyObject*, PyObject*);
PyObject* ISA_sample_posterior(ISAObject*, PyObject*, PyObject*);
PyObject* ISA_sample_posterior_ais(ISAObject*, PyObject*, PyObject*);
PyObject* ISA_sample_scales(ISAObject*, PyObject*, PyObject*);
PyObject* ISA_sample_ais(ISAObject*, PyObject*, PyObject*);

PyObject* ISA_matching_pursuit(ISAObject*, PyObject*, PyObject*);

PyObject* ISA_prior_energy(ISAObject*, PyObject*, PyObject*);
PyObject* ISA_prior_energy_gradient(ISAObject*, PyObject*, PyObject*);
PyObject* ISA_loglikelihood(ISAObject*, PyObject*, PyObject*);
PyObject* ISA_evaluate(ISAObject*, PyObject*, PyObject*);

PyObject* ISA_reduce(ISAObject*, PyObject*, PyObject*);
PyObject* ISA_setstate(ISAObject*, PyObject*, PyObject*);

#endif
