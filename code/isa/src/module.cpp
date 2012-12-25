#define PY_ARRAY_UNIQUE_SYMBOL ISA_ARRAY_API

#include <Python.h>
#include <arrayobject.h>
#include <structmember.h>
#include <stdlib.h>
#include <time.h>
#include "isainterface.h"
#include "gsminterface.h"
#include "Eigen/Core"

static PyGetSetDef ISA_getset[] = {
	{"dim", (getter)ISA_dim, 0, "The number of visible units."},
	{"num_visibles", (getter)ISA_num_visibles, 0, "The number of visible units."},
	{"num_hiddens", (getter)ISA_num_hiddens, 0, "The number of hidden units."},
	{"A", (getter)ISA_A, (setter)ISA_set_A, "The current basis of the model."},
	{0}
};



static PyMethodDef ISA_methods[] = {
	{"default_parameters", (PyCFunction)ISA_default_parameters, METH_VARARGS, ISA_default_parameters_doc},
	{"basis", (PyCFunction)ISA_basis, METH_NOARGS, ISA_basis_doc},
	{"set_basis", (PyCFunction)ISA_set_basis, METH_VARARGS|METH_KEYWORDS, ISA_set_basis_doc},
	{"hidden_states", (PyCFunction)ISA_hidden_states, METH_NOARGS, ISA_hidden_states_doc},
	{"set_hidden_states", (PyCFunction)ISA_set_hidden_states, METH_VARARGS|METH_KEYWORDS, ISA_set_hidden_states_doc},
	{"nullspace_basis", (PyCFunction)ISA_nullspace_basis, METH_NOARGS, ISA_nullspace_basis_doc},
	{"subspaces", (PyCFunction)ISA_subspaces, METH_NOARGS, ISA_subspaces_doc},
	{"set_subspaces", (PyCFunction)ISA_set_subspaces, METH_VARARGS|METH_KEYWORDS, ISA_set_subspaces_doc},
	{"initialize", (PyCFunction)ISA_initialize, METH_VARARGS|METH_KEYWORDS, ISA_initialize_doc},
	{"orthogonalize", (PyCFunction)ISA_orthogonalize, METH_NOARGS, ISA_orthogonalize_doc},
	{"train", (PyCFunction)ISA_train, METH_VARARGS|METH_KEYWORDS, ISA_train_doc},
	{"sample", (PyCFunction)ISA_sample, METH_VARARGS|METH_KEYWORDS, ISA_sample_doc},
	{"sample_prior", (PyCFunction)ISA_sample_prior, METH_VARARGS|METH_KEYWORDS, ISA_sample_prior_doc},
	{"sample_nullspace", (PyCFunction)ISA_sample_nullspace, METH_VARARGS|METH_KEYWORDS, ISA_sample_nullspace_doc},
	{"sample_posterior", (PyCFunction)ISA_sample_posterior, METH_VARARGS|METH_KEYWORDS, ISA_sample_posterior_doc},
	{"sample_posterior_ais", (PyCFunction)ISA_sample_posterior_ais, METH_VARARGS|METH_KEYWORDS, ISA_sample_posterior_ais_doc},
	{"sample_scales", (PyCFunction)ISA_sample_scales, METH_VARARGS|METH_KEYWORDS, ISA_sample_scales_doc},
	{"sample_ais", (PyCFunction)ISA_sample_ais, METH_VARARGS|METH_KEYWORDS, ISA_sample_ais_doc},
	{"matching_pursuit", (PyCFunction)ISA_matching_pursuit, METH_VARARGS|METH_KEYWORDS, ISA_matching_pursuit_doc},
	{"prior_energy", (PyCFunction)ISA_prior_energy, METH_VARARGS|METH_KEYWORDS, ISA_prior_energy_doc},
	{"prior_energy_gradient", (PyCFunction)ISA_prior_energy_gradient, METH_VARARGS|METH_KEYWORDS, ISA_prior_energy_gradient_doc},
	{"prior_loglikelihood", (PyCFunction)ISA_prior_loglikelihood, METH_VARARGS|METH_KEYWORDS, ISA_prior_loglikelihood_doc},
	{"loglikelihood", (PyCFunction)ISA_loglikelihood, METH_VARARGS|METH_KEYWORDS, ISA_loglikelihood_doc},
	{"evaluate", (PyCFunction)ISA_evaluate, METH_VARARGS|METH_KEYWORDS, ISA_evaluate_doc},
	{"__reduce__", (PyCFunction)ISA_reduce, METH_NOARGS, 0},
	{"__setstate__", (PyCFunction)ISA_setstate, METH_VARARGS, 0},
	{0}
};



PyTypeObject ISA_type = {
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
	ISA_doc,                   /*tp_doc*/
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



static PyGetSetDef GSM_getset[] = {
	{"dim", (getter)GSM_dim, 0, "Dimensionality of the distribution."},
	{"num_scales", (getter)GSM_num_scales, 0, "Number of possible standard deviations."},
	{"scales", (getter)GSM_scales, (setter)GSM_set_scales, "Possible standard deviations."},
	{0}
};



static PyMethodDef GSM_methods[] = {
	{"train", (PyCFunction)GSM_train, METH_VARARGS|METH_KEYWORDS, GSM_train_doc},
	{"posterior", (PyCFunction)GSM_posterior, METH_VARARGS|METH_KEYWORDS, GSM_posterior_doc},
	{"variance", (PyCFunction)GSM_variance, METH_NOARGS, GSM_variance_doc},
	{"normalize", (PyCFunction)GSM_normalize, METH_NOARGS, GSM_normalize_doc},
	{"sample", (PyCFunction)GSM_sample, METH_VARARGS|METH_KEYWORDS, GSM_sample_doc},
	{"sample_posterior", (PyCFunction)GSM_sample_posterior, METH_VARARGS|METH_KEYWORDS, GSM_sample_posterior_doc},
	{"loglikelihood", (PyCFunction)GSM_loglikelihood, METH_VARARGS|METH_KEYWORDS, GSM_loglikelihood_doc},
	{"energy", (PyCFunction)GSM_energy, METH_VARARGS|METH_KEYWORDS, GSM_energy_doc},
	{"energy_gradient", (PyCFunction)GSM_energy_gradient, METH_VARARGS|METH_KEYWORDS, GSM_energy_gradient_doc},
	{"__reduce__", (PyCFunction)GSM_reduce, METH_NOARGS, 0},
	{"__setstate__", (PyCFunction)GSM_setstate, METH_VARARGS, 0},
	{0}
};



PyTypeObject GSM_type = {
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
	GSM_doc,                   /*tp_doc*/
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



PyMODINIT_FUNC initisa() {
	// set random seed
	timeval time;
	gettimeofday(&time, 0);
	srand(time.tv_usec * time.tv_sec);

	// initialize NumPy
	import_array();

	// create module object
	PyObject* module = Py_InitModule("isa", 0);

	// initialize types
	if(PyType_Ready(&ISA_type) < 0)
		return;
	if(PyType_Ready(&GSM_type) < 0)
		return;

	// initialize Eigen
	Eigen::initParallel();

	// add types to module
	Py_INCREF(&ISA_type);
	PyModule_AddObject(module, "ISA", reinterpret_cast<PyObject*>(&ISA_type));
	Py_INCREF(&GSM_type);
	PyModule_AddObject(module, "GSM", reinterpret_cast<PyObject*>(&GSM_type));
}
