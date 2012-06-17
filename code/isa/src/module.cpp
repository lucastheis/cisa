#define PY_ARRAY_UNIQUE_SYMBOL ISA_ARRAY_API

#include <Python.h>
#include <arrayobject.h>
#include <structmember.h>
#include <stdlib.h>
#include <time.h>
#include "isainterface.h"
#include "gsminterface.h"

static PyGetSetDef ISA_getset[] = {
	{"dim", (getter)ISA_dim, 0, 0},
	{"num_visibles", (getter)ISA_num_visibles, 0, 0},
	{"num_hiddens", (getter)ISA_num_hiddens, 0, 0},
	{"A", (getter)ISA_A, (setter)ISA_set_A, 0},
	{0}
};



static PyMethodDef ISA_methods[] = {
	{"default_parameters", (PyCFunction)ISA_default_parameters, METH_VARARGS, 0},
	{"basis", (PyCFunction)ISA_basis, METH_NOARGS, 0},
	{"set_basis", (PyCFunction)ISA_set_basis, METH_VARARGS|METH_KEYWORDS, 0},
	{"nullspace_basis", (PyCFunction)ISA_nullspace_basis, METH_NOARGS, 0},
	{"subspaces", (PyCFunction)ISA_subspaces, METH_NOARGS, 0},
	{"set_subspaces", (PyCFunction)ISA_set_subspaces, METH_VARARGS|METH_KEYWORDS, 0},
	{"initialize", (PyCFunction)ISA_initialize, METH_VARARGS|METH_KEYWORDS, 0},
	{"orthogonalize", (PyCFunction)ISA_orthogonalize, METH_NOARGS, 0},
	{"train", (PyCFunction)ISA_train, METH_VARARGS|METH_KEYWORDS, 0},
	{"sample", (PyCFunction)ISA_sample, METH_VARARGS|METH_KEYWORDS, 0},
	{"sample_prior", (PyCFunction)ISA_sample_prior, METH_VARARGS|METH_KEYWORDS, 0},
	{"sample_nullspace", (PyCFunction)ISA_sample_nullspace, METH_VARARGS|METH_KEYWORDS, 0},
	{"sample_posterior", (PyCFunction)ISA_sample_posterior, METH_VARARGS|METH_KEYWORDS, 0},
	{"sample_posterior_ais", (PyCFunction)ISA_sample_posterior_ais, METH_VARARGS|METH_KEYWORDS, 0},
	{"sample_scales", (PyCFunction)ISA_sample_scales, METH_VARARGS|METH_KEYWORDS, 0},
	{"matching_pursuit", (PyCFunction)ISA_matching_pursuit, METH_VARARGS|METH_KEYWORDS, 0},
	{"prior_energy", (PyCFunction)ISA_prior_energy, METH_VARARGS|METH_KEYWORDS, 0},
	{"prior_energy_gradient", (PyCFunction)ISA_prior_energy_gradient, METH_VARARGS|METH_KEYWORDS, 0},
	{"loglikelihood", (PyCFunction)ISA_loglikelihood, METH_VARARGS|METH_KEYWORDS, 0},
	{"evaluate", (PyCFunction)ISA_evaluate, METH_VARARGS|METH_KEYWORDS, 0},
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



static PyGetSetDef GSM_getset[] = {
	{"dim", (getter)GSM_dim, 0, 0},
	{"num_scales", (getter)GSM_num_scales, 0, 0},
	{"scales", (getter)GSM_scales, (setter)GSM_set_scales, 0},
	{0}
};



static PyMethodDef GSM_methods[] = {
	{"train", (PyCFunction)GSM_train, METH_VARARGS|METH_KEYWORDS, 0},
	{"posterior", (PyCFunction)GSM_posterior, METH_VARARGS|METH_KEYWORDS, 0},
	{"variance", (PyCFunction)GSM_variance, METH_NOARGS, 0},
	{"normalize", (PyCFunction)GSM_normalize, METH_NOARGS, 0},
	{"sample", (PyCFunction)GSM_sample, METH_VARARGS|METH_KEYWORDS, 0},
	{"sample_posterior", (PyCFunction)GSM_sample_posterior, METH_VARARGS|METH_KEYWORDS, 0},
	{"loglikelihood", (PyCFunction)GSM_loglikelihood, METH_VARARGS|METH_KEYWORDS, 0},
	{"energy", (PyCFunction)GSM_energy, METH_VARARGS|METH_KEYWORDS, 0},
	{"energy_gradient", (PyCFunction)GSM_energy_gradient, METH_VARARGS|METH_KEYWORDS, 0},
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

	// add types to module
	Py_INCREF(&ISA_type);
	PyModule_AddObject(module, "ISA", reinterpret_cast<PyObject*>(&ISA_type));
	Py_INCREF(&GSM_type);
	PyModule_AddObject(module, "GSM", reinterpret_cast<PyObject*>(&GSM_type));
}
