#include <Python.h>
#include <arrayobject.h>
#include <structmember.h>
#include <stdlib.h>
#include <time.h>
#include "isainterface.h"

PyMODINIT_FUNC initisa() {
	// set random seed
	timeval time;
	gettimeofday(&time, 0);
	srand(time.tv_usec * time.tv_sec);

	// initialize NumPy
	import_array();

	// create module object
	PyObject* module = Py_InitModule("isa", 0);

	// initialize type
	if(PyType_Ready(&ISA_type) < 0)
		return;
	
	// add type to module
	Py_INCREF(&ISA_type);
	PyModule_AddObject(module, "ISA", reinterpret_cast<PyObject*>(&ISA_type));
}
