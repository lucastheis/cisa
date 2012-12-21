#ifndef PYUTILS_H
#define PYUTILS_H

#define PY_ARRAY_UNIQUE_SYMBOL ISA_ARRAY_API
#define NO_IMPORT_ARRAY

#include <Python.h>
#include <arrayobject.h>
#include "Eigen/Core"

using namespace Eigen;

PyObject* PyArray_FromMatrixXf(const MatrixXf& mat);
MatrixXf PyArray_ToMatrixXf(PyObject* array);

#endif
