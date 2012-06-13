#include "callbacktrain.h"

CallbackTrain::CallbackTrain(ISAObject* isa, PyObject* callback) : 
	mIsa(isa), 
	mCallback(callback) 
{
	Py_INCREF(mIsa);
	Py_INCREF(mCallback);
}



CallbackTrain::CallbackTrain(const CallbackTrain& callbackTrain) :
	mIsa(callbackTrain.mIsa),
	mCallback(callbackTrain.mCallback)
{
	Py_INCREF(mIsa);
	Py_INCREF(mCallback);
}



CallbackTrain::~CallbackTrain() {
	Py_DECREF(mIsa);
	Py_DECREF(mCallback);
}



CallbackTrain& CallbackTrain::operator=(const CallbackTrain& callbackTrain) {
	Py_DECREF(mIsa);
	Py_DECREF(mCallback);

	mIsa = callbackTrain.mIsa;
	mCallback = callbackTrain.mCallback;

	Py_INCREF(mIsa);
	Py_INCREF(mCallback);

	return *this;
}



CallbackTrain* CallbackTrain::copy() {
	return new CallbackTrain(*this);
}



bool CallbackTrain::operator()(int iter, const ISA&) {
	// call Python object
	PyObject* args = Py_BuildValue("(iO)", iter, mIsa);
	PyObject* result = PyObject_CallObject(mCallback, args);

	Py_DECREF(args);

	// if cont is false, training will be aborted
	bool cont = true;
	if(result) {
		if(PyBool_Check(result))
			cont = (result == Py_True);
		Py_DECREF(result);
	} else {
		throw Exception("Some error occured during callback().");
	}

	return cont;
}
