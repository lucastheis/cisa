#ifndef CALLBACKTRAIN_H
#define CALLBACKTRAIN_H

#include <Python.h>
#include "isa.h"

struct ISAObject;

class CallbackTrain : public ISA::Callback {
	public:
		CallbackTrain(ISAObject* isa, PyObject* callback);
		CallbackTrain(const CallbackTrain& callbackTrain);
		virtual ~CallbackTrain();
		virtual CallbackTrain& operator=(const CallbackTrain& callbackTrain);
		virtual CallbackTrain* copy();
		virtual bool operator()(int iter, const ISA&);

	private:
		ISAObject* mIsa;
		PyObject* mCallback;
};

#endif
