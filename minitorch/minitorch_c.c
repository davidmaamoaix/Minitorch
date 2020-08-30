#include <Python.h>
#include <stdlib.h>

/* tensor definition */

typedef struct {
	PyObject_HEAD

	double *values;
	int length;
} Tensor;

static PyObject *newTensor(PyTypeObject *type, PyObject *args, PyObject *kw) {
	Tensor *self = (Tensor *) type->tp_alloc(type, 0);

	return (PyObject *) self;
}

static int initTensor(Tensor *self, PyObject *args, PyObject *kw) {
	if (!PyArg_ParseTuple(args, "i", &self->length)) {
		return -1;
	}

	self->values = malloc(self->length * sizeof(double));

	return 0;
}

static void deallocTensor(Tensor *self) {
	free(self->values);
	Py_TYPE(self)->tp_free((PyObject *) self);
}


/* tensor properties */

static Py_ssize_t tensorLength(Tensor *self) {
	return self->length;
}

static PyObject *indexTensor(Tensor *self, Py_ssize_t i) {
	if (i >= self->length) {
		PyErr_Format(
			PyExc_IndexError,
			"Index %d is out of bounds of tensor with length %d",
			i, self->length
		);
		return NULL;
	}

	return PyFloat_FromDouble(self->values[i]);
}

static int assignTensor(Tensor *self, Py_ssize_t i, PyObject *value) {
	if (i >= self->length) {
		PyErr_Format(
			PyExc_IndexError,
			"Index %d is out of bounds of tensor with length %d",
			i, self->length
		);
		return -1;
	}

	self->values[i] = PyFloat_AsDouble(value);

	return 0;
}

static PyMethodDef tensorMethodTable[] = {
	{NULL, NULL, 0, NULL}
};

static PySequenceMethods tensorSequence = {
	.sq_length = (lenfunc) tensorLength,
	.sq_item = (ssizeargfunc) indexTensor,
	.sq_ass_item = (ssizeobjargproc) assignTensor
};

static PyTypeObject TensorType = {
	PyVarObject_HEAD_INIT(NULL, 0)

	.tp_name = "minitorch.minitorch_c.Tensor",
	.tp_doc = "A cool tensor type",
	.tp_basicsize = sizeof(Tensor),
	.tp_itemsize = 0,
	.tp_flags = Py_TPFLAGS_DEFAULT,
	.tp_new = newTensor,
	.tp_init = (initproc) initTensor,
	.tp_dealloc = (destructor) deallocTensor,
	.tp_methods = tensorMethodTable,
	.tp_as_sequence = &tensorSequence
};


/* tensor operations */

static PyObject *addNum(PyObject *self, PyObject *args) {
	double a, b;

	if (!PyArg_ParseTuple(args, "dd", &a, &b)) {
		return NULL;
	}

	return PyFloat_FromDouble(a + b);
}


/* module setup */

static PyMethodDef methodTable[] = {
	{"_add_num", addNum, METH_VARARGS, "Adds 2 nums"},
	{NULL, NULL, 0, NULL}
};

static struct PyModuleDef minitorch = {
	PyModuleDef_HEAD_INIT,
	"minitorch_c",
	"Submodule of minitorch that speeds stuff up",
	-1,
	methodTable
};

PyMODINIT_FUNC PyInit_minitorch_c() {
	PyObject *module = PyModule_Create(&minitorch);

	if (PyType_Ready(&TensorType) < 0) {
		return NULL;
	}

	Py_INCREF(&TensorType);
	if (PyModule_AddObject(module, "Tensor", (PyObject *) &TensorType) < 0) {
		Py_DECREF(&TensorType);
		Py_DECREF(module);
		return NULL;
	}

	return module;
}