#include <Python.h>
#include <stdlib.h>

/* tensor definition */
typedef struct {
	PyObject_HEAD

	double *values;
	int length;
} Tensor;

static PyObject *newTensor(PyTypeObject *type, PyObject *args, PyObject *kwargs) {
	Tensor *self = (Tensor *) type->tp_alloc(type, 0);

	if (self != NULL) {

		if (!PyArg_ParseTuple(args, "i", &self->length)) {
			return NULL;
		}

		self->values = malloc(self->length * sizeof(double));
	}

	return (PyObject *) self;
}

static void deallocTensor(Tensor *self) {
	free(self->values);
	Py_TYPE(self)->tp_free((PyObject *) self);
}

static PyTypeObject TensorType = {
	PyVarObject_HEAD_INIT(NULL, 0)

	.tp_name = "minitorch.minitorch_c.Tensor",
	.tp_doc = "A cool tensor type",
	.tp_basicsize = sizeof(Tensor),
	.tp_itemsize = 0,
	.tp_flags = Py_TPFLAGS_DEFAULT,
	.tp_new = PyType_GenericNew,
	.tp_dealloc = (destructor) deallocTensor
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

	return module;
}