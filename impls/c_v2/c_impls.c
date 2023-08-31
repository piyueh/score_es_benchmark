#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <Python.h>
#include <numpy/arrayobject.h>
#include <stdio.h>
#include <assert.h>
#include <xmmintrin.h>
#include <pmmintrin.h>


double _score_es_1(
    const double ** const x, const double ** const y,
    const int n1, const int n2
) {
    double score_1 = 0.0, score_2 = 0.0;
    const double **xi = NULL, **yj = NULL;
    const double **xend = x + n1, **yend = y + n2;

    for (xi = x; xi < xend; xi++) {
        for (yj = y; yj < yend; yj++)
            score_1 += Py_ABS(**xi - **yj);
        for (yj = xi + 1; yj < xend; yj++)
            score_2 += Py_ABS(**xi - **yj);
    }

    // separate divisions to avoid stack overflow; they are int not long...
    score_1 /= (double) n1;
    score_1 /= (double) n2;
    score_2 /= (double) n1;
    score_2 /= (double) (n1 - 1);

    return score_1 - score_2;
}

double _score_es_2(
    const double ** const x, const double ** const y,
    const int n1, const int n2
) {
    double score_1 = 0.0, score_2 = 0.0;
    const double **xi = NULL, **yj = NULL;
    const double **xend = x + n1, **yend = y + n2;
    double tmp = 0.0, norm = 0.0;
    int k = 0;

    for (xi = x; xi < xend; xi++) {
        for (yj = y; yj < yend; yj++) {
            norm = 0.0;
            for (k = 0; k < 2; ++k) {
                tmp = (*(*xi + k) - *(*yj + k));
                tmp *= tmp;
                norm += tmp;
            }
            score_1 += sqrt(norm);
        }
        for (yj = xi + 1; yj < xend; yj++) {
            norm = 0.0;
            for (k = 0; k < 2; ++k) {
                tmp = (*(*xi + k) - *(*yj + k));
                tmp *= tmp;
                norm += tmp;
            }
            score_2 += sqrt(norm);
        }
    }

    // separate divisions to avoid stack overflow; they are int not long...
    score_1 /= (double) n1;
    score_1 /= (double) n2;
    score_2 /= (double) n1;
    score_2 /= (double) (n1 - 1);

    return score_1 - score_2;
}

double _score_es_6(
    const double ** const x, const double ** const y,
    const int n1, const int n2
) {
    double score_1 = 0.0, score_2 = 0.0;
    const double **xi = NULL, **yj = NULL;
    const double **xend = x + n1, **yend = y + n2;
    double tmp = 0.0, norm = 0.0;
    int k = 0;

    for (xi = x; xi < xend; xi++) {
        for (yj = y; yj < yend; yj++) {
            norm = 0.0;
            for (k = 0; k < 6; ++k) {
                tmp = (*(*xi + k) - *(*yj + k));
                tmp *= tmp;
                norm += tmp;
            }
            score_1 += sqrt(norm);
        }
        for (yj = xi + 1; yj < xend; yj++) {
            norm = 0.0;
            for (k = 0; k < 6; ++k) {
                tmp = (*(*xi + k) - *(*yj + k));
                tmp *= tmp;
                norm += tmp;
            }
            score_2 += sqrt(norm);
        }
    }

    // separate divisions to avoid stack overflow; they are int not long...
    score_1 /= (double) n1;
    score_1 /= (double) n2;
    score_2 /= (double) n1;
    score_2 /= (double) (n1 - 1);

    return score_1 - score_2;
}

double _score_es_12(
    const double ** const x, const double ** const y,
    const int n1, const int n2
) {
    double score_1 = 0.0, score_2 = 0.0;
    const double **xi = NULL, **yj = NULL;
    const double **xend = x + n1, **yend = y + n2;
    double tmp = 0.0, norm = 0.0;
    int k = 0;

    for (xi = x; xi < xend; xi++) {
        for (yj = y; yj < yend; yj++) {
            norm = 0.0;
            for (k = 0; k < 12; ++k) {
                tmp = (*(*xi + k) - *(*yj + k));
                tmp *= tmp;
                norm += tmp;
            }
            score_1 += sqrt(norm);
        }
        for (yj = xi + 1; yj < xend; yj++) {
            norm = 0.0;
            for (k = 0; k < 12; ++k) {
                tmp = (*(*xi + k) - *(*yj + k));
                tmp *= tmp;
                norm += tmp;
            }
            score_2 += sqrt(norm);
        }
    }

    // separate divisions to avoid stack overflow; they are int not long...
    score_1 /= (double) n1;
    score_1 /= (double) n2;
    score_2 /= (double) n1;
    score_2 /= (double) (n1 - 1);

    return score_1 - score_2;
}

PyObject *score_es_c_v2(PyObject *self, PyObject *args) {

    double result = 0.0;  // final value
    PyArrayObject *vec_1 = NULL, *vec_2 = NULL;  // pointers to ndarray objects
    int ndim_1 = -1, ndim_2 = -1;  // dimensions
    npy_intp *shape_1 = NULL, *shape_2 = NULL;  // shapes
    double **data_1 = NULL, **data_2 = NULL;  // underlying c array

    // convert args to multiple PyObject and save to arr
    PyArg_ParseTuple(args, "OO", &vec_1, &vec_2);
    if (PyErr_Occurred()) return NULL;  // errors occurred

    // check if the two arguments are legit numpy arrays
    if (! PyArray_Check(vec_1)) {
        PyErr_SetString(PyExc_TypeError, "vec_1 must be a numpy array");
        return NULL;
    }

    if (! PyArray_Check(vec_2)) {
        PyErr_SetString(PyExc_TypeError, "vec_2 must be a numpy array");
        return NULL;
    }

    ndim_1 = PyArray_NDIM(vec_1);
    ndim_2 = PyArray_NDIM(vec_2);
    assert(ndim_1 == 2);
    assert(ndim_2 == 2);

    shape_1 = PyArray_SHAPE(vec_1);
    shape_2 = PyArray_SHAPE(vec_2);
    assert(shape_1[1] == shape_2[1]);

    // get the C array of the numpy (auto convert dtype & make contiguous)
    PyArray_AsCArray(
        (PyObject **) &vec_1, (void *) &data_1, shape_1, ndim_1,
        PyArray_DescrFromType(NPY_DOUBLE)
    );
    if (PyErr_Occurred()) return NULL;

    PyArray_AsCArray(
        (PyObject **) &vec_2, (void *) &data_2, shape_2, ndim_2,
        PyArray_DescrFromType(NPY_DOUBLE)
    );
    if (PyErr_Occurred()) return NULL;

    // calculation
    switch(shape_1[1]) {
        case 1:
            result = _score_es_1(data_1, data_2, shape_1[0], shape_2[0]);
            break;
        case 2:
            result = _score_es_2(data_1, data_2, shape_1[0], shape_2[0]);
            break;
        case 6:
            result = _score_es_6(data_1, data_2, shape_1[0], shape_2[0]);
            break;
        case 12:
            result = _score_es_12(data_1, data_2, shape_1[0], shape_2[0]);
            break;
        default:
            PyErr_SetString(
                PyExc_NotImplementedError,
                "Not implemented for event size != 1, 2, 6, or 12"
            );
            return NULL;
    }

    // free the pointers
    PyArray_Free((PyObject *) vec_1, (void *) data_1);
    PyArray_Free((PyObject *) vec_2, (void *) data_2);
    if (PyErr_Occurred()) return NULL;

    return PyFloat_FromDouble(result);
}

PyDoc_STRVAR(doc, "score_es_c_v2(vec_1, vec_2, /)");

static PyMethodDef methods[] = {
    {"score_es_c_v2", score_es_c_v2, METH_VARARGS, doc},
    {NULL, NULL, 0, NULL}
};

#if defined(CLANG)
static struct PyModuleDef c_impl_backend = {
    PyModuleDef_HEAD_INIT,
    "c_v2_clang",
    "C implementation of c v2",
    -1,
    methods
};

PyMODINIT_FUNC PyInit_c_v2_clang() {
    _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_OFF);
    _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_OFF);

    PyObject *module = PyModule_Create(&c_impl_backend);
    import_array();
    return module;
}
#else
static struct PyModuleDef c_impl_backend = {
    PyModuleDef_HEAD_INIT,
    "c_v2_gcc",
    "C implementation of c v2",
    -1,
    methods
};

PyMODINIT_FUNC PyInit_c_v2_gcc() {
    _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_OFF);
    _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_OFF);

    PyObject *module = PyModule_Create(&c_impl_backend);
    import_array();
    return module;
}
#endif
