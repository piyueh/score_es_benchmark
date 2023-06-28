#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <Python.h>
#include <numpy/arrayobject.h>
#include <stdio.h>
#include <assert.h>
#include <xmmintrin.h>
#include <pmmintrin.h>

// TODO: seriously consider changing int to long for sizes

double _score_es_general(
    const double ** const x, const double ** const y,
    const int n1, const int n2, const int ncols
) {
    double score_1 = 0.0, score_2 = 0.0;
    const double **xi = NULL, **yj = NULL;
    const double **xend = x + n1, **yend = y + n2;
    const double *xik = NULL, *yjk = NULL;
    const double *xik_end = NULL;
    double tmp = 0.0, norm = 0.0;

    for (xi = x; xi < xend; xi++) {
        xik_end = *xi + ncols;
        for (yj = y; yj < yend; yj++) {
            norm = 0.0;
            for (xik = *xi, yjk = *yj; xik < xik_end; xik++, yjk++) {
                tmp = *xik - *yjk;
                tmp *= tmp;
                norm += tmp;
            }
            score_1 += sqrt(norm);
        }
        for (yj = xi + 1; yj < xend; yj++) {
            norm = 0.0;
            for (xik = *xi, yjk = *yj; xik < xik_end; xik++, yjk++) {
                tmp = *xik - *yjk;
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

double _score_es_1_contiguous(
    const double * const x, const double * const y,
    const int n1, const int n2
) {
    double score_1 = 0.0, score_2 = 0.0;
    const double *xi = NULL, *yj = NULL;
    const double *xend = x + n1, *yend = y + n2;

    for (xi = x; xi < xend; xi++) {
        for (yj = y; yj< yend; yj++)
            score_1 += Py_ABS(*xi - *yj);
        for (yj = xi + 1; yj < xend; yj++)
            score_2 += Py_ABS(*xi - *yj);
    }

    // separate divisions to avoid stack overflow; they are int not long...
    score_1 /= (double) n1;
    score_1 /= (double) n2;
    score_2 /= (double) n1;
    score_2 /= (double) (n1 - 1);

    return score_1 - score_2;
}

double _score_es_2_contiguous(
    const double * const x, const double * const y,
    const int n1, const int n2
) {
    double score_1 = 0.0, score_2 = 0.0;
    const double *xi = NULL, *yj = NULL;
    const double *xend = x + n1 * 2, *yend = y + n2 * 2;
    double tmp = 0.0, norm = 0.0;
    int k = 0;

    for (xi = x; xi < xend; xi+=2) {
        for (yj = y; yj < yend; yj+=2) {
            norm = 0.0;
            for (k = 0; k < 2; k++) {
                tmp = (*(xi + k) - *(yj + k));
                tmp *= tmp;
                norm += tmp;
            }
            score_1 += sqrt(norm);
        }
        for (yj = xi + 2; yj < xend; yj+=2) {
            norm = 0.0;
            for (k = 0; k < 2; k++) {
                tmp = (*(xi + k) - *(yj + k));
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

double _score_es_6_contiguous(
    const double * const x, const double * const y,
    const int n1, const int n2
) {
    double score_1 = 0.0, score_2 = 0.0;
    const double *xi = NULL, *yj = NULL;
    const double *xend = x + n1 * 6, *yend = y + n2 * 6;
    double tmp = 0.0, norm = 0.0;
    int k = 0;

    for (xi = x; xi < xend; xi+=6) {
        for (yj = y; yj < yend; yj+=6) {
            norm = 0.0;
            for (k = 0; k < 6; k++) {
                tmp = (*(xi + k) - *(yj + k));
                tmp *= tmp;
                norm += tmp;
            }
            score_1 += sqrt(norm);
        }
        for (yj = xi + 6; yj < xend; yj+=6) {
            norm = 0.0;
            for (k = 0; k < 6; k++) {
                tmp = (*(xi + k) - *(yj + k));
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

double _score_es_12_contiguous(
    const double * const x, const double * const y,
    const int n1, const int n2
) {
    double score_1 = 0.0, score_2 = 0.0;
    const double *xi = NULL, *yj = NULL;
    const double *xend = x + n1 * 12, *yend = y + n2 * 12;
    double tmp = 0.0, norm = 0.0;
    int k = 0;

    for (xi = x; xi < xend; xi+=12) {
        for (yj = y; yj < yend; yj+=12) {
            norm = 0.0;
            for (k = 0; k < 12; k++) {
                tmp = (*(xi + k) - *(yj + k));
                tmp *= tmp;
                norm += tmp;
            }
            score_1 += sqrt(norm);
        }
        for (yj = xi + 12; yj < xend; yj+=12) {
            norm = 0.0;
            for (k = 0; k < 12; k++) {
                tmp = (*(xi + k) - *(yj + k));
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

PyObject *score_es_general(PyObject *self, PyObject *args) {

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
    result = _score_es_general(data_1, data_2, shape_1[0], shape_2[0], shape_2[1]);

    // free the pointers
    PyArray_Free((PyObject *) vec_1, (void *) data_1);
    PyArray_Free((PyObject *) vec_2, (void *) data_2);
    if (PyErr_Occurred()) return NULL;

    return PyFloat_FromDouble(result);
}

PyObject *score_es(PyObject *self, PyObject *args) {

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

PyObject *score_es_contiguous(PyObject *self, PyObject *args) {

    double result = 0.0;  // final value
    PyArrayObject *vec_1 = NULL, *vec_2 = NULL;  // pointers to ndarray objects
    npy_intp *shape_1 = NULL, *shape_2 = NULL;  // shapes
    double *data_1 = NULL, *data_2 = NULL;  // underlying c array

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

    if (PyArray_TYPE(vec_1) != NPY_DOUBLE) {
        PyErr_SetString(PyExc_TypeError, "vec_1's dtype is not 64bit floats");
        return NULL;
    }

    if (PyArray_TYPE(vec_2) != NPY_DOUBLE) {
        PyErr_SetString(PyExc_TypeError, "vec_2's dtype is not 64bit floats");
        return NULL;
    }

    if (! PyArray_IS_C_CONTIGUOUS(vec_1)) {
        PyErr_SetString(PyExc_TypeError, "vec_1 is not C-contiguous");
        return NULL;
    }

    if (! PyArray_IS_C_CONTIGUOUS(vec_2)) {
        PyErr_SetString(PyExc_TypeError, "vec_2 is not C-contiguous");
        return NULL;
    }

    assert(PyArray_NDIM(vec_1) == 2);
    assert(PyArray_NDIM(vec_2) == 2);

    shape_1 = PyArray_SHAPE(vec_1);
    shape_2 = PyArray_SHAPE(vec_2);
    assert(shape_1[1] == shape_2[1]);

    // get the C array of the numpy (auto convert dtype & make contiguous)
    data_1 = PyArray_DATA(vec_1);
    if (PyErr_Occurred()) return NULL;

    data_2 = PyArray_DATA(vec_2);
    if (PyErr_Occurred()) return NULL;

    switch(shape_1[1]) {
        case 1:
            result = _score_es_1_contiguous(data_1, data_2, shape_1[0], shape_2[0]);
            break;
        case 2:
            result = _score_es_2_contiguous(data_1, data_2, shape_1[0], shape_2[0]);
            break;
        case 6:
            result = _score_es_6_contiguous(data_1, data_2, shape_1[0], shape_2[0]);
            break;
        case 12:
            result = _score_es_12_contiguous(data_1, data_2, shape_1[0], shape_2[0]);
            break;
        default:
            PyErr_SetString(
                PyExc_NotImplementedError,
                "Not implemented for event size != 1, 2, 6, or 12"
            );
            return NULL;
    }

    return PyFloat_FromDouble(result);
}

PyDoc_STRVAR(
    score_es_general_doc,
    "score_es_general(vec_1, vec_2, /)\n"
    "--\n\n"
    "Calculate the summed pairwise Euclidean distance of two 2D arrays. "
    "No assumption on the memory contiguity and the number of columns."

    "Arguments\n"
    "---------\n"
    "vec_1, vec_2 : 2D numpy.ndarray of shapes (M1, N) and (M2, N)\n\n"
    "Returns\n"
    "-------\n"
    "score : float\n\n"
    "Notes\n"
    "-----\n"
    "The type of score is Python native float, and it's usually 8 bytes."
);

PyDoc_STRVAR(
    score_es_doc,
    "score_es(vec_1, vec_2, /)\n"
    "--\n\n"
    "Calculate the summed pairwise Euclidean distance of two 2D arrays. "
    "Currently, only arrays of which the second dimensions of the arrays are 1 "
    "or 2 are supported.\n\n" 
    "This function does not need the memory space to be contiguous for the "
    "input arrays. Nor does this function need the input arrays's dtypes to be "
    "double-precision floats (64bits). However, it hence suffers overhead for "
    "making sure the contiguous memory and float precision internally.\n\n"
    "Arguments\n"
    "---------\n"
    "vec_1, vec_2 : 2D numpy.ndarray of shapes (M1, N) and (M2, N)\n\n"
    "Returns\n"
    "-------\n"
    "score : float\n\n"
    "Notes\n"
    "-----\n"
    "The type of score is Python native float, and it's usually 8 bytes."
);

PyDoc_STRVAR(
    score_es_contiguous_doc,
    "score_es_contiguous(vec_1, vec_2, /)\n"
    "--\n\n"
    "Calculate the summed pairwise Euclidean distance of two 2D arrays. "
    "Currently, only arrays of which the second dimensions of the arrays are 1 "
    "or 2 are supported.\n\n" 
    "The memory space of the input arrays must be C-contiguous, and the "
    "underlying floats must be double-precision (64bits).\n\n"
    "Arguments\n"
    "---------\n"
    "vec_1, vec_2 : 2D numpy.ndarray of shapes (M1, N) and (M2, N)\n\n"
    "Returns\n"
    "-------\n"
    "score : float\n\n"
    "Notes\n"
    "-----\n"
    "The type of score is Python native float, and it's usually 8 bytes."
);

static PyMethodDef methods[] = {
    {"score_es_general", score_es_general, METH_VARARGS, score_es_general_doc},
    {"score_es", score_es, METH_VARARGS, score_es_doc},
    {"score_es_contiguous", score_es_contiguous, METH_VARARGS, score_es_contiguous_doc},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef c_impl_backend = {
    PyModuleDef_HEAD_INIT,
    "score_es_c",
    "C implementation of score_es",
    -1,
    methods
};

#if defined(GCC_O3)
PyMODINIT_FUNC PyInit_c_impl_GCC_O3() {
    _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_OFF);
    _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_OFF);

    PyObject *module = PyModule_Create(&c_impl_backend);
    import_array();
    return module;
}
#elif defined(GCC_FASTMATH)
PyMODINIT_FUNC PyInit_c_impl_GCC_FASTMATH() {
    _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_OFF);
    _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_OFF);

    PyObject *module = PyModule_Create(&c_impl_backend);
    import_array();
    return module;
}
#elif defined(CLANG_O3)
PyMODINIT_FUNC PyInit_c_impl_CLANG_O3() {
    _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_OFF);
    _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_OFF);

    PyObject *module = PyModule_Create(&c_impl_backend);
    import_array();
    return module;
}
#elif defined(CLANG_FASTMATH)
PyMODINIT_FUNC PyInit_c_impl_CLANG_FASTMATH() {
    _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_OFF);
    _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_OFF);

    PyObject *module = PyModule_Create(&c_impl_backend);
    import_array();
    return module;
}
#endif
