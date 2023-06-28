#! /usr/bin/env python3
# vim:fenc=utf-8
#
# Copyright © 2023 Pi-Yueh Chuang <pychuang@pm.me>
#
# Distributed under terms of the BSD 3-Clause license.

"""Original score_es
"""
import numpy
import math
from numba import njit


@njit(fastmath=True, nogil=True, boundscheck=False)
def score_es_numba_v1(preds, obsrvs):
    """Naive Numba impl. Credit: Niteya

    Notes
    -----
    We tried to mimic the original Python code as much as possible. However, the original NumPy's
    norm function accepts an argument `axis`, while Numba's own norm implementation does not.
    We therefore had to modified the code somehow.
    """
    obs_size, obs_event = obsrvs.shape
    pred_size, pred_event = preds.shape
    assert obs_event == pred_event, "The second dimensions do not match"

    score_1 = 0.0
    score_2 = 0.0
    for i in range(pred_size):
        for j in range(obs_size):
            score_1 += numpy.linalg.norm(preds[i] - obsrvs[j])

        for j in range(i+1, pred_size):
            score_2 += numpy.linalg.norm(preds[i] - preds[j])

    score_1 /= (obs_size * pred_size)
    score_2 /= (pred_size * (pred_size - 1))

    return score_1 - score_2


@njit(fastmath=True, nogil=True, boundscheck=False)
def score_es_numba_v2(preds, obsrvs):
    """Specialized version of JIT score_es for shape[1] = 2.
    """
    obs_size, obs_event = obsrvs.shape
    pred_size, pred_event = preds.shape
    assert obs_event == pred_event, "The second dimensions do not match"

    score_1 = 0.0
    score_2 = 0.0
    for i in range(pred_size):
        for j in range(obs_size):
            norm = 0.0
            for k in range(obs_event):
                tmp = preds[i, k] - obsrvs[j, k]
                norm += tmp * tmp
            score_1 += math.sqrt(norm)

        for j in range(i+1, pred_size):
            norm = 0.0
            for k in range(obs_event):
                tmp = preds[i, k] - preds[j, k]
                norm += tmp * tmp
            score_2 += math.sqrt(norm)

    score_1 /= (obs_size * pred_size)
    score_2 /= (pred_size * (pred_size - 1))

    return score_1 - score_2


@njit(fastmath=True, nogil=True, boundscheck=False)
def _score_es_numba_v3_1(preds, obsrvs):
    """Specialized version of JIT score_es for shape[1] = 1.
    """
    obs_size, obs_event = obsrvs.shape
    pred_size, pred_event = preds.shape
    assert obs_event == 1, "The second dimension of observations must be 1"
    assert pred_event == 1, "The second dimension of predictions must be 1"

    score_1 = 0.0
    score_2 = 0.0
    for i in range(pred_size):
        for j in range(obs_size):
            score_1 += abs(preds[i, 0] - obsrvs[j, 0])

        for j in range(i+1, pred_size):
            score_2 += abs(preds[i, 0] - preds[j, 0])

    score_1 /= (obs_size * pred_size)
    score_2 /= (pred_size * (pred_size - 1))

    return score_1 - score_2


@njit(fastmath=True, nogil=True, boundscheck=False)
def _score_es_numba_v3_2(preds, obsrvs):
    """Specialized version of JIT score_es for shape[1] = 2.
    """
    obs_size, obs_event = obsrvs.shape
    pred_size, pred_event = preds.shape
    assert obs_event == 2, "The second dimension of observations must be 2"
    assert pred_event == 2, "The second dimension of predictions must be 2"

    score_1 = 0.0
    score_2 = 0.0
    for i in range(pred_size):
        for j in range(obs_size):
            norm = 0.0
            for k in range(2):
                tmp = preds[i, k] - obsrvs[j, k]
                tmp *= tmp
                norm += tmp
            score_1 += math.sqrt(norm)

        for j in range(i+1, pred_size):
            norm = 0.0
            for k in range(2):
                tmp = preds[i, k] - preds[j, k]
                tmp *= tmp
                norm += tmp
            score_2 += math.sqrt(norm)

    score_1 /= (obs_size * pred_size)
    score_2 /= (pred_size * (pred_size - 1))

    return score_1 - score_2


@njit(fastmath=True, nogil=True, boundscheck=False)
def _score_es_numba_v3_6(preds, obsrvs):
    """Specialized version of JIT score_es for shape[1] = 6.
    """
    obs_size, obs_event = obsrvs.shape
    pred_size, pred_event = preds.shape
    assert obs_event == 6, "The second dimension of observations must be 6"
    assert pred_event == 6, "The second dimension of predictions must be 6"

    score_1 = 0.0
    score_2 = 0.0
    for i in range(pred_size):
        for j in range(obs_size):
            norm = 0.0
            for k in range(6):
                tmp = preds[i, k] - obsrvs[j, k]
                tmp *= tmp
                norm += tmp
            score_1 += math.sqrt(norm)

        for j in range(i+1, pred_size):
            norm = 0.0
            for k in range(6):
                tmp = preds[i, k] - preds[j, k]
                tmp *= tmp
                norm += tmp
            score_2 += math.sqrt(norm)

    score_1 /= (obs_size * pred_size)
    score_2 /= (pred_size * (pred_size - 1))

    return score_1 - score_2


@njit(fastmath=True, nogil=True, boundscheck=False)
def _score_es_numba_v3_12(preds, obsrvs):
    """Specialized version of JIT score_es for shape[1] = 12.
    """
    obs_size, obs_event = obsrvs.shape
    pred_size, pred_event = preds.shape
    assert obs_event == 12, "The second dimension of observations must be 12"
    assert pred_event == 12, "The second dimension of predictions must be 12"

    score_1 = 0.0
    score_2 = 0.0
    for i in range(pred_size):
        for j in range(obs_size):
            norm = 0.0
            for k in range(12):
                tmp = preds[i, k] - obsrvs[j, k]
                tmp *= tmp
                norm += tmp
            score_1 += math.sqrt(norm)

        for j in range(i+1, pred_size):
            norm = 0.0
            for k in range(12):
                tmp = preds[i, k] - preds[j, k]
                tmp *= tmp
                norm += tmp
            score_2 += math.sqrt(norm)

    score_1 /= (obs_size * pred_size)
    score_2 /= (pred_size * (pred_size - 1))

    return score_1 - score_2


@njit(fastmath=True, nogil=True, boundscheck=False)
def score_es_numba_v3(preds, obsrvs):
    """Wrapper for Numba v3.
    """
    event_size = obsrvs.shape[1]

    match event_size:
        case 1:
            return _score_es_numba_v3_1(preds, obsrvs)
        case 2:
            return _score_es_numba_v3_2(preds, obsrvs)
        case 6:
            return _score_es_numba_v3_6(preds, obsrvs)
        case 12:
            return _score_es_numba_v3_12(preds, obsrvs)
        case _:
            raise ValueError(f"Event size {event_size} is not supported.")


if __name__ == "__main__":
    import sys
    import pathlib
    import importlib

    def _import_helper(module_name, filepath):
        """Import a module using its file path.

        Arguments
        ---------
        module_name : str
            The module name that will be registered in sys.modules.
        filepath : str or os.PathLike
            The path to the source file.

        Returns
        -------
        module : a Python module
        """
        spec = importlib.util.spec_from_file_location(module_name, filepath)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module

    root = pathlib.Path(__file__).resolve().parent
    profiler = getattr(_import_helper("profiler", root.parent.joinpath("profiler.py")), "profiler")

    result = profiler(score_es_numba_v1, "score_es_numba_v1", n_rep=1)
    result = profiler(score_es_numba_v2, "score_es_numba_v2", n_rep=1)
    result = profiler(score_es_numba_v3, "score_es_numba_v3", n_rep=1)
