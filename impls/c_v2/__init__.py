#! /usr/bin/env python3
# vim:fenc=utf-8
#
# Copyright © 2023 Pi-Yueh Chuang <pychuang@pm.me>
#
# Distributed under terms of the BSD 3-Clause license.

"""C v2
"""
import math
from numba import njit
from .c_v2_gcc import score_es_c_v2 as score_es_c_v2_gcc
from .c_v2_clang import score_es_c_v2 as score_es_c_v2_clang


@njit(nogil=True, boundscheck=False)
def _score_es_c_v2_numba_1(preds, obsrvs):
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


@njit(nogil=True, boundscheck=False)
def _score_es_c_v2_numba_2(preds, obsrvs):
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


@njit(nogil=True, boundscheck=False)
def _score_es_c_v2_numba_6(preds, obsrvs):
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


@njit(nogil=True, boundscheck=False)
def _score_es_c_v2_numba_12(preds, obsrvs):
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


@njit(nogil=True, boundscheck=False)
def score_es_c_v2_numba(preds, obsrvs):
    """Wrapper for Numba v3.
    """
    event_size = obsrvs.shape[1]

    match event_size:
        case 1:
            return _score_es_c_v2_numba_1(preds, obsrvs)
        case 2:
            return _score_es_c_v2_numba_2(preds, obsrvs)
        case 6:
            return _score_es_c_v2_numba_6(preds, obsrvs)
        case 12:
            return _score_es_c_v2_numba_12(preds, obsrvs)
        case _:
            raise ValueError(f"Event size {event_size} is not supported.")
