#! /usr/bin/env python3
# vim:fenc=utf-8

"""C v4
"""
import math
import numba
from .c_v4_gcc import score_es_c_v4 as score_es_c_v4_gcc
from .c_v4_clang import score_es_c_v4 as score_es_c_v4_clang


@numba.njit(
    [
        numba.float64(numba.float64[:, ::1], numba.float64[:, ::1]),
        numba.float32(numba.float32[:, ::1], numba.float32[:, ::1]),
    ],
    fastmath=True, nogil=True, boundscheck=False
)
def _score_es_c_v4_numba_1(preds, obsrvs):
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


@numba.njit(
    [
        numba.float64(numba.float64[:, ::1], numba.float64[:, ::1]),
        numba.float32(numba.float32[:, ::1], numba.float32[:, ::1]),
    ],
    fastmath=True, nogil=True, boundscheck=False
)
def _score_es_c_v4_numba_2(preds, obsrvs):
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


@numba.njit(
    [
        numba.float64(numba.float64[:, ::1], numba.float64[:, ::1]),
        numba.float32(numba.float32[:, ::1], numba.float32[:, ::1]),
    ],
    fastmath=True, nogil=True, boundscheck=False
)
def _score_es_c_v4_numba_6(preds, obsrvs):
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


@numba.njit(
    [
        numba.float64(numba.float64[:, ::1], numba.float64[:, ::1]),
        numba.float32(numba.float32[:, ::1], numba.float32[:, ::1]),
    ],
    fastmath=True, nogil=True, boundscheck=False
)
def _score_es_c_v4_numba_12(preds, obsrvs):
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


def score_es_c_v4_numba(preds, obsrvs):
    """Wrapper for Numba v4.
    """
    event_size = obsrvs.shape[1]

    assert preds.flags["C_CONTIGUOUS"]
    assert obsrvs.flags["C_CONTIGUOUS"]

    match event_size:
        case 1:
            return _score_es_c_v4_numba_1(preds, obsrvs)
        case 2:
            return _score_es_c_v4_numba_2(preds, obsrvs)
        case 6:
            return _score_es_c_v4_numba_6(preds, obsrvs)
        case 12:
            return _score_es_c_v4_numba_12(preds, obsrvs)
        case _:
            raise ValueError(f"Event size {event_size} is not supported.")
