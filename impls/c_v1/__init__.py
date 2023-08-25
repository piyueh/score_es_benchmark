#! /usr/bin/env python3
# vim:fenc=utf-8
#
# Copyright © 2023 Pi-Yueh Chuang <pychuang@pm.me>
#
# Distributed under terms of the BSD 3-Clause license.

"""C v1
"""
import math
from numba import njit
from .c_v1_gcc import score_es_c_v1 as score_es_c_v1_gcc
from .c_v1_clang import score_es_c_v1 as score_es_c_v1_clang


@njit(nogil=True, boundscheck=False)
def score_es_c_v1_numba(preds, obsrvs):
    """C v1 implementation using Numba.
    """
    obs_size, obs_event = obsrvs.shape
    pred_size, pred_event = preds.shape

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
