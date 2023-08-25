#! /usr/bin/env python3
# vim:fenc=utf-8
#
# Copyright © 2023 Pi-Yueh Chuang <pychuang@pm.me>
#
# Distributed under terms of the BSD 3-Clause license.

"""Test
"""
import numpy
from impls import original
from impls import c_v1
from impls import c_v2
from impls import c_v3
from impls import c_v4

for i in [1, 2, 6, 12]:
    a = numpy.random.rand(10, i)
    b = numpy.random.rand(10, i)
    res = all([
        original.score_es_original_numpy(a, b),
        original.score_es_original_scipy(a, b),
        original.score_es_original_numba(a, b),
        c_v1.score_es_c_v1_numba(a, b),
        c_v1.score_es_c_v1_gcc(a, b),
        c_v1.score_es_c_v1_clang(a, b),
        c_v2.score_es_c_v2_numba(a, b),
        c_v2.score_es_c_v2_gcc(a, b),
        c_v2.score_es_c_v2_clang(a, b),
        c_v3.score_es_c_v3_numba(a, b),
        c_v3.score_es_c_v3_gcc(a, b),
        c_v3.score_es_c_v3_clang(a, b),
        c_v4.score_es_c_v4_numba(a, b),
        c_v4.score_es_c_v4_gcc(a, b),
        c_v4.score_es_c_v4_clang(a, b),
    ])
    print(f"n_cols: {i}: {'PASSED' if res else 'FAILED'}")
