#! /usr/bin/env python3
# vim:fenc=utf-8
#
# Copyright © 2023 Pi-Yueh Chuang <pychuang@pm.me>
#
# Distributed under terms of the BSD 3-Clause license.

"""__init__.py
"""
from .python_impl import score_es_original
from .python_impl import score_es_scipy
from .numba_impl import score_es_numba_v1
from .numba_impl import score_es_numba_v2
from .numba_impl import score_es_numba_v3
from .c_impl import score_es_gcc_o3_v1
from .c_impl import score_es_gcc_o3_v2
from .c_impl import score_es_gcc_o3_v3
from .c_impl import score_es_gcc_fastmath_v1
from .c_impl import score_es_gcc_fastmath_v2
from .c_impl import score_es_gcc_fastmath_v3
from .c_impl import score_es_clang_o3_v1
from .c_impl import score_es_clang_o3_v2
from .c_impl import score_es_clang_o3_v3
from .c_impl import score_es_clang_fastmath_v1
from .c_impl import score_es_clang_fastmath_v2
from .c_impl import score_es_clang_fastmath_v3

options = [
    "original",
    "scipy",
    "numba_v1",
    "numba_v2",
    "numba_v3",
    "gcc_o3_v1",
    "gcc_o3_v2",
    "gcc_o3_v3",
    "gcc_fastmath_v1",
    "gcc_fastmath_v2",
    "gcc_fastmath_v3",
    "clang_o3_v1",
    "clang_o3_v2",
    "clang_o3_v3",
    "clang_fastmath_v1",
    "clang_fastmath_v2",
    "clang_fastmath_v3",
]
