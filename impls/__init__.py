#! /usr/bin/env python3
# vim:fenc=utf-8
#
# Copyright © 2023 Pi-Yueh Chuang <pychuang@pm.me>
#
# Distributed under terms of the BSD 3-Clause license.

"""__init__.py
"""
from .original import score_es_original_numpy
from .original import score_es_original_scipy
from .original import score_es_original_numba
from .c_v1 import score_es_c_v1_numba
from .c_v1 import score_es_c_v1_gcc
from .c_v1 import score_es_c_v1_clang
from .c_v2 import score_es_c_v2_numba
from .c_v2 import score_es_c_v2_gcc
from .c_v2 import score_es_c_v2_clang
from .c_v3 import score_es_c_v3_numba
from .c_v3 import score_es_c_v3_gcc
from .c_v3 import score_es_c_v3_clang
from .c_v4 import score_es_c_v4_numba
from .c_v4 import score_es_c_v4_gcc
from .c_v4 import score_es_c_v4_clang

options = {
    ("Original", "NumPy"): score_es_original_numpy,
    ("Original", "SciPy"): score_es_original_scipy,
    ("Original", "Numba"): score_es_original_numba,
    ("C v1", "Numba"): score_es_c_v1_numba,
    ("C v1", "GCC"): score_es_c_v1_gcc,
    ("C v1", "Clang"): score_es_c_v1_clang,
    ("C v2", "Numba"): score_es_c_v2_numba,
    ("C v2", "GCC"): score_es_c_v2_gcc,
    ("C v2", "Clang"): score_es_c_v2_clang,
    ("C v3", "Numba"): score_es_c_v3_numba,
    ("C v3", "GCC"): score_es_c_v3_gcc,
    ("C v3", "Clang"): score_es_c_v3_clang,
    ("C v4", "Numba"): score_es_c_v4_numba,
    ("C v4", "GCC"): score_es_c_v4_gcc,
    ("C v4", "Clang"): score_es_c_v4_clang,
}
