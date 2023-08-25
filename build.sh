#! /bin/sh
#
# build.sh
# Copyright (C) 2023 Pi-Yueh Chuang <pychuang@pm.me>
#
# Distributed under terms of the BSD 3-Clause license.
#


# c v1
# =============================================================================
gcc \
    -I${CONDA_PREFIX}/include/python3.10 \
    -I$(python -c "import numpy; print(numpy.get_include())") \
    -fPIC \
    -shared \
    -march=native \
    -O3 \
    -o ./impls/c_v1/c_v1_gcc.so \
    ./impls/c_v1/c_impls.c

clang \
    -D CLANG \
    -I${CONDA_PREFIX}/include/python3.10 \
    -I$(python -c "import numpy; print(numpy.get_include())") \
    -fPIC \
    -shared \
    -march=native \
    -O3 \
    -o ./impls/c_v1/c_v1_clang.so \
    ./impls/c_v1/c_impls.c


# c v2
# =============================================================================
gcc \
    -I${CONDA_PREFIX}/include/python3.10 \
    -I$(python -c "import numpy; print(numpy.get_include())") \
    -fPIC \
    -shared \
    -march=native \
    -O3 \
    -o ./impls/c_v2/c_v2_gcc.so \
    ./impls/c_v2/c_impls.c

clang \
    -D CLANG \
    -I${CONDA_PREFIX}/include/python3.10 \
    -I$(python -c "import numpy; print(numpy.get_include())") \
    -fPIC \
    -shared \
    -march=native \
    -O3 \
    -o ./impls/c_v2/c_v2_clang.so \
    ./impls/c_v2/c_impls.c


# c v3
# =============================================================================
gcc \
    -I${CONDA_PREFIX}/include/python3.10 \
    -I$(python -c "import numpy; print(numpy.get_include())") \
    -fPIC \
    -shared \
    -march=native \
    -Ofast -funroll-loops -ffast-math \
    -o ./impls/c_v3/c_v3_gcc.so \
    ./impls/c_v3/c_impls.c

clang \
    -D CLANG \
    -I${CONDA_PREFIX}/include/python3.10 \
    -I$(python -c "import numpy; print(numpy.get_include())") \
    -fPIC \
    -shared \
    -march=native \
    -O3 \
    -o ./impls/c_v3/c_v3_clang.so \
    ./impls/c_v3/c_impls.c


# c v4
# =============================================================================
gcc \
    -I${CONDA_PREFIX}/include/python3.10 \
    -I$(python -c "import numpy; print(numpy.get_include())") \
    -fPIC \
    -shared \
    -march=native \
    -Ofast -funroll-loops -ffast-math \
    -o ./impls/c_v4/c_v4_gcc.so \
    ./impls/c_v4/c_impls.c

clang \
    -D CLANG \
    -I${CONDA_PREFIX}/include/python3.10 \
    -I$(python -c "import numpy; print(numpy.get_include())") \
    -fPIC \
    -shared \
    -march=native \
    -Ofast -funroll-loops -ffast-math \
    -o ./impls/c_v4/c_v4_clang.so \
    ./impls/c_v4/c_impls.c
