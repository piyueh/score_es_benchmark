#! /usr/bin/env python3
# vim:fenc=utf-8
#
# Copyright © 2023 Pi-Yueh Chuang <pychuang@pm.me>
#
# Distributed under terms of the BSD 3-Clause license.

"""C implementations (front end).
"""
import os
import sys
import sysconfig
import importlib
import subprocess
import pathlib
import numpy

_root = pathlib.Path(__file__).resolve().parent


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


def compile_c_ext(cc, optname, optflags):
    """Compile C extensions and return the module.
    """

    macro = f"{cc.upper()}_{optname.upper()}"
    ext = _root.joinpath(f"c_impl_{macro}.so")
    cmd = [
        cc, "-fPIC", "-shared", f"-D{macro}", "-march=native", "-funroll-loops",
        f"-I{sysconfig.get_config_var('INCLUDEPY')}",
        f"-I{numpy.get_include()}",
        f"-o", f"{ext}",
        f"{_root.joinpath('c_impl_backend.c')}"
    ]
    cmd.extend(optflags)

    print(f"Compiling {macro}")
    res = subprocess.run(cmd, check=False, capture_output=True)

    if res.returncode != 0:
        print(res.stderr.decode("utf-8"))
        print(res.stdout.decode("utf-8"))
        raise RuntimeError(f"{cc} build failed")

    module = _import_helper(f"c_impl_{macro}", ext)
    return module

# build and import extentions
_c_impl_GCC_O3 = compile_c_ext("gcc", "O3", ["-O3"])
score_es_gcc_o3_v1 = getattr(_c_impl_GCC_O3, "score_es_general")
score_es_gcc_o3_v2 = getattr(_c_impl_GCC_O3, "score_es")
score_es_gcc_o3_v3 = getattr(_c_impl_GCC_O3, "score_es_contiguous")

_c_impl_GCC_FASTMATH = compile_c_ext("gcc", "FASTMATH", ["-Ofast", "-ffast-math"])
score_es_gcc_fastmath_v1 = getattr(_c_impl_GCC_FASTMATH, "score_es_general")
score_es_gcc_fastmath_v2 = getattr(_c_impl_GCC_FASTMATH, "score_es")
score_es_gcc_fastmath_v3 = getattr(_c_impl_GCC_FASTMATH, "score_es_contiguous")

_c_impl_CLANG_O3 = compile_c_ext("clang", "O3", ["-O3"])
score_es_clang_o3_v1 = getattr(_c_impl_CLANG_O3, "score_es_general")
score_es_clang_o3_v2 = getattr(_c_impl_CLANG_O3, "score_es")
score_es_clang_o3_v3 = getattr(_c_impl_CLANG_O3, "score_es_contiguous")

_c_impl_CLANG_FASTMATH = compile_c_ext("clang", "FASTMATH", ["-Ofast", "-ffast-math"])
score_es_clang_fastmath_v1 = getattr(_c_impl_CLANG_FASTMATH, "score_es_general")
score_es_clang_fastmath_v2 = getattr(_c_impl_CLANG_FASTMATH, "score_es")
score_es_clang_fastmath_v3 = getattr(_c_impl_CLANG_FASTMATH, "score_es_contiguous")

if __name__ == "__main__":
    profiler = getattr(_import_helper("profiler", _root.parent.joinpath("profiler.py")), "profiler")

    result = profiler(score_es_gcc_o3_v1, "score_es_gcc_o3_v1", n_rep=1)
    result = profiler(score_es_gcc_o3_v2, "score_es_gcc_o3_v2", n_rep=1)
    result = profiler(score_es_gcc_o3_v3, "score_es_gcc_o3_v3", n_rep=1)
    result = profiler(score_es_gcc_fastmath_v1, "score_es_gcc_fastmath_v1", n_rep=1)
    result = profiler(score_es_gcc_fastmath_v2, "score_es_gcc_fastmath_v2", n_rep=1)
    result = profiler(score_es_gcc_fastmath_v3, "score_es_gcc_fastmath_v3", n_rep=1)
    result = profiler(score_es_clang_o3_v1, "score_es_clang_o3_v1", n_rep=1)
    result = profiler(score_es_clang_o3_v2, "score_es_clang_o3_v2", n_rep=1)
    result = profiler(score_es_clang_o3_v3, "score_es_clang_o3_v3", n_rep=1)
    result = profiler(score_es_clang_fastmath_v1, "score_es_clang_fastmath_v1", n_rep=1)
    result = profiler(score_es_clang_fastmath_v2, "score_es_clang_fastmath_v2", n_rep=1)
    result = profiler(score_es_clang_fastmath_v3, "score_es_clang_fastmath_v3", n_rep=1)
