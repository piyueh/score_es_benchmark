#! /usr/bin/env python3
# vim:fenc=utf-8
#
# Copyright © 2023 Pi-Yueh Chuang <pychuang@pm.me>
#
# Distributed under terms of the BSD 3-Clause license.

"""setup.py

Note that per pep621, most of the native Python subpackages and modules are
handled by pyproject.toml. This setup.py only handles C extensions.
"""
import os
import copy
import setuptools
import pybind11.setup_helpers
from concurrent.futures import ThreadPoolExecutor

class BuildExt(pybind11.setup_helpers.build_ext):
    """Customized build_ext.
    """

    def build_extensions(self) -> None:
        """Customize build_ext to handle the earth-mover differently.
        """
        specials = [
            ext for ext in self.extensions if "earth_mover_loss" in ext.name
        ]

        # remove the customized extensions
        for ext in specials:
            self.extensions.remove(ext)

        if self.parallel:
            if isinstance(self.parallel, bool):
                workers = os.cpu_count()
            else:
                workers = self.parallel

            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = [
                    executor.submit(self.earth_mover_builder, ext)
                    for ext in specials
                ]
                for ext, fut in zip(specials, futures):
                    with self._filter_build_errors(ext):
                        fut.result()
        else:
            for ext in specials:
                self.earth_mover_builder(ext)

        # build remaining extensions
        super().build_extensions()

        # add the removed extensions back
        self.extensions.extend(specials)

    def earth_mover_builder(self, ext) -> None:
        """Customized extension builder for the Earth-Mover loss.
        """

        # aloases
        log = setuptools.command.build_ext.log
        fullextpath = self.get_ext_fullpath(ext.name)
        outdir = self.build_temp
        incdirs = ext.include_dirs
        debug = self.debug

        # process macros and args
        macros = ext.define_macros + [(_,) for _ in ext.undef_macros]
        args = [_ for _ in ext.extra_compile_args if not _.startswith("-g")]
        args.append("-fPIC")

        # get a new compiler rather than using the auto-detected one
        if ("COMPILER", "GCC") in macros:
            compiler = copy.deepcopy(self.compiler)
            compiler.set_executable("preprocessor", None)
            compiler.set_executable("compiler", ["g++"])
            compiler.set_executable("compiler_so", ["g++"])
            compiler.set_executable("compiler_cxx", ["g++"])
            compiler.set_executable("linker_so", ["g++", "-shared"])
            compiler.set_executable("linker_exe", ["g++"])
            outdir = outdir + ".gcc"  # distinguish ourput dir
        elif ("COMPILER", "CLANG") in macros:
            compiler = copy.deepcopy(self.compiler)
            compiler.set_executable("preprocessor", None)
            compiler.set_executable("compiler", ["clang++"])
            compiler.set_executable("compiler_so", ["clang++"])
            compiler.set_executable("compiler_cxx", ["clang++"])
            compiler.set_executable("linker_so", ["clang++", "-shared"])
            compiler.set_executable("linker_exe", ["clang++"])
            outdir = outdir + ".clang"  # distinguish ourput dir
        else:
            raise RuntimeError("Must have the macro `COMPILER` (GCC or CLANG)")

        # hard-code source files and arguments for non-fast-math files
        nfm_srcs = [
            "quantom_prototype/envs/loss/earth_mover_loss/impls/c_impls.cpp",
            "quantom_prototype/envs/loss/earth_mover_loss/impls/c_impl_v1.cpp",
            "quantom_prototype/envs/loss/earth_mover_loss/impls/c_impl_v2.cpp",
            "quantom_prototype/envs/loss/earth_mover_loss/impls/c_impl_v3.cpp",
        ]

        nfm_args = args + ["-O3",]

        # hard-code source files and arguments for non-fast-math files
        fm_srcs = [
            "quantom_prototype/envs/loss/earth_mover_loss/impls/c_impl_v4.cpp",
        ]

        fm_args = args + ["-Ofast", "-ffast-math", "-funroll-loops"]

        if not (
            self.force or
            setuptools.dep_util.newer_group(nfm_srcs, fullextpath, 'newer') or
            setuptools.dep_util.newer_group(fm_srcs, fullextpath, 'newer')
        ):
            log.debug("skipping '%s' extension (up-to-date)", ext.name)
            return
        else:
            log.info("building '%s' extension", ext.name)

        # compile the source code to object files.
        nfm_objs = compiler.compile(
            nfm_srcs, outdir, macros, incdirs, debug, None, nfm_args)

        fm_objs = compiler.compile(
            fm_srcs, outdir, macros, incdirs, debug, None, fm_args)

        # linking
        compiler.link_shared_object(
            objects=nfm_objs+fm_objs,
            output_filename=fullextpath,
            libraries=self.get_libraries(ext),
            library_dirs=ext.library_dirs,
            runtime_library_dirs=ext.runtime_library_dirs,
            extra_postargs=ext.extra_link_args,
            export_symbols=self.get_export_symbols(ext),
            debug=debug,
            build_temp=outdir,
            target_lang="c++",
        )


exts = [
    pybind11.setup_helpers.Pybind11Extension(
        name="quantom_prototype.envs.loss.earth_mover_loss.impls.c_clang",
        sources=[],
        define_macros=[("COMPILER", "CLANG")],
        extra_compile_args=["-Wall", "-march=native"],
        language="c++", optional=False, py_limited_api=True,
        cxx_std=20, include_pybind11=True,
    ),
    pybind11.setup_helpers.Pybind11Extension(
        name="quantom_prototype.envs.loss.earth_mover_loss.impls.c_gcc",
        sources=[],
        define_macros=[("COMPILER", "GCC")],
        extra_compile_args=["-Wall", "-march=native"],
        language="c++", optional=False, py_limited_api=True,
        cxx_std=20, include_pybind11=True,
    ),
]

# enable parallel build within one extension
pybind11.setup_helpers.ParallelCompile(
    default=os.cpu_count()//2, max=os.cpu_count(),
).install()

setuptools.setup(ext_modules=exts, cmdclass={"build_ext": BuildExt})
