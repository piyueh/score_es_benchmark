#! /usr/bin/env python3
# vim:fenc=utf-8
#
# Copyright © 2023 Pi-Yueh Chuang <pychuang@pm.me>
#
# Distributed under terms of the BSD 3-Clause license.

"""Unit tests.
"""
import pathlib
import unittest
import warnings
import numpy
import impls

warnings.simplefilter("error")


def func(self, kernel):
    for shape in self.shapes:
        cal = kernel(self.preds[shape], self.obsrvs[shape])
        print(shape, kernel.__name__, cal, self.ans[shape])
        self.assertAlmostEqual(self.ans[shape], cal, 10)


class TestScoreESGeneral(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.rng = numpy.random.default_rng()
        cls.shapes = [(1000, 1), (10000, 1), (10000, 2), (10000, 6), (10000, 12)]
        cls.preds = {shape: cls.rng.random(size=shape, dtype=numpy.float64) for shape in cls.shapes}
        cls.obsrvs = {shape: cls.rng.random(size=shape, dtype=numpy.float64) for shape in cls.shapes}
        cls.ans = {shape: impls.score_es_original(cls.preds[shape], cls.obsrvs[shape]) for shape in cls.shapes}

    def test_score_es_scipy(self):
        func(self, impls.score_es_scipy)

    def test_score_es_numba_v1(self):
        func(self, impls.score_es_numba_v1)

    def test_score_es_numba_v2(self):
        func(self, impls.score_es_numba_v2)

    def test_score_es_numba_v3(self):
        func(self, impls.score_es_numba_v3)

    def test_score_es_gcc_o3_v1(self):
        func(self, impls.score_es_gcc_o3_v1)

    def test_score_es_gcc_o3_v2(self):
        func(self, impls.score_es_gcc_o3_v2)

    def test_score_es_gcc_o3_v3(self):
        func(self, impls.score_es_gcc_o3_v3)

    def test_score_es_gcc_fastmath_v1(self):
        func(self, impls.score_es_gcc_fastmath_v1)

    def test_score_es_gcc_fastmath_v2(self):
        func(self, impls.score_es_gcc_fastmath_v2)

    def test_score_es_gcc_fastmath_v3(self):
        func(self, impls.score_es_gcc_fastmath_v3)

    def test_score_es_clang_o3_v1(self):
        func(self, impls.score_es_clang_o3_v1)

    def test_score_es_clang_o3_v2(self):
        func(self, impls.score_es_clang_o3_v2)

    def test_score_es_clang_o3_v3(self):
        func(self, impls.score_es_clang_o3_v3)

    def test_score_es_clang_fastmath_v1(self):
        func(self, impls.score_es_clang_fastmath_v1)

    def test_score_es_clang_fastmath_v2(self):
        func(self, impls.score_es_clang_fastmath_v2)

    def test_score_es_clang_fastmath_v3(self):
        func(self, impls.score_es_clang_fastmath_v3)


if __name__ == "__main__":
    unittest.main()
