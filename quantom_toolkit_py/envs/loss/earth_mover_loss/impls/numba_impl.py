#! /usr/bin/env python3
# vim:fenc=utf-8
#
# Copyright © 2023 Pi-Yueh Chuang <pychuang@pm.me>
#
# Distributed under terms of the BSD 3-Clause license.

"""Numba implementations for the Earth-Mover loss.
"""
import math
import numba
from numba import float32 as f32
from numba import float64 as f64
from numba import size_t


@numba.njit(nogil=True, boundscheck=False)
def earth_mover_v1(preds, obsrvs):
    """C v1 implementation using Numba.
    """

    np, ncols = preds.shape
    no = obsrvs.shape[0]
    score_1, score_2 = 0.0, 0.0

    if ncols == 1:
        for i in range(np):
            for j in range(no):
                score_1 += abs(preds[i, 0]-obsrvs[j, 0])

            for j in range(i+1, np):
                score_2 += abs(preds[i, 0]-preds[j, 0])
    else:
        for i in range(np):
            for j in range(no):
                norm = 0.0
                for k in range(ncols):
                    tmp = preds[i, k] - obsrvs[j, k]
                    norm += tmp * tmp
                score_1 += math.sqrt(norm)

            for j in range(i+1, np):
                norm = 0.0
                for k in range(ncols):
                    tmp = preds[i, k] - preds[j, k]
                    norm += tmp * tmp
                score_2 += math.sqrt(norm)

    score_1 /= no
    score_2 /= (np - 1)
    return (score_1 - score_2) / np

def earth_mover_v2(preds, obsrvs):
    """C v2 implementation using Numba.
    """
    ncols = preds.shape[1]

    try:
        return earth_mover_v2.kernels[ncols](preds, obsrvs)
    except KeyError:
        pass

    if ncols == 1:
        @numba.njit(nogil=True, boundscheck=False)
        def kernel(x, y):
            nx, ny = x.shape[0], y.shape[0]
            score_1, score_2 = 0.0, 0.0
            for i in range(nx):
                for j in range(ny):
                    score_1 += abs(x[i, 0]-y[j, 0])

                for j in range(i+1, nx):
                    score_2 += abs(x[i, 0]-x[j, 0])

            score_1 /= ny
            score_2 /= (nx - 1)
            return (score_1 - score_2) / nx

    else:
        @numba.njit(nogil=True, boundscheck=False)
        def kernel(x, y):
            nx, ny = x.shape[0], y.shape[0]
            score_1, score_2 = 0.0, 0.0
            for i in range(nx):
                for j in range(ny):
                    norm = 0.0
                    for k in range(ncols):
                        tmp = x[i, k] - y[j, k]
                        tmp *= tmp
                        norm += tmp
                    score_1 += math.sqrt(norm)

                for j in range(i+1, nx):
                    norm = 0.0
                    for k in range(ncols):
                        tmp = x[i, k] - x[j, k]
                        tmp *= tmp
                        norm += tmp
                    score_2 += math.sqrt(norm)

            score_1 /= ny
            score_2 /= (nx - 1)
            return (score_1 - score_2) / nx

    earth_mover_v2.kernels[ncols] = kernel
    return earth_mover_v2.kernels[ncols](preds, obsrvs)

earth_mover_v2.kernels = {}

def earth_mover_v3(preds, obsrvs):
    """C v3 implementation using Numba.
    """
    ncols = preds.shape[1]

    try:
        return earth_mover_v3.kernels[ncols](preds, obsrvs)
    except KeyError:
        pass

    if ncols == 1:
        @numba.njit([
            f64(f64[:, ::1], f64[:, ::1]),
            f32(f32[:, ::1], f32[:, ::1]),
        ], nogil=True, boundscheck=False)
        def kernel(x, y):
            nx, ny = x.shape[0], y.shape[0]
            score_1, score_2 = 0.0, 0.0
            for i in range(nx):
                for j in range(ny):
                    score_1 += abs(x[i, 0]-y[j, 0])

                for j in range(i+1, nx):
                    score_2 += abs(x[i, 0]-x[j, 0])

            score_1 /= ny
            score_2 /= (nx - 1)
            return (score_1 - score_2) / nx

    else:
        @numba.njit([
            f64(f64[:, ::1], f64[:, ::1]),
            f32(f32[:, ::1], f32[:, ::1]),
        ], nogil=True, boundscheck=False)
        def kernel(x, y):
            nx, ny = x.shape[0], y.shape[0]
            score_1, score_2 = 0.0, 0.0
            for i in range(nx):
                for j in range(ny):
                    norm = 0.0
                    for k in range(ncols):
                        tmp = x[i, k] - y[j, k]
                        tmp *= tmp
                        norm += tmp
                    score_1 += math.sqrt(norm)

                for j in range(i+1, nx):
                    norm = 0.0
                    for k in range(ncols):
                        tmp = x[i, k] - x[j, k]
                        tmp *= tmp
                        norm += tmp
                    score_2 += math.sqrt(norm)

            score_1 /= ny
            score_2 /= (nx - 1)
            return (score_1 - score_2) / nx

    earth_mover_v3.kernels[ncols] = kernel
    return earth_mover_v3.kernels[ncols](preds, obsrvs)

earth_mover_v3.kernels = {}

def earth_mover_v4(preds, obsrvs):
    """C v4 implementation using Numba.
    """
    ncols = preds.shape[1]

    try:
        return earth_mover_v4.kernels[ncols](preds, obsrvs)
    except KeyError:
        pass

    if ncols == 1:
        @numba.njit([
            f64(f64[:, ::1], f64[:, ::1]),
            f32(f32[:, ::1], f32[:, ::1]),
        ], fastmath=True, nogil=True, boundscheck=False)
        def kernel(x, y):
            nx, ny = x.shape[0], y.shape[0]
            score_1, score_2 = 0.0, 0.0
            for i in range(nx):
                for j in range(ny):
                    score_1 += abs(x[i, 0]-y[j, 0])

                for j in range(i+1, nx):
                    score_2 += abs(x[i, 0]-x[j, 0])

            score_1 /= ny
            score_2 /= (nx - 1)
            return (score_1 - score_2) / nx

    else:
        @numba.njit([
            f64(f64[:, ::1], f64[:, ::1]),
            f32(f32[:, ::1], f32[:, ::1]),
        ], fastmath=True, nogil=True, boundscheck=False)
        def kernel(x, y):
            nx, ny = x.shape[0], y.shape[0]
            score_1, score_2 = 0.0, 0.0
            for i in range(nx):
                for j in range(ny):
                    norm = 0.0
                    for k in range(ncols):
                        tmp = x[i, k] - y[j, k]
                        tmp *= tmp
                        norm += tmp
                    score_1 += math.sqrt(norm)

                for j in range(i+1, nx):
                    norm = 0.0
                    for k in range(ncols):
                        tmp = x[i, k] - x[j, k]
                        tmp *= tmp
                        norm += tmp
                    score_2 += math.sqrt(norm)

            score_1 /= ny
            score_2 /= (nx - 1)
            return (score_1 - score_2) / nx

    earth_mover_v4.kernels[ncols] = kernel
    return earth_mover_v4.kernels[ncols](preds, obsrvs)

earth_mover_v4.kernels = {}


@numba.njit([
    f64(f64, f64, f64, f64, f64, f64, f64, f64, f64, f64, f64, f64),
    f32(f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32),
], fastmath=True, nogil=True, boundscheck=False, inline="always")
def cdist_8x4_1d(x0, x1, x2, x3, x4, x5, x6, x7, y0, y1, y2, y3):
    """Pairwise Euclidean distance for length-8 vs length-4 vectors.
    """
    ans = 0.0
    ans += abs(x0-y0); ans += abs(x0-y1); ans += abs(x0-y2); ans += abs(x0-y3);
    ans += abs(x1-y0); ans += abs(x1-y1); ans += abs(x1-y2); ans += abs(x1-y3);
    ans += abs(x2-y0); ans += abs(x2-y1); ans += abs(x2-y2); ans += abs(x2-y3);
    ans += abs(x3-y0); ans += abs(x3-y1); ans += abs(x3-y2); ans += abs(x3-y3);
    ans += abs(x4-y0); ans += abs(x4-y1); ans += abs(x4-y2); ans += abs(x4-y3);
    ans += abs(x5-y0); ans += abs(x5-y1); ans += abs(x5-y2); ans += abs(x5-y3);
    ans += abs(x6-y0); ans += abs(x6-y1); ans += abs(x6-y2); ans += abs(x6-y3);
    ans += abs(x7-y0); ans += abs(x7-y1); ans += abs(x7-y2); ans += abs(x7-y3);
    return ans


@numba.njit([
    f64(f64, f64, f64, f64, f64, f64, f64, f64,
        f64, f64, f64, f64, f64, f64, f64, f64),
    f32(f32, f32, f32, f32, f32, f32, f32, f32,
        f32, f32, f32, f32, f32, f32, f32, f32),
], fastmath=True, nogil=True, boundscheck=False, inline="always")
def cdist_8x8_1d(
    x0, x1, x2, x3, x4, x5, x6, x7, y0, y1, y2, y3, y4, y5, y6, y7
):
    """Pairwise Euclidean distance for length-8 vs length-8 vectors.
    """

    ans = 0.0

    ans += abs(x0-y0); ans += abs(x0-y1); ans += abs(x0-y2); ans += abs(x0-y3);
    ans += abs(x0-y4); ans += abs(x0-y5); ans += abs(x0-y6); ans += abs(x0-y7);

    ans += abs(x1-y0); ans += abs(x1-y1); ans += abs(x1-y2); ans += abs(x1-y3);
    ans += abs(x1-y4); ans += abs(x1-y5); ans += abs(x1-y6); ans += abs(x1-y7);

    ans += abs(x2-y0); ans += abs(x2-y1); ans += abs(x2-y2); ans += abs(x2-y3);
    ans += abs(x2-y4); ans += abs(x2-y5); ans += abs(x2-y6); ans += abs(x2-y7);

    ans += abs(x3-y0); ans += abs(x3-y1); ans += abs(x3-y2); ans += abs(x3-y3);
    ans += abs(x3-y4); ans += abs(x3-y5); ans += abs(x3-y6); ans += abs(x3-y7);

    ans += abs(x4-y0); ans += abs(x4-y1); ans += abs(x4-y2); ans += abs(x4-y3);
    ans += abs(x4-y4); ans += abs(x4-y5); ans += abs(x4-y6); ans += abs(x4-y7);

    ans += abs(x5-y0); ans += abs(x5-y1); ans += abs(x5-y2); ans += abs(x5-y3);
    ans += abs(x5-y4); ans += abs(x5-y5); ans += abs(x5-y6); ans += abs(x5-y7);

    ans += abs(x6-y0); ans += abs(x6-y1); ans += abs(x6-y2); ans += abs(x6-y3);
    ans += abs(x6-y4); ans += abs(x6-y5); ans += abs(x6-y6); ans += abs(x6-y7);

    ans += abs(x7-y0); ans += abs(x7-y1); ans += abs(x7-y2); ans += abs(x7-y3);
    ans += abs(x7-y4); ans += abs(x7-y5); ans += abs(x7-y6); ans += abs(x7-y7);

    return ans


@numba.njit([
    f64(f64, f64, f64, f64, f64, f64, f64, f64),
    f32(f32, f32, f32, f32, f32, f32, f32, f32),
], fastmath=True, nogil=True, boundscheck=False, inline="always")
def pdist_8x8_1d(x0, x1, x2, x3, x4, x5, x6, x7):
    """Self-pairwise Euclidean distance for a length-8 vector.
    """
    ans = 0.0

    ans += abs(x0-x1); ans += abs(x0-x2); ans += abs(x0-x3); ans += abs(x0-x4);
    ans += abs(x0-x5); ans += abs(x0-x6); ans += abs(x0-x7);

    ans += abs(x1-x2); ans += abs(x1-x3); ans += abs(x1-x4); ans += abs(x1-x5);
    ans += abs(x1-x6); ans += abs(x1-x7);

    ans += abs(x2-x3); ans += abs(x2-x4); ans += abs(x2-x5); ans += abs(x2-x6);
    ans += abs(x2-x7);

    ans += abs(x3-x4); ans += abs(x3-x5); ans += abs(x3-x6); ans += abs(x3-x7);
    ans += abs(x4-x5); ans += abs(x4-x6); ans += abs(x4-x7);
    ans += abs(x5-x6); ans += abs(x5-x7);
    ans += abs(x6-x7);

    return ans


def earth_mover_v5(preds, obsrvs):
    """C v5 implementation using Numba.
    """
    ncols = preds.shape[1]

    # TODO: not optimized for ncols > 4

    try:
        return earth_mover_v5.kernels[ncols](preds, obsrvs)
    except KeyError:
        pass

    if ncols == 1:
        @numba.njit([
            f64(f64[:, ::1], f64[:, ::1]), f32(f32[:, ::1], f32[:, ::1]),
        ], fastmath=True, nogil=True, boundscheck=False)
        def kernel(x, y):
            nx, ny = x.shape[0], y.shape[0]
            score_1, score_2 = 0.0, 0.0

            nx8 = nx & -8  # = nx - nx % 8 = (nx // 8) * 8
            ny4 = ny & -4  # = ny - ny % 4 = (ny // 4) * 4
            for i in range(0, nx8, 8):
                x0, x1, x2, x3 = x[i, 0], x[i+1, 0], x[i+2, 0], x[i+3, 0]
                x4, x5, x6, x7 = x[i+4, 0], x[i+5, 0], x[i+6, 0], x[i+7, 0]

                for j in range(0, ny4, 4):
                    y0, y1, y2, y3 = y[j, 0], y[j+1, 0], y[j+2, 0], y[j+3, 0]

                    score_1 += cdist_8x4_1d(
                        x0, x1, x2, x3, x4, x5, x6, x7,
                        y0, y1, y2, y3
                    )

                score_2 += pdist_8x8_1d(x0, x1, x2, x3, x4, x5, x6, x7)

                for j in range(i+8, nx8, 8):
                    y0, y1, y2, y3 = x[j, 0], x[j+1, 0], x[j+2, 0], x[j+3, 0]
                    y4, y5, y6, y7 = x[j+4, 0], x[j+5, 0], x[j+6, 0], x[j+7, 0]

                    score_2 += cdist_8x8_1d(
                        x0, x1, x2, x3, x4, x5, x6, x7,
                        y0, y1, y2, y3, y4, y5, y6, y7,
                    )

            bcx = (nx != nx8)
            bcy = (ny != ny4)

            if bcx:
                for j in range(0, ny4, 4):
                    y0, y1, y2, y3 = y[j, 0], y[j+1, 0], y[j+2, 0], y[j+3, 0]
                    for i in range(nx8, nx):
                        score_1 += abs(x[i, 0]-y0); score_1 += abs(x[i, 0]-y1);
                        score_1 += abs(x[i, 0]-y2); score_1 += abs(x[i, 0]-y3);

                for i in range(0, nx8, 8):
                    x0, x1, x2, x3 = x[i, 0], x[i+1, 0], x[i+2, 0], x[i+3, 0]
                    x4, x5, x6, x7 = x[i+4, 0], x[i+5, 0], x[i+6, 0], x[i+7, 0]
                    for j in range(nx8, nx):
                        y0 = x[j, 0]
                        score_2 += abs(x0-y0); score_2 += abs(x1-y0);
                        score_2 += abs(x2-y0); score_2 += abs(x3-y0);
                        score_2 += abs(x4-y0); score_2 += abs(x5-y0);
                        score_2 += abs(x6-y0); score_2 += abs(x7-y0);

                for i in range(nx8, nx):
                    x0 = x[i, 0]
                    for j in range(i+1, nx):
                        score_2 += abs(x0-x[j, 0])

            if bcy:
                for i in range(0, nx8, 8):
                    x0, x1, x2, x3 = x[i, 0], x[i+1, 0], x[i+2, 0], x[i+3, 0]
                    x4, x5, x6, x7 = x[i+4, 0], x[i+5, 0], x[i+6, 0], x[i+7, 0]
                    for j in range(ny4, ny):
                        y0 = y[j, 0]
                        score_1 += abs(x0-y0); score_1 += abs(x1-y0);
                        score_1 += abs(x2-y0); score_1 += abs(x3-y0);
                        score_1 += abs(x4-y0); score_1 += abs(x5-y0);
                        score_1 += abs(x6-y0); score_1 += abs(x7-y0);

            if bcx and bcy:
                for i in range(nx8, nx):
                    x0 = x[i, 0]
                    for j in range(ny4, ny):
                        score_1 += abs(x0-y[j, 0])

            score_1 /= ny
            score_2 /= (nx - 1)
            return (score_1 - score_2) / nx

    else:
        @numba.njit([
            f64(f64[:, ::1], f64[:, ::1]), f32(f32[:, ::1], f32[:, ::1]),
        ], fastmath=True, nogil=True, boundscheck=False)
        def kernel(x, y):
            nx, ny = x.shape[0], y.shape[0]
            score_1, score_2 = 0.0, 0.0

            nx4 = nx & -4  # = nx - nx % 4 = (nx // 4) * 4
            ny4 = ny & -4  # = ny - ny % 4 = (ny // 4) * 4
            for i in range(0, nx4, 4):
                x0 = x[i, :]; x1 = x[i+1, :]; x2 = x[i+2, :]; x3 = x[i+3, :];
                for j in range(0, ny4, 4):
                    y0 = y[j, :]; y1 = y[j+1, :]; y2 = y[j+2, :]; y3 = y[j+3, :];

                    c00, c01, c02, c03 = 0.0, 0.0, 0.0, 0.0
                    c10, c11, c12, c13 = 0.0, 0.0, 0.0, 0.0
                    c20, c21, c22, c23 = 0.0, 0.0, 0.0, 0.0
                    c30, c31, c32, c33 = 0.0, 0.0, 0.0, 0.0

                    for k in range(ncols):
                        x0k, x1k, x2k, x3k = x0[k], x1[k], x2[k], x3[k]
                        y0k, y1k, y2k, y3k = y0[k], y1[k], y2[k], y3[k]

                        c00 += ((x0k-y0k)**2); c01 += ((x0k-y1k)**2);
                        c02 += ((x0k-y2k)**2); c03 += ((x0k-y3k)**2);
                        c10 += ((x1k-y0k)**2); c11 += ((x1k-y1k)**2);
                        c12 += ((x1k-y2k)**2); c13 += ((x1k-y3k)**2);
                        c20 += ((x2k-y0k)**2); c21 += ((x2k-y1k)**2);
                        c22 += ((x2k-y2k)**2); c23 += ((x2k-y3k)**2);
                        c30 += ((x3k-y0k)**2); c31 += ((x3k-y1k)**2);
                        c32 += ((x3k-y2k)**2); c33 += ((x3k-y3k)**2);

                    score_1 += (
                        math.sqrt(c00) + math.sqrt(c01) + math.sqrt(c02) + math.sqrt(c03) +
                        math.sqrt(c10) + math.sqrt(c11) + math.sqrt(c12) + math.sqrt(c13) +
                        math.sqrt(c20) + math.sqrt(c21) + math.sqrt(c22) + math.sqrt(c23) +
                        math.sqrt(c30) + math.sqrt(c31) + math.sqrt(c32) + math.sqrt(c33)
                    )

                c01, c02, c03 = 0.0, 0.0, 0.0
                c12, c13 = 0.0, 0.0
                c23 = 0.0
                for k in range(ncols):
                    x0k, x1k, x2k, x3k = x0[k], x1[k], x2[k], x3[k]
                    c01 += ((x0k-x1k)**2); c02 += ((x0k-x2k)**2); c03 += ((x0k-x3k)**2);
                    c12 += ((x1k-x2k)**2); c13 += ((x1k-x3k)**2);
                    c23 += ((x2k-x3k)**2);

                score_2 += (
                    math.sqrt(c01) + math.sqrt(c02) + math.sqrt(c03) +
                    math.sqrt(c12) + math.sqrt(c13) +
                    math.sqrt(c23)
                )

                for j in range(i+4, nx4, 4):
                    y0 = x[j, :]; y1 = x[j+1, :]; y2 = x[j+2, :]; y3 = x[j+3, :];

                    c00, c01, c02, c03 = 0.0, 0.0, 0.0, 0.0
                    c10, c11, c12, c13 = 0.0, 0.0, 0.0, 0.0
                    c20, c21, c22, c23 = 0.0, 0.0, 0.0, 0.0
                    c30, c31, c32, c33 = 0.0, 0.0, 0.0, 0.0

                    for k in range(ncols):
                        x0k, x1k, x2k, x3k = x0[k], x1[k], x2[k], x3[k]
                        y0k, y1k, y2k, y3k = y0[k], y1[k], y2[k], y3[k]

                        c00 += ((x0k-y0k)**2); c01 += ((x0k-y1k)**2);
                        c02 += ((x0k-y2k)**2); c03 += ((x0k-y3k)**2);
                        c10 += ((x1k-y0k)**2); c11 += ((x1k-y1k)**2);
                        c12 += ((x1k-y2k)**2); c13 += ((x1k-y3k)**2);
                        c20 += ((x2k-y0k)**2); c21 += ((x2k-y1k)**2);
                        c22 += ((x2k-y2k)**2); c23 += ((x2k-y3k)**2);
                        c30 += ((x3k-y0k)**2); c31 += ((x3k-y1k)**2);
                        c32 += ((x3k-y2k)**2); c33 += ((x3k-y3k)**2);

                    score_2 += (
                        math.sqrt(c00) + math.sqrt(c01) + math.sqrt(c02) + math.sqrt(c03) +
                        math.sqrt(c10) + math.sqrt(c11) + math.sqrt(c12) + math.sqrt(c13) +
                        math.sqrt(c20) + math.sqrt(c21) + math.sqrt(c22) + math.sqrt(c23) +
                        math.sqrt(c30) + math.sqrt(c31) + math.sqrt(c32) + math.sqrt(c33)
                    )

            bcx = (nx != nx4)
            bcy = (ny != ny4)

            if bcx:
                raise NotImplementedError
            #     for j in range(0, ny4, 4):
            #         y0, y1, y2, y3 = y[j, 0], y[j+1, 0], y[j+2, 0], y[j+3, 0]
            #         for i in range(nx4, nx):
            #             score_1 += abs(x[i, 0]-y0); score_1 += abs(x[i, 0]-y1);
            #             score_1 += abs(x[i, 0]-y2); score_1 += abs(x[i, 0]-y3);

            #     for i in range(0, nx4, 4):
            #         x0, x1, x2, x3 = x[i, 0], x[i+1, 0], x[i+2, 0], x[i+3, 0]
            #         x4, x5, x6, x7 = x[i+4, 0], x[i+5, 0], x[i+6, 0], x[i+7, 0]
            #         for j in range(nx4, nx):
            #             y0 = x[j, 0]
            #             score_2 += abs(x0-y0); score_2 += abs(x1-y0);
            #             score_2 += abs(x2-y0); score_2 += abs(x3-y0);
            #             score_2 += abs(x4-y0); score_2 += abs(x5-y0);
            #             score_2 += abs(x6-y0); score_2 += abs(x7-y0);

            #     for i in range(nx4, nx):
            #         x0 = x[i, 0]
            #         for j in range(i+1, nx):
            #             score_2 += abs(x0-x[j, 0])

            if bcy:
                raise NotImplementedError
            #     for i in range(0, nx4, 4):
            #         x0, x1, x2, x3 = x[i, 0], x[i+1, 0], x[i+2, 0], x[i+3, 0]
            #         x4, x5, x6, x7 = x[i+4, 0], x[i+5, 0], x[i+6, 0], x[i+7, 0]
            #         for j in range(ny4, ny):
            #             y0 = y[j, 0]
            #             score_1 += abs(x0-y0); score_1 += abs(x1-y0);
            #             score_1 += abs(x2-y0); score_1 += abs(x3-y0);
            #             score_1 += abs(x4-y0); score_1 += abs(x5-y0);
            #             score_1 += abs(x6-y0); score_1 += abs(x7-y0);

            # if bcx and bcy:
            #     for i in range(nx4, nx):
            #         x0 = x[i, 0]
            #         for j in range(ny4, ny):
            #             score_1 += abs(x0-y[j, 0])

            score_1 /= ny
            score_2 /= (nx - 1)
            return (score_1 - score_2) / nx

    earth_mover_v5.kernels[ncols] = kernel
    return earth_mover_v5.kernels[ncols](preds, obsrvs)

earth_mover_v5.kernels = {}
