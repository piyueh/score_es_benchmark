#! /usr/bin/env python3
# vim:fenc=utf-8

"""Earth-Mover loss implementations similiar to the original NumPy code.
"""
import numpy
import scipy
import pylibraft.distance
import cupy
import cupyx.scipy.spatial
import torch
import numba


def earth_mover_numpy(preds, obsrvs):
    """Computes the energy score between the predicted and observed events.

    Notes
    -----
    This is a simplified version of the original code from Emil C..

    Parameters
    ----------
    preds : numpy.ndarray
        A 2D array of predicted events.

    obsrvs : numpy.ndarray
        A 2D array of observed events.

    Returns
    -------
    float
        The discriminative score between the predicted and observed events.
    """
    # sanity checks; can be disabled w/ CMD flag `-O`
    assert preds.ndim == obsrvs.ndim == 2
    assert preds.shape[1] == obsrvs.shape[1]  # i.e., nfeatures

    np = len(preds.shape)

    # Calculate score1 (preds[:, None].shape = (np, 1, nfeatures))
    score1 = numpy.linalg.norm(preds[:, None]-obsrvs, ord=2, axis=2)
    score1 = numpy.mean(score1)

    # Calculate score2 (preds[:, None].shape = (np, 1, nfeatures))
    if np > 1:
        score2 = numpy.linalg.norm(preds[:, None]-preds, ord=2, axis=2)
        numpy.fill_diagonal(score2, 0)  # should already be zero "theoretically"
        score2 = numpy.sum(score2) / (2 * np * (np - 1))
    else:
        score2 = 0.0

    return score1 - score2


def earth_mover_scipy(preds, obsrvs):
    """Earth-Mover loss using SciPy for pairwise distance matrices.
    """
    # sanity checks; can be disabled w/ CMD flag `-O`
    assert preds.ndim == obsrvs.ndim == 2
    assert preds.shape[1] == obsrvs.shape[1]  # i.e., nfeatures

    np = len(preds)

    # Calculate score1
    score1 = numpy.mean(scipy.spatial.distance.cdist(preds, obsrvs))

    # Calculate score2
    score2 = numpy.sum(scipy.spatial.distance.pdist(preds))
    score2 /= (np * (np - 1))  # note pdist only calculate half the matrix

    return score1 - score2


@numba.njit(nogil=True, boundscheck=False)
def earth_mover_numba(preds, obsrvs):
    """Naive Numba implementations.

    Notes
    -----
    * Credit: Niteya
    * We tried to mimic the original Python code as much as possible. However,
      the original NumPy's norm function accepts an argument `axis`, while
      Numba's own norm implementation does not. We therefore had to modified the
      code somehow.
    """
    # sanity checks; can be disabled w/ CMD flag `-O`
    assert preds.ndim == obsrvs.ndim == 2
    assert preds.shape[1] == obsrvs.shape[1]  # i.e., nfeatures

    no = len(obsrvs)
    np = len(preds)

    score1 = 0.0
    score2 = 0.0
    for i in range(np):
        for j in range(no):
            score1 += numpy.linalg.norm(preds[i]-obsrvs[j])

        for j in range(i+1, np):
            score2 += numpy.linalg.norm(preds[i]-preds[j])

    score1 /= (no * np)
    score2 /= (np * (np - 1))

    return score1 - score2


def earth_mover_cupy(preds, obsrvs):
    """Earth-Mover loss using CuPy for pairwise distance matrices.

    Notes
    -----
    * This is GPU code.
    * CuPy's cdist and pdist only support float32 as of Oct 2023
    """
    # sanity checks; can be disabled w/ CMD flag `-O`
    assert preds.ndim == obsrvs.ndim == 2
    assert preds.shape[1] == obsrvs.shape[1]  # i.e., nfeatures
    assert preds.dtype == cupy.float32, "CuPy's cdist only supports float32"
    assert obsrvs.dtype == cupy.float32, "CuPy's cdist only supports float32"

    np = len(preds)

    # Calculate score1
    score1 = cupy.mean(cupyx.scipy.spatial.distance.cdist(preds, obsrvs))

    # Calculate score2
    score2 = cupy.sum(cupyx.scipy.spatial.distance.pdist(preds))
    score2 /= (np * (np - 1))

    return score1 - score2


def earth_mover_raft(preds, obsrvs):
    """Earth-Mover loss using Raft for pairwise distance matrices.

    Notes
    -----
    * This is GPU code.
    * The input arrays are CuPy arrays.
    """
    # sanity checks; can be disabled w/ CMD flag `-O`
    assert preds.ndim == obsrvs.ndim == 2
    assert preds.shape[1] == obsrvs.shape[1]  # i.e., nfeatures

    np = len(preds)

    # pre-allocate memory
    out1 = cupy.zeros((pred_size, obsrvs.shape[0]), dtype=preds.dtype)
    out2 = cupy.zeros((pred_size, pred_size), dtype=preds.dtype)

    # Calculate score1
    pylibraft.distance.pairwise_distance(preds, obsrvs, out=out1)

    # Calculate score2
    pylibraft.distance.pairwise_distance(preds, preds, out=out2)

    score1 = out1.mean()
    score2 = out2.sum()
    score2 /= (2 * np * (np - 1))  # note score2 was the full matrix

    return score1 - score2


def earth_mover_torch(preds, obsrvs):
    """Earth-Mover loss using PyTorch for pairwise distance matrices.

    Notes
    -----
    This function can run on either CPU or GPU, depending on the location of
    the input tensors.
    """
    # sanity checks; can be disabled w/ CMD flag `-O`
    assert preds.ndim == obsrvs.ndim == 2
    assert preds.shape[1] == obsrvs.shape[1]  # i.e., nfeatures

    np = len(preds)

    # Calculate score1
    score1 = torch.cdist(preds, obsrvs).mean()  # = euclidean

    # Calculate score2
    score2 = torch.nn.functional.pdist(preds).sum()  # = euclidean
    score2 /= (np * (np - 1))

    return score1 - score2


@torch.jit.script
def earth_mover_torchscript(preds, obsrvs):
    """Earth-Mover loss using TorchScript for pairwise distance matrices.

    Notes
    -----
    This function can run on either CPU or GPU, depending on the location of
    the input tensors.
    """
    # sanity checks; can be disabled w/ CMD flag `-O`
    assert preds.ndim == obsrvs.ndim == 2
    assert preds.shape[1] == obsrvs.shape[1]  # i.e., nfeatures

    np = len(preds)

    # Calculate score1
    score1 = torch.cdist(preds, obsrvs).mean()  # euclidean norm

    # Calculate score2
    score2 = torch.nn.functional.pdist(preds).sum()  # euclidean norm
    score2 /= (np * (np - 1))

    return score1 - score2
