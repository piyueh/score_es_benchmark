#! /usr/bin/env python3
# vim:fenc=utf-8
#
# Copyright © 2023 Pi-Yueh Chuang <pychuang@pm.me>
#
# Distributed under terms of the BSD 3-Clause license.

"""Misc. utilities.
"""
import pathlib
import numbers
import numpy
import scipy.integrate
import scipy.interpolate
import impls


class DataLoader:
    """A dummy data loader.

    Arguments
    ---------
    paths : list[str|os.PathLike]
        A list of paths to the data files.
    prefix : str|os.PathLike|None
        A prefix to be prepended to the paths. Optional.

    Notes
    -----
    If `prefix` is None, the paths are assumed to be relative to the current
    work directory, or they can be absolute paths.
    """

    def __init__(self, paths, prefix=None, seed=None):
        # random number generator
        self.rng = numpy.random.default_rng(seed)

        # make them os.PathLike objects and use absolute paths
        if prefix is not None:
            self.prefix = pathlib.Path(prefix).expanduser().resolve()
        else:
            self.prefix = pathlib.Path.cwd().expanduser().resolve()


        # first force them to be os.PathLike objects
        self.paths = [pathlib.Path(p) for p in paths]

        # then make sure they are absolute paths
        self.paths = [
            p if p.is_absolute() else self.prefix.joinpath(p)
            for p in self.paths
        ]

        # load data to RAM; hopefully they are not too large...
        self.events = [numpy.load(p)["events"] for p in self.paths]
        self.norms = [numpy.load(p)["norm"] for p in self.paths]

    def bootstrap(self, nevents, channels=None):
        """Get a bootstrap from a selected data channels.
        """
        if channels is None:
            channels = range(len(self.paths))

        # make sure `nevts` is a correct-length sequence or a positive integer
        if isinstance(nevents, numbers.Integral):
            if nevents <= 0:
                raise ValueError("`nevents` can not be <= 0.")
            nevents = [nevents] * len(channels)
        else:
            assert len(nevents) == len(channels), \
                "`nevents`: length not match `channels`."
            assert all([n > 0 for n in nevents]), \
                "`nevents`: all elements must be > 0."

        events = []
        for n, ch in zip(nevents, channels):
            events.append(self.rng.choice(self.events[ch], n, True))

        return events


class QCF:
    """A mock QCF.

    Arguments
    ---------
    xmin, xmax : float
        The lower and upper limit of allowed x values.
    """

    def __init__(self, xmin=1e-5, xmax=1.-1e-5):
        self.xmin = xmin
        self.xmax = xmax

    @staticmethod
    def get_u(x, params):
        """Get u.
        """
        params = numpy.asarray(params)  # zero-overhead if params is ndarray
        x = numpy.asarray(x)  # zero-overhead if x is ndarray
        u = params[0] * (x**params[1]) * ((1.0 - x)**params[2])
        return u

    @staticmethod
    def get_d(x, params):
        params = numpy.asarray(params)  # zero-overhead if params is ndarray
        x = numpy.asarray(x)  # zero-overhead if x is ndarray
        d = params[0] * (x**params[1]) * ((1.0 - x)**params[2])
        return d

    @staticmethod
    def get_ud(x, params):
        params = numpy.asarray(params)  # zero-overhead if params is ndarray
        x = numpy.asarray(x)  # zero-overhead if x is ndarray
        u = params[0] * (x**params[1]) * ((1.0 - x)**params[2])
        d = params[3] * (x**params[4]) * ((1.0 - x)**params[5])
        return u, d

    def __call__(self, x, params):
        return self.get_ud(x, params)

    def verify_x(self, x):
        """Check if values in an array x are all within the limits.
        """
        return numpy.logical_and(x >= xmin, x <= xmax)


class CrossSectionBase:
    """A base class for cross-section given a QCF model.

    Arguments
    ---------
    qcf : QCF
        An instance of class `QCF`.
    """

    def __init__(self, qcf: QCF):
        self.qcf = qcf  # a ref, not a hard copy
        self.xmin = self.qcf.xmin  # for convenience
        self.xmax = self.qcf.xmax  # for convenience
        self._quad = scipy.integrate.quad  # for convenience

        # caches
        self.cache_params = None
        self.cache_norm = None

    def get_cross_section(self, x, params):
        """Get cross-section at x with given QCF parameters.
        """
        raise NotImplementedError

    def __call__(self, x, params):
        return self.get_cross_section(x, params)

    def get_norm(self, params):
        """Get the norm of the cross-section with given QCF parameters.
        """
        params = numpy.asarray(params)  # zero-overhead if params is ndarray

        if not numpy.array_equal(params, self.cache_params):
            self.cache_params = params
            self.cache_norm = self._quad(
                self.get_cross_section, self.xmin, self.xmax, (params,))[0]

        return self.cache_norm


class CrossSectionIMPL1(CrossSectionBase):
    """A mock cross-section given a QCF model.

    Arguments
    ---------
    qcf : QCF
        An instance of class `QCF`.
    """

    def get_cross_section(self, x, params):
        u, d = self.qcf(x, params)
        # print(f"x: {x}", 4.0*u+d)
        return 4.0 * u + d


class CrossSectionIMPL2(CrossSectionBase):
    """A mock cross-section given a QCF model.

    Arguments
    ---------
    qcf : QCF
        An instance of class `QCF`.
    """

    def get_cross_section(self, x, params):
        u, d = self.qcf(x, params)
        return u + 4.0 * d


class Sampler:
    """A simple inverse CDF sampler.
    """

    def __init__(self, cross, npts=100, seed=None):
        self.cross = cross
        self.npts = npts
        self.xmin = self.cross.xmin  # for convenience
        self.xmax = self.cross.xmax  # for convenience
        self.x = numpy.linspace(self.xmin, self.xmax, self.npts)
        self._quad = numpy.vectorize(scipy.integrate.quad, excluded={"b", 2, "args", 3})
        self._rng = numpy.random.default_rng(seed)

    def pdf(self, x, params, norm=None):
        """Evaluate PDF at x with given QCF params.
        """
        if norm is None:
            norm = self.cross.get_norm(params)
        return self.cross(x, params) / norm

    def cdf(self, x, params):
        """Evaluate CDF at x with given QCF parameters.
        """
        norm = self.cross.get_norm(params)
        return self._quad(self.pdf, x, self.xmax, (params, norm))[0]

    def get_icdf_func(self, params):
        """Get a callable object representing the inverse CDF function.
        """
        return scipy.interpolate.interp1d(
            self.cdf(self.x, params), self.x,
            bounds_error=False, fill_value=0
        )

    def icdf(self, y, params):
        """Evaluate inverse CDF values at y with given QCF parameters.

        This function: probability density \mapsto x
        """
        icdf_func = self.get_icdf_func(params)
        return icdf_func(y)

    def get_samples(self, nevents, params):
        """Get samples from the underlying cross-section with given QCF params.
        """
        samples = self.icdf(self._rng.uniform(0., 1., nevents), params)
        samples = samples.reshape(-1, 1)
        return samples

    def __call__(self, nevents, params):
        return self.get_samples(nevents, params)


class StatsLoss:
    """Loss function for stats workflow 1 using NumPy.
    """

    def __init__(self, n_channels, weights=1., impl=("C v4", "Clang")):

        # make aliases to frequently used settings for conveniences
        self.impl = impl
        self.n_channels = n_channels

        if isinstance(weights, numbers.Number):
            self.weights = numpy.full(n_channels*2, weights, dtype=float)
        else:
            self.weights = numpy.asarray(weights, dtype=float)

        # temporary data holders for logging purpose outside this class
        self.cost = 0.
        self.costs = numpy.zeros((2*n_channels,), dtype=float)
        self.weighted_costs = numpy.zeros((2*n_channels,), dtype=float)

        # actual implementation of the earth-mover loss
        self.score_es = impls.options[self.impl]

    def forward(self, preds: list, obsrvs: list) -> float:
        """Calculate the total loss.

        Arguments
        ---------
        obsrvs : list[dict(events: ndarray, norm: float)]
            Its length should be the number of channels. Each element in it is a
            dictionary of two key-value pairs:
            1) `events`: an array of shape (n_events, event_size) for observed
               events of the corresponding data channel;
            2) `norm`: float-like; the norm of the observed events.
        preds : list[dict(events: ndarray, norm: float)]
            The predicted events. `preds` has the same structure as `obsrvs`.
            However, the array in `events` can have different number of rows.

        Returns
        -------
        cost : float or a numpy.floating
            Weighted and aggregated loss.
        """

        for ch, (pred, obsrv) in enumerate(zip(preds, obsrvs)):
            self.costs[ch*2] = self.score_es(pred["events"], obsrv["events"])
            self.costs[ch*2+1] = (obsrv["norm"] - pred["norm"])**2

        self.weighted_costs[...] = self.weights * self.costs
        self.cost = numpy.sum(self.weighted_costs)

        return self.cost
