/**
 * \file c_impl_v1.cpp
 * \brief C implementations of Earth-Mover loss.
 * \author Pi-Yueh Chuang
 * \version 0.1-alpha
 * \date 2023-09-29
 */

#include <cassert>
#include <cmath>
#include "private.hpp"

// kernel v1
template<typename T>
T kernel_v1(
    const pybind11::detail::unchecked_reference<T, 2> &x,
    const pybind11::detail::unchecked_reference<T, 2> &y,
    const size_t &nx,
    const size_t &ny,
    const size_t &ncols
) {
    double score_1 = 0.0;  // always use double to avoid overflow
    double score_2 = 0.0;  // always use double to avoid overflow
    T dist = 0.0;
    T tmp = 0.0;

    if (ncols == 1) {
        for (size_t xi = 0; xi < nx; xi++) {
            for (size_t yj = 0; yj < ny; yj++) {
                score_1 += std::abs(x(xi, 0) - y(yj, 0));
            }
            for (size_t xj = xi + 1; xj < nx; xj++) {
                score_2 += std::abs(x(xi, 0) - x(xj, 0));
            }
        }
    } else {
        for (size_t xi = 0; xi < nx; xi++) {
            for (size_t yj = 0; yj < ny; yj++) {
                dist = 0.0;
                for (size_t col = 0; col < ncols; col++) {
                    tmp = x(xi, col) - y(yj, col);
                    tmp *= tmp;
                    dist += tmp;
                }
                score_1 += std::sqrt(dist);
            }
            for (size_t xj = xi + 1; xj < nx; xj++) {
                dist = 0.0;
                for (size_t col = 0; col < ncols; col++) {
                    tmp = x(xi, col) - x(xj, col);
                    tmp *= tmp;
                    dist += tmp;
                }
                score_2 += std::sqrt(dist);
            }
        }
    }

    // separate divisions to avoid overflow
    score_1 /= (T) ny;
    score_2 /= (T) (nx - 1);
    return (score_1 - score_2) / (T) nx;
}

// v1 front interface for Python
template <typename T>
T earth_mover_v1(pybind11::array_t<T> &preds, pybind11::array_t<T> &obsrvs) {
    const auto &p_buf = preds.request();  // standard Buffer Protocol
    const auto &o_buf = obsrvs.request();  // standard Buffer Protocol
    const auto &np = p_buf.shape[0];  // number of rows in preds
    const auto &no = o_buf.shape[0];  // number of rows in obsrvs
    const auto &ncols = p_buf.shape[1];  // number of columns
    const auto &p_ptr = preds.template unchecked<2>();
    const auto &o_ptr = obsrvs.template unchecked<2>();

    // sanity checks
    assert(p_buf.ndim == 2);
    assert(o_buf.ndim == 2);
    assert(o_buf.shape[1] == ncols);
    assert(p_buf.format == o_buf.format);

    return kernel_v1(p_ptr, o_ptr, np, no, ncols);
}

// instantiations
template float earth_mover_v1<float>(
    pybind11::array_t<float> &preds, pybind11::array_t<float> &obsrvs
);

template double earth_mover_v1<double>(
    pybind11::array_t<double> &preds, pybind11::array_t<double> &obsrvs
);
