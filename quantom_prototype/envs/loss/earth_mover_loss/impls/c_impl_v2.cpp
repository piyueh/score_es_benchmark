/**
 * \file c_impl_v2.cpp
 * \brief C implementations of Earth-Mover loss.
 * \author Pi-Yueh Chuang
 * \version 0.1-alpha
 * \date 2023-09-29
 */

#include <cassert>
#include <cmath>
#include "private.hpp"

// kernel v2
template <typename T, size_t NCOLS>
T kernel_v2(
    const pybind11::detail::unchecked_reference<T, 2> &x,
    const pybind11::detail::unchecked_reference<T, 2> &y,
    const size_t &nx,
    const size_t &ny
) {
    double score_1 = 0.0;  // always use double to avoid overflow
    double score_2 = 0.0;  // always use double to avoid overflow
    T dist = 0.0;
    T tmp = 0.0;

    for (size_t xi = 0; xi < nx; xi++) {
        for (size_t yj = 0; yj < ny; yj++) {
            if constexpr(NCOLS == 1) {  // compile-time if; avail since C++17
                score_1 += std::abs(x(xi, 0) - y(yj, 0));
            } else {
                dist = 0.0;
                for (size_t col = 0; col < NCOLS; col++) {
                    tmp = x(xi, col) - y(yj, col);
                    tmp *= tmp;
                    dist += tmp;
                }
                score_1 += std::sqrt(dist);
            }
        }
        for (size_t xj = xi + 1; xj < nx; xj++) {
            if constexpr(NCOLS == 1) {  // compile-time if; avail since C++17
                score_2 += std::abs(x(xi, 0) - x(xj, 0));
            } else {
                dist = 0.0;
                for (size_t col = 0; col < NCOLS; col++) {
                    tmp = x(xi, col) - x(xj, col);
                    tmp *= tmp;
                    dist += tmp;
                }
                score_2 += std::sqrt(dist);
            }
        }
    }

    // calculation and return
    // separate divisions to avoid overflow; they are int nyt long...
    score_1 /= (T) ny;
    score_2 /= (T) (nx - 1);
    return (score_1 - score_2) / (T) nx;
}

// kernel holder templates
template<typename T>
const KERNEL_ARRY_t_py<T> v2_kernels{FACTORY(T, KERNEL_ARRY_t_py, kernel_v2)};

// instantiations: actual kernel holders for different FPs
template const KERNEL_ARRY_t_py<float> v2_kernels<float>;
template const KERNEL_ARRY_t_py<double> v2_kernels<double>;

// v2 front interface for Python
template <typename T>
T earth_mover_v2(pybind11::array_t<T> &preds, pybind11::array_t<T> &obsrvs) {
    const auto &p_buf = preds.request();  // standard Buffer Protocol
    const auto &o_buf = obsrvs.request();  // standard Buffer Protocol
    const auto &np = preds.shape()[0];  // number of rows in preds
    const auto &no = obsrvs.shape()[0];  // number of rows in obsrvs
    const auto &ncols = preds.shape()[1];  // number of columns
    const auto &p_ptr = preds.template unchecked<2>();
    const auto &o_ptr = obsrvs.template unchecked<2>();

    // get the corresponding kernel
    const auto &kernel = v2_kernels<T>.at(ncols-1);  // w/ boundcheck

    // sanity checks
    assert(p_buf.ndim == 2);
    assert(o_buf.ndim == 2);
    assert(o_buf.shape[1] == ncols);
    assert(p_buf.format == o_buf.format);

    return kernel(p_ptr, o_ptr, np, no);
}

// instantiations
template float earth_mover_v2<float>(
    pybind11::array_t<float> &preds, pybind11::array_t<float> &obsrvs
);

template double earth_mover_v2<double>(
    pybind11::array_t<double> &preds, pybind11::array_t<double> &obsrvs
);
