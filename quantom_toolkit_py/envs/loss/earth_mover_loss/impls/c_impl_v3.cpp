/**
 * \file c_impl_b3.cpp
 * \brief C implementations of Earth-Mover loss.
 * \author Pi-Yueh Chuang
 * \version 0.1-alpha
 * \date 2023-09-29
 */

#include <cassert>
#include <cmath>
#include "private.hpp"

// kernel v3
template<typename T, size_t NCOLS>
T kernel_v3(
    const T* &x, const T* &y, const size_t &nx, const size_t &ny
) {
    double score_1 = 0.0;  // always use double to avoid overflow
    double score_2 = 0.0;  // always use double to avoid overflow
    T dist = 0.0;
    T tmp = 0.0;

    for (size_t xi = 0; xi < nx; xi++) {
        for (size_t yi = 0; yi < ny; yi++) {
            if constexpr(NCOLS == 1) {  // compile-time if; avail since C++17
                score_1 += std::abs(x[xi] - y[yi]);
            } else {
                dist = 0.0;
                for (size_t col = 0; col < NCOLS; col++) {
                    tmp = x[xi * NCOLS + col] - y[yi * NCOLS + col];
                    tmp *= tmp;
                    dist += tmp;
                }
                score_1 += std::sqrt(dist);
            }
        }
        for (size_t xj = xi + 1; xj < nx; xj++) {
            if constexpr(NCOLS == 1) {  // compile-time if; avail since C++17
                score_2 += std::abs(x[xi] - x[xj]);
            } else {
                dist = 0.0;
                for (size_t col = 0; col < NCOLS; col++) {
                    tmp = x[xi * NCOLS + col] - x[xj * NCOLS + col];
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

// kernel holder templates
template<typename T>
const KERNEL_ARRY_t_c<T> v3_kernels{FACTORY(T, KERNEL_ARRY_t_c, kernel_v3)};

// instantiations: actual kernel holders for different FPs
template const KERNEL_ARRY_t_c<float> v3_kernels<float>;
template const KERNEL_ARRY_t_c<double> v3_kernels<double>;

// v3 front interface for Python
template<typename T>
T earth_mover_v3(  // args need .noconvert in pybind11 module init
    pybind11::array_t<T, pybind11::array::c_style> &preds,
    pybind11::array_t<T, pybind11::array::c_style> &obsrvs
) {
    const auto &p_buf = preds.request();  // standard Buffer Protocol
    const auto &o_buf = obsrvs.request();  // standard Buffer Protocol
    const auto &np = p_buf.shape[0];  // number of rows in preds
    const auto &no = o_buf.shape[0];  // number of rows in obsrvs
    const auto &ncols = p_buf.shape[1];  // number of columns
    const T *p_ptr{static_cast<const T*>(p_buf.ptr)};  // raw array pointer
    const T *o_ptr{static_cast<const T*>(o_buf.ptr)};  // raw array pointer

    // get the corresponding kernel
    const auto &kernel = v3_kernels<T>.at(ncols-1);  // w/ boundcheck

    // sanity checks
    assert(p_buf.ndim == 2);
    assert(o_buf.ndim == 2);
    assert(o_buf.shape[1] == ncols);
    assert(p_buf.format == o_buf.format);

    return kernel(p_ptr, o_ptr, np, no);
}

// instantiations
template float earth_mover_v3<float>(
    pybind11::array_t<float, pybind11::array::c_style> &preds,
    pybind11::array_t<float, pybind11::array::c_style> &obsrvs
);

template double earth_mover_v3<double>(
    pybind11::array_t<double, pybind11::array::c_style> &preds,
    pybind11::array_t<double, pybind11::array::c_style> &obsrvs
);
