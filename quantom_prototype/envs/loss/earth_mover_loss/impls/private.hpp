#pragma once
#include <functional>
#include <utility>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>


// compile-time constant NCOLS
#ifndef NKERNELS
#define NKERNELS 20
#endif

// compile-time constant COMPILER
#ifndef MODULENAME
    #if defined(__clang__)
        #define MODULENAME c_clang
    #elif defined(__GNUG__)  // must be the last as others also define __GNUG__
        #define MODULENAME c_gcc
    #else
        #error
    #endif
#endif

// factory macro; requires C++20
#define FACTORY(T, ARRY_t, FUNC) \
    []<size_t ...IDX>(std::index_sequence<IDX...>) constexpr { \
        return ARRY_t<T>{FUNC<T, IDX+1>...}; \
    }(std::make_index_sequence<NKERNELS>{})

// type alias to the kernel function's signature for v2
template<typename T>
using KERNEL_t_py = std::function<
    T(
        const pybind11::detail::unchecked_reference<T, 2>&,
        const pybind11::detail::unchecked_reference<T, 2>&,
        const size_t&, const size_t&
    )
>;

// type alias to the kernel function's signature v3
template<typename T>
using KERNEL_t_c = std::function<
    T(const T*&, const T*&, const size_t&, const size_t&)
>;

// type alias to an array of kernel functions v2
template<typename T>
using KERNEL_ARRY_t_py = std::array<KERNEL_t_py<T>, NKERNELS>;

// type alias to an array of kernel functions v3
template<typename T>
using KERNEL_ARRY_t_c = std::array<KERNEL_t_c<T>, NKERNELS>;

// v1 front interface for Python
template <typename T>
T earth_mover_v1(pybind11::array_t<T> &preds, pybind11::array_t<T> &obsrvs);

// v2 front interface for Python
template <typename T>
T earth_mover_v2(pybind11::array_t<T> &preds, pybind11::array_t<T> &obsrvs);

// v3 front interface for Python
template<typename T>
T earth_mover_v3(  // args need .noconvert in pybind11 module init
    pybind11::array_t<T, pybind11::array::c_style> &preds,
    pybind11::array_t<T, pybind11::array::c_style> &obsrvs
);

// v4 front interface for Python
template<typename T>
T earth_mover_v4(  // args need .noconvert in pybind11 module init
    pybind11::array_t<T, pybind11::array::c_style> &preds,
    pybind11::array_t<T, pybind11::array::c_style> &obsrvs
);
