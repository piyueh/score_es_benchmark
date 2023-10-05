/**
 * \file c.cpp
 * \brief C implementations of Earth-Mover loss.
 * \author Pi-Yueh Chuang
 * \version 0.1-alpha
 * \date 2023-09-29
 */

#include <xmmintrin.h>
#include <pmmintrin.h>
#include "private.hpp"

PYBIND11_MODULE(MODULENAME, m) {

    // force to disable FTZ and DAS
    _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_OFF);
    _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_OFF);

    // documentation of this module
    m.doc() = "C implementations of the Earth-Mover loss";

    // implementation version 1; fp32
    m.def(
        "earth_mover_v1", &earth_mover_v1<double>,
        "C Implementation of the Earth-Mover loss, v1; flp32",
        pybind11::arg("preds"), pybind11::arg("obsrvs")
    );

    // implementation version 1; fp64
    m.def(
        "earth_mover_v1", &earth_mover_v1<float>,
        "C Implementation of the Earth-Mover loss, v1; flp64",
        pybind11::arg("preds"), pybind11::arg("obsrvs")
    );

    // implementation version 2; fp32
    m.def(
        "earth_mover_v2", &earth_mover_v2<double>,
        "C Implementation of the Earth-Mover loss, v2; flp32",
        pybind11::arg("preds"), pybind11::arg("obsrvs")
    );

    // implementation version 2; fp64
    m.def(
        "earth_mover_v2", &earth_mover_v2<float>,
        "C Implementation of the Earth-Mover loss, v2; flp64",
        pybind11::arg("preds"), pybind11::arg("obsrvs")
    );

    // implementation version 3; fp32
    m.def(
        "earth_mover_v3", &earth_mover_v3<double>,
        "C Implementation of the Earth-Mover loss, v3; flp32",
        pybind11::arg("preds").noconvert(), pybind11::arg("obsrvs").noconvert()
    );

    // implementation version 3; fp64
    m.def(
        "earth_mover_v3", &earth_mover_v3<float>,
        "C Implementation of the Earth-Mover loss, v3; flp64",
        pybind11::arg("preds").noconvert(), pybind11::arg("obsrvs").noconvert()
    );

    // implementation version 4; fp32
    m.def(
        "earth_mover_v4", &earth_mover_v4<double>,
        "C Implementation of the Earth-Mover loss, v4; flp32",
        pybind11::arg("preds").noconvert(), pybind11::arg("obsrvs").noconvert()
    );

    // implementation version 4; fp64
    m.def(
        "earth_mover_v4", &earth_mover_v4<float>,
        "C Implementation of the Earth-Mover loss, v4; flp64",
        pybind11::arg("preds").noconvert(), pybind11::arg("obsrvs").noconvert()
    );
}
