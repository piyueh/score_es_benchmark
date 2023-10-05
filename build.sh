#! /bin/sh

g++ -std=c++20 $(python3 -m pybind11 --includes) -fPIC -march=native -O3 -o c_impl_v1.o -c c_impl_v1.cpp
g++ -std=c++20 $(python3 -m pybind11 --includes) -fPIC -march=native -O3 -o c_impl_v2.o -c c_impl_v2.cpp
g++ -std=c++20 $(python3 -m pybind11 --includes) -fPIC -march=native -O3 -o c_impl_v3.o -c c_impl_v3.cpp
g++ -std=c++20 $(python3 -m pybind11 --includes) -fPIC -march=native -Ofast -ffast-math -funroll-loops -o c_impl_v4.o -c c_impl_v4.cpp
g++ -std=c++20 $(python3 -m pybind11 --includes) -fPIC -march=native -O3 -o c_impls.o -c c_impls.cpp
g++ -shared -o c_gcc$(python3-config --extension-suffix) c_impl_v1.o c_impl_v2.o c_impl_v3.o c_impl_v4.o c_impls.o
\rm c_impl_v1.o c_impl_v2.o c_impl_v3.o c_impl_v4.o c_impls.o

clang++ -std=c++20 $(python3 -m pybind11 --includes) -fPIC -march=native -O3 -o c_impl_v1.o -c c_impl_v1.cpp
clang++ -std=c++20 $(python3 -m pybind11 --includes) -fPIC -march=native -O3 -o c_impl_v2.o -c c_impl_v2.cpp
clang++ -std=c++20 $(python3 -m pybind11 --includes) -fPIC -march=native -O3 -o c_impl_v3.o -c c_impl_v3.cpp
clang++ -std=c++20 $(python3 -m pybind11 --includes) -fPIC -march=native -Ofast -ffast-math -funroll-loops -o c_impl_v4.o -c c_impl_v4.cpp
clang++ -std=c++20 $(python3 -m pybind11 --includes) -fPIC -march=native -O3 -o c_impls.o -c c_impls.cpp
clang++ -shared -o c_clang$(python3-config --extension-suffix) c_impl_v1.o c_impl_v2.o c_impl_v3.o c_impl_v4.o c_impls.o
\rm c_impl_v1.o c_impl_v2.o c_impl_v3.o c_impl_v4.o c_impls.o
