#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace cppBpe
{
    void register_module(const pybind11::module_& m);
}