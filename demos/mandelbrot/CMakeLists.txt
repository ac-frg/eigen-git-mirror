# SPDX-FileCopyrightText: The Eigen Authors
# SPDX-License-Identifier: MPL-2.0

add_executable(mandelbrot mandelbrot.cpp mandelbrot.h)
set_target_properties(mandelbrot PROPERTIES AUTOMOC ON)
target_compile_features(mandelbrot PRIVATE cxx_std_17)
target_link_libraries(mandelbrot PRIVATE Eigen3::Eigen Qt6::Widgets)
add_dependencies(demos mandelbrot)
