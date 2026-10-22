# SPDX-FileCopyrightText: The Eigen Authors
# SPDX-License-Identifier: MPL-2.0

project(EigenDemos)

add_custom_target(demos)

option(EIGEN_TEST_NOQT "Disable Qt-based demos" OFF)
if(NOT EIGEN_TEST_NOQT)
  find_package(Qt6 QUIET COMPONENTS Widgets)
  if(Qt6Widgets_FOUND)
    add_subdirectory(mandelbrot)
    add_subdirectory(opengl)
  else()
    message(STATUS "Qt6 Widgets not found, so disabling the mandelbrot and opengl demos")
  endif()
endif()
