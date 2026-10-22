# SPDX-FileCopyrightText: The Eigen Authors
# SPDX-License-Identifier: MPL-2.0

find_package(Qt6 QUIET COMPONENTS OpenGLWidgets)
find_package(OpenGL QUIET)

if(Qt6OpenGLWidgets_FOUND AND TARGET OpenGL::GL AND TARGET OpenGL::GLU)
  add_executable(quaternion_demo gpuhelper.cpp icosphere.cpp camera.cpp trackball.cpp
                 quaternion_demo.cpp quaternion_demo.h)
  set_target_properties(quaternion_demo PROPERTIES AUTOMOC ON)
  target_compile_features(quaternion_demo PRIVATE cxx_std_17)
  target_link_libraries(quaternion_demo PRIVATE Eigen3::Eigen Qt6::OpenGLWidgets OpenGL::GL OpenGL::GLU)
  add_dependencies(demos quaternion_demo)
else()
  message(STATUS "OpenGL demo disabled because Qt6 OpenGLWidgets, desktop OpenGL or GLU was not found")
endif()
