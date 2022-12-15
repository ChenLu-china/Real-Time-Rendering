# Copyright 2022 Real-Time Rendering Engineer

# NOTE: This CMakeLists is for test purpose only
cmake_minimum_required(VERSION 2.8.12)

project(pybind11_example)

add_subdirectory(3rd/pybind11)

pybind11_add_module()

find_package(pybind11 REQUIRED)
find_package(Torch REQUIRED)