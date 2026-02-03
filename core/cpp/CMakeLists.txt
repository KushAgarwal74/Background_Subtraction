cmake_minimum_required(VERSION 3.15)        # macOS users need >= 3.15

project(gmm_cpp LANGUAGES CXX)
# Project name = Python module name
# This matters: the .so will be called gmm_cpp.*.so

# ---------------------
# 1. C++ STANDARD
# ---------------------

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# pybind11 works best with C++17
# Avoids compiler mismatch bugs

# ---------------------
# 2. Find Python
# ---------------------
# instead of harcoding and typing a path like /usr/local/include use find_package
# Interpreter: Finds the python executable to check version compatibility.
# Development: Finds the headers (Python.h) and libraries needed to link C++ to Python.
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

# ---------------------
# 3. Find pybind11
# ---------------------
# If installed via pip: pybind11 provides CMake config
find_package(pybind11 CONFIG REQUIRED)

# ------------------------------------------
# 4. Create/Build Python extension module
# ------------------------------------------

# In CMake, a "Target" is an object (like an executable or library). Here, gmm_cpp is our target.
# This does three things automatically:
# builds a shared library
# links Python + pybind11
# names it correctly for Python import .so or .pyd
pybind11_add_module(gmm_cpp 
    gmm_cpp.cpp
)

# ------------------------------
# 5. Include headers
# ------------------------------
# PRIVATE: This means "Only this module needs C++17 and these headers." 
# It prevents "leaking" settings to other parts of a larger project.

# Set C++17 requirement specifically for THIS target
target_compile_features(gmm_cpp PRIVATE cxx_std_17)

# Professional way but Optional: You don't need a "source path" for C++17 headers because they are part of the Standard Library built
# into your compiler (Clang/GCC).
# Tell the target where its own headers (gmm_model.hpp) are
target_include_directories(gmm_cpp
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# Optional: speed flags
# CMake often builds in "Debug" mode (slow). Professionals force high-speed flags for computer vision math.
if (CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    target_compile_options(gmm_cpp PRIVATE -O3 -march=native)
endif()

# ---------------------------
# 6. Organize the output (temporary only when there is no pyproject.toml and proper CMake installation )
# ---------------------------
# This is a cleaner alternative to 'install' for local development.
# It puts the .so file directly next to your Python script.
# set_target_properties(gmm_cpp PROPERTIES 
#     LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/../../"
# )

# message(STATUS "The Project Root is: ${CMAKE_SOURCE_DIR}")

install(
    TARGETS gmm_cpp
    LIBRARY DESTINATION background_subtraction_gmm
)

