# Architecture Overview

## High-Level Design

The project implements Gaussian Mixture Model (GMM) background subtraction
using three progressively optimized backends:

1. Loop-based Python reference
2. Vectorized NumPy implementation
3. High-performance C++ backend (pybind11)

All backends share:
- identical mathematical formulation
- identical parameterization
- identical input/output behavior


## Backend Evolution

### 1. Loop-based Python
- Per-pixel processing
- Easy to reason about
- Used as the reference implementation
- Computationally expensive due to Python loops

### 2. Vectorized NumPy
- Eliminates Python loops
- Uses dense array operations
- Separates:
  - matching
  - classification
  - model update
- Major performance improvement with identical outputs

### 3. C++ Backend
- Single fused kernel
- Matching, classification, and update performed in one pass
- Exposed to Python via pybind11
- Maximizes cache locality and minimizes overhead


## Why the C++ Backend Uses a Single Kernel

Unlike the Python implementations, the C++ backend intentionally fuses all
stages of the algorithm into a single `apply()` function.

This design:
- eliminates intermediate memory allocations
- avoids Python↔C++ transitions
- improves cache locality
- reflects how production vision kernels are implemented

Internal stage separation is an implementation detail rather than an API concern.


## Correctness Validation

All implementations were validated using:
- frame-by-frame parity tests
- multi-frame drift analysis
- mismatch heatmap visualization

Results:
- 100% pixel-level agreement across loop, vectorized, and C++ backends
- no numerical drift over time
