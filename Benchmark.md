# Performance Benchmark — GMM Background Subtraction

## Benchmark Setup

All benchmarks were run on the same machine using identical video input and
identical algorithm parameters.

Implementations compared:
- Loop-based Python reference implementation
- Fully vectorized NumPy implementation
- C++ implementation (pybind11, fused kernel)

Measurements:
- Frames per second (FPS)
- Resolution scaling from 160×120 to 1280×720
- Preprocessing included (resize + grayscale)


## Resolution Scaling Results

| Resolution | Loop FPS | Vectorized FPS | C++ FPS |
|-----------|---------:|---------------:|--------:|
| 160×120   | 10.86    | 593.73         | 1692.87 |
| 320×240   | 2.71     | 163.85         | 434.38  |
| 640×480   | 0.69     | 43.57          | 115.98  |
| 800×600   | 0.44     | 27.88          | 73.36   |
| 1024×768  | 0.26     | 16.18          | 43.98   |
| 1280×720  | 0.22     | 14.41          | 38.34   |


## Speedup Summary

- Vectorized NumPy vs Loop:
  - Up to **54× speedup**
- C++ vs Loop:
  - Up to **170× speedup**
- C++ vs Vectorized:
  - ~**3× speedup**

The C++ backend consistently outperforms both Python implementations
across all tested resolutions.


## Observations

- Loop-based implementation becomes unusable beyond VGA resolution.
- Vectorization dramatically improves performance but still scales linearly
  with pixel count.
- C++ implementation benefits from:
  - fused operations
  - contiguous memory access
  - removal of Python interpreter overhead
- Preprocessing dominates runtime only at very small resolutions.
