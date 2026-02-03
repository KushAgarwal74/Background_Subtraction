# GMM Background Subtraction — Python & C++

This project implements Gaussian Mixture Model (GMM) background subtraction
with a strong focus on correctness, performance, and engineering rigor.

The same algorithm is implemented three ways:
- reference Python implementation
- optimized NumPy version
- high-performance C++ backend


## Features

- Identical behavior across Python and C++ backends
- Vectorized and fused-kernel implementations
- CLI-driven execution
- YAML-based configuration
- Pixel-level parity testing
- Resolution-scaling benchmarks


## Usage

Run background subtraction:

```bash
python -m app.run_video \
  --video data/videos/bowling.mp4 \
  --mode cpp \
  --config configs/gmm.yaml

# Supported mode are loop, vec, cpp
```

---

### Performance Snapshot
```md
### Performance Snapshot

At 1280×720 resolution:

- Loop: ~0.22 FPS
- Vectorized: ~14.4 FPS
- C++: ~38.3 FPS

Full benchmark results are available in `BENCHMARK.md`.


# Validation

All backends were validated using automated parity tests ensuring:

- 100% pixel-level agreement
- no temporal drift
- consistent behavior across resolutions


## Why This Matters

This project demonstrates:
- algorithmic understanding
- numerical correctness
- performance optimization
- clean Python↔C++ integration
- real-world engineering practices
