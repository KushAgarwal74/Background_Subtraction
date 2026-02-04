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
  --camera camera_index \
  --scale 0.5  \
  --config configs/gmm.yaml \
  --mode cpp \

# scale between (0, 1]
# run app/webcam.py to find the camera index once
# Supported mode are loop, vec, cpp
```

---

### Performance Snapshot
```md
### Performance Snapshot

At 1280×720 resolution:

- Loop: ~0.29 FPS
- Loop Scaled@4x ~1.14 FPS

- Vectorized: ~13.63 FPS
- Vectorized Scaled@4x ~50.12 FPS

- C++: ~37.65 FPS
- C++ Scaled@4x ~137.12 FPS

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
