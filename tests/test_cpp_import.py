import numpy as np
from background_subtraction_gmm import gmm_cpp

H, W = 100, 100
first = np.zeros((H, W), dtype=np.float32)
model = gmm_cpp.GMMModel(first, k=3, alpha=0.01, threshold=2.5, bg_threshold=0.7)

frame = np.random.rand(H, W).astype(np.float32)
mask = model.apply(frame)

print(mask.shape, mask.dtype, mask.min(), mask.max())
