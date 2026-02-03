import cv2
import numpy as np

def show_frame(gray_norm: np.ndarray, window_name="Video"):
    """
    Expects grayscale image in [0,1]
    """
    gray_uint8 = (gray_norm * 255).astype("uint8")
    cv2.imshow(window_name, gray_uint8)
