import numpy as np
import cv2

from core.pixel_model_loop import SingleGaussianPixel,GMMPixel
from .pixel_model_vectorized import GMMPixelVectorized

class SingleGaussianBackground:
    def __init__(self, first_frame: np.ndarray, alpha = 0.01):
        h, w = first_frame.shape
        self.model = [
            [SingleGaussianPixel(first_frame[i,j],alpha) for j in range(w)]
            for i in range(h)
        ]
        
    def apply(self, frame:np.ndarray):
        """
        frame : grayscale normalized [0, 1]
        return : foreground mask
        """
        h, w = frame.shape
        fg_mask = np.zeros((h,w), dtype = np.uint8)

        for i in range(h):
            for j in range(w):
                pixel = frame[i][j]
                g = self.model[i][j]
            
                if g.is_foreground((pixel)):
                    fg_mask[i,j] = 255
                else:
                    g.update(pixel)

        return fg_mask

# Image wide GMM
class GMMBackground:
    def __init__(self, first_frame: np.ndarray, k: int = 3, alpha:float = 0.01,
                 threshold : float = 2.5, bg_threshold: float = 0.7):
        h, w = first_frame.shape
        self.model = [
            [GMMPixel(first_frame[i][j], k, alpha, threshold = threshold, bg_threshold = bg_threshold) for j in range(w)] for i in range(h)
        ]

    def apply(self,frame:np.ndarray):
        """
        frame = grayscale normalized [0,1]
        return : foreground mask
        """
        h, w = frame.shape
        fg_mask = np.zeros((h,w), dtype = np.uint8)

        for i in range(h):
            for j in range(w):
                pixel = frame[i][j]
                p = self.model[i][j]

                matched = p.get_best_match(pixel)

                if not p.is_background(matched):
                    fg_mask[i, j] = 255
                p.update(pixel, matched)
        
        return fg_mask
    
class GMMBackgroundVectorized:
    def __init__(self, first_frame: np.ndarray, k: int = 3, alpha:float = 0.01,
                 threshold : float = 2.5, bg_threshold: float = 0.7):
        # Instead of millions of objects, we have ONE vectorized object
        self.pixel_logic = GMMPixelVectorized(first_frame, k, alpha, threshold, bg_threshold)

    def apply(self, frame: np.ndarray):
        # Step 1: Matching
        matches, any_match, best_k, dists = self.pixel_logic.get_best_match(frame)
        
        # Step 2: Decide Background/Foreground
        fg_mask = self.pixel_logic.is_background(any_match, best_k)
        
        # Step 3: Selective Update
        # We create a mask where it's okay to update the background model.
        # Usually, we only update pixels that were classified as Background (0).
        # update_mask = (fg_mask == 0)
        update_mask = np.ones_like(any_match, dtype=bool)  # update everything
        
        # Pass update_mask to your update function
        self.pixel_logic.update(frame, matches, any_match, best_k, update_mask)

        return fg_mask

        # Post Processing step done after numerical validation, benchmarking and CLI 

        # 1. ----------- Morphological Operations Post Processing -----------
        # Before morphology, use a 5x5 or 7x7 median blur
        fg_mask = cv2.medianBlur(fg_mask, 5) 

        # Opening removes noise, Closing fills holes
        # 1. Morphological Opening: Removes small white noise (dots) of size less than kernel
        # 2. Morphological Closing: Fills small black holes inside objects
        
        # 2. ------------- CLEANING BACKGROUND (Opening - fill white areas ) --------------
        # cluster of noise survived the blur, erosion will shirnk them untill they disappear
        open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
        # anything smaller than 5x5 is considered not an object
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, open_kernel)

        # 3. ------------- BRIDGING GAPS (Closing - expand white areas) --------------
        # moving objects appear broken, closing bridges these small gaps
        close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10,10))
        # if your gap is larger than 7 pixels it won't be bridged increase it
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, close_kernel)
        
        # 4. ------------- HOLE FILLING (OPTIONAL BUT POWERFUL) ----------------
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        solid_mask = np.zeros_like(fg_mask)

        for cnt in contours:
            cv2.drawContours(solid_mask, [cnt], -1, 255, thickness=-1)
        
        fg_mask = solid_mask

        # 5. ----- Component Filtering: Remove tiny blobs that survived -------
        # This kills any remaining noise that survived the kernel
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(fg_mask)

        # Create a blank mask to keep only significant objects
        clean_mask = np.zeros_like(fg_mask)

        for i in range(1, num_labels):             # Start from 1 to skip background
            area = stats[i,cv2.CC_STAT_AREA]
            if area > 100:         # Adjust based on object size (e.g., bowling ball)
                clean_mask[labels == i] = 255

        return clean_mask

         