import numpy as np

min_var = 1e-4

class GMMPixelVectorized:
    def __init__(self, first_frame: np.ndarray, k: int = 3, alpha:float = 0.01,
                threshold : float = 2.5 , bg_threshold : float = 0.7):
        self.k = k
        self.alpha = alpha
        self.threshold = threshold
        self.bg_threshold = bg_threshold
        self.h , self.w = first_frame.shape

        # Stacks : (K, H, W)
        self.means = np.zeros((k, self.h, self.w), dtype="float32")
        self.vars = np.full((k, self.h, self.w), 0.01, dtype="float32")
        self.weights = np.zeros((k, self.h, self.w), dtype="float32")

        # Initialize the 1 Gaussian frame values
        self.means[0] = first_frame
        self.weights[0] = 1.0

    def get_best_match(self, frame):
        # Calculate distance for all K layers
        stds = np.sqrt(np.maximum(self.vars, min_var))
        distance = np.abs(frame - self.means) / stds
        # dists = np.array([10.5, 2.1, 1.8])

        # boolean masks
        matches = distance < self.threshold
        # matches -> [False, True, True]
        any_match = np.any(matches, axis = 0)   # simple 2D boolean array
        # any_match -> True

        # The Goal: You only need one of the Gaussians in your mixture to match the
        # pixel for it to be considered part of the background.
        # The Result:
            # True: At least one Gaussian matched -> Background.
            # False: No Gaussians matched -> Foreground/Moving Object.

        # Find best K index (closest match)
        # We fill non-matches with infinity so argmin ignores them
        temp_dists = np.where(matches, distance, np.inf)
        # temp_dists -> [inf, 2.1, 1.8]
        best_k = np.argmin(temp_dists, axis=0)    # which layer has the smallest value 
        # best_k -> 2

        return matches, any_match, best_k, distance

    def update(self, frame, matches, any_match, best_k, update_mask):
        # 1. -------------- Decay weigth-------------
        self.weights[:,update_mask] *= (1.0-self.alpha)

        # 2. --------------Update Matched--------------
        # for k in range(self.k):
        #     mask = (matches[k]) & (best_k == k)     # bitwise & operator
        #     # best_k == k: Tells you "This component is the closest."
        #     # matches[k]: Tells you "This component is close enough."
        #     if np.any(mask):
        #         diff = frame[mask] - self.means[k,mask]
        #         self.means[k,mask] += self.alpha * diff
        #         self.vars[k, mask] += self.alpha*(diff**2 - self.vars[k, mask])
        #         self.weights[k,mask] += self.alpha

        # -----------------------------------------------
        # above loop is competetive learning
        # below is the 3D mask to eliminate the loop and optimize memory operation
        # -----------------------------------------------

        # 1. --------------Create the 3D mask (K, H, W)--------------
        # This compares the index of each layer (0 to K-1) against the best_k values
        k_indices = np.arange(self.k)[:, np.newaxis, np.newaxis] # Shape (K, 1, 1)
        mask_3d = (matches) & (best_k[None,:,:] == k_indices) & (update_mask[None,:,:])   # Shape (K, H, W)

        # 2. --------------Update everything in one go---------------
        # NumPy will only update the (k, y, x) positions where mask_3d is True
        diff = frame - self.means  # Broadcaster frame (H,W) to (K,H,W)
        self.means[mask_3d] += self.alpha * diff[mask_3d]
        self.vars[mask_3d] += self.alpha * (diff[mask_3d]**2 - self.vars[mask_3d])
        self.weights[mask_3d] += self.alpha

        # 3. --------------Replace Unmatched (weakest)--------------
        no_match = (~any_match)
        
        if np.any(no_match):
            weakest_k = np.argmin(self.weights, axis=0)
            # Advanced indexing
            rows, cols = np.where(no_match)
            k_idx = weakest_k[no_match]
            
            self.means[k_idx, rows, cols] = frame[rows, cols]
            self.vars[k_idx, rows, cols] = 0.02
            self.weights[k_idx, rows, cols] = 0.05

            # I'm not sure what this new color is yet, so I'll give it a wide range (variance)
            # but low importance (weight). If it stays here for many frames,
            # its weight will grow until it eventually becomes part of the background."

        # -----------------------------------------------
        # above loop is competetive learning
        # below is the 3D mask to eliminate the loop and optimize memory operation
        # -----------------------------------------------

        # 2. Create the 3D mask for replacement
        # Condition 1: The pixel matched nothing (2D -> 3D)
        # Condition 2: This specific layer is the weakest (2D == 3D -> 3D)
        # weakest_k = np.argmin(self.weights,axis=0)
        # no_match_3d = (no_match) & (weakest_k == k_indices)

        # # 3. Update everything in one shot
        # self.means[no_match_3d] = frame[no_match] # Broadcasting frame[no_match]
        # self.vars[no_match_3d] = 0.02
        # self.weights[no_match_3d] = 0.05

        # 4. -------------- Normalize--------------
        self.weights /= (np.sum(self.weights, axis=0) + 1e-8) 
    
    def is_background(self, any_match, best_k):
        # 1. -------------- Sort fitness weight/sqrt(var)--------------
        # 1. Get the sort indices (K, H, W)
        fitness = self.weights/np.sqrt(np.maximum(self.vars, min_var))
        rank = np.argsort(-fitness, axis=0)     # high to low

        # fg_mask = np.ones((self.h, self.w), dtype="uint8") * 255
        # cum_weight = np.zeros((self.h, self.w), dtype="float32")

        # # Step through sorted Gaussians
        # for i in np.arange(self.k):
        #     k_layer = rank[i]
        #     # Use NumPy indices to get weight values across the grid
        #     r, c = np.indices((self.h, self.w))
        #     cum_weight += self.weights[k_layer, r, c]
            
        #     # If current frame matched this layer and it's within T_bg, it's BG
        #     is_bg = (best_k == k_layer) & (any_match)
        #     fg_mask[is_bg] = 0
            
        #     if np.all(cum_weight >= bg_threshold): break

        # -----------------------------------------------
        # above loop is competetive learning and creates two massive arrays which are time taking
        # below is the 3D mask to eliminate the loop and optimize memory operation
        # -----------------------------------------------

        # 2. Reorder weights so Layer 0 is the "Fittest"
        sorted_weights = np.take_along_axis(self.weights, rank, axis=0)

        # 3. Calculate running totals of weights down the stack
        cum_weights = np.cumsum(sorted_weights, axis=0)     # [0.5, 0.4, 0.3]
        # Shift down so the first layer always sees '0' as its previous sum
        prev_cum_weights = np.zeros_like(cum_weights)       # [0.0, 0.0, 0.0]
        prev_cum_weights[1:] = cum_weights[:-1]             # [0.0, 0.5, 0.4]

        # 4. Identify which layers are "Background Quality"
        # According to the paper, the first B distributions that sum to bg_threshold
        is_bg_layer = prev_cum_weights < self.bg_threshold

        # 5. Create a mask of the actual winning layer in the sorted stack
        # We need to know where the 'best_k' ended up after the sort
        k_indices = np.arange(self.k)[:, np.newaxis, np.newaxis]
        # was_best_match = (rank == k_indices)
        was_best_match = (rank == best_k[None, :, :])  # (K,H,W)

        # 6. Combined logic: Did it match? AND is that layer part of the stable BG?
        background_mask = np.any(was_best_match & is_bg_layer, axis=0) &  any_match

        # 7. Convert to the uint8 0/255 mask
        fg_mask = np.where(background_mask, 0, 255).astype("uint8")

        return fg_mask