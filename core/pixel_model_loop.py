# Objective

# Each pixel (x, y) stores:
    # Mean μ
    # Variance σ²
    # Weight = 1 (constant)

# Update using Exponential Moving Average (EMA)

# This builds intuition for:
    # Background adaptation
    # Noise
    # Ghosting


# You generally don't calculate these for a single pixel in a single frame 
# because a single number (like 128) doesn't have a spread. 
# Instead, you calculate these stats across regions or time: 
    # Spatial Stats (Across an area): You take a 5X5 block of pixels and calculate
        # the mean/std dev to understand the local texture or brightness.
    # Temporal Stats (Across time): You look at the same pixel across 100 frames of video.
        # Mean: The average brightness of that spot over time.
        # Std Dev: How much that pixel "flickers." High std dev at one pixel often indicates motion or sensor noise.

# 1. Building the IntuitionInstead of seeing a pixel as just a "color," you are seeing it as a history. 

    # • Mean (): Represents the "Permanent" background (e.g., the road, the wall). 
    # • Variance (): Represents the "Noise" or "Stability" (e.g., leaves rustling in the wind have high variance; a solid wall has low variance). 
    # • EMA (Exponential Moving Average): This allows the model to "forget" old frames and "adapt" to new ones (like the sun shifting or a car parking). 

# 2. The Math (The "Update" Step)For every new frame, you don't recalculate from scratch. You update the existing values using a learning rate ($\alpha$), typically a small value like 0.01 or 0.05: 

    # • Update Mean: $\mu_{\text{new}} = (1 - \alpha) \cdot \mu_{\text{old}} + \alpha \cdot \text{Pixel}_{\text{current}}$ 
    # • Update Variance: $\sigma_{\text{new}}^{2} = (1 - \alpha) \cdot \sigma_{\text{old}}^{2} + \alpha \cdot (\text{Pixel}_{\text{current}} - \mu_{\text{new}})^{2}$ 

# 3. Explaining the Phenomena 

    # • Background Adaptation: If a person stands perfectly still for 5 minutes, $\alpha$ eventually incorporates them into the $\mu$. They "become" the background. 
    # • Noise: If $\sigma^{2}$ is high, it means the pixel value is jumping around. If you are doing motion detection, you ignore pixels where $(\text{Pixel}_{\text{current}} - \mu) < 3\sigma$. 
    # • Ghosting: When a parked car drives away, the $\mu$ still contains the "image" of the car. Until the EMA updates enough times to show the road underneath, a "ghost" of the car remains. 

import numpy as np

# ------------------Step 2-------------------
class SingleGaussianPixel:
    def __init__(self, init_value: float, alpha: float = 0.01):
        self.mean = init_value
        self.var = 0.01
        self.alpha = alpha

    def update(self, x:float):
        """
        Exponential Moving Average update
        """
        diff = x - self.mean
        self.mean += self.alpha * diff
        self.var +=  self.alpha * (diff**2-self.var)

    def is_foreground(self, x: float, thrshold: float = 2.5) -> bool:
        """
        Foreground if |x-mean| > k * std
        """
        std = np.sqrt(self.var) + 1e-6
        return abs(x-self.mean) > thrshold * std
    
# Single Gaussian fails because:
#     A pixel can have multiple valid values over time
#         Tree leaves
#         Water
#         Monitor flicker
#         Shadows
#     Each pixel = mixture of K Gaussians

# For each pixel:
#     One Gaussian ≈ “road”
#     One Gaussian ≈ “shadow”
#     One Gaussian ≈ “reflection”
#     Foreground = value that doesn’t fit dominant background modes


# ----------------Step 3---------------------

class Gaussian:
    def __init__(self, mean, var, weight):
        self.mean = mean
        self.var = var
        self.weight = weight

# ------Per-frame logic (pixel-wise)-------
# For incoming pixel value x:
# 1. Match against existing Gaussians
#            |x - mu_k| <= T * sigma_k
# 2. If matched:
#     Update μ, σ²
#     Increase weight
# 3. If not matched:
#     Replace weakest Gaussian with new one
# 4. Normalize weights

# α — Learning Rate
#     α	Effect
#     Large ->	Fast adaptation, ghosting
#     Small	->  Stable background, lag

# K — Number of Gaussians
#     K	Effect
#     Small	->  Misses multimodal behavior
#     Large	->  Slower, more memory

MIN_VAR = 1e-4       

def update_variance(var: float, diff: float, alpha: float) -> float:
    """
    Numerically stable EMA variance update.
    var  : previous variance
    diff : (x - mean)
    """
    var_new = var + alpha * (diff * diff - var)
    return max(var_new, MIN_VAR)

def safe_std(var: float) -> float:
    if not np.isfinite(var):
        return np.sqrt(MIN_VAR)
    return np.sqrt(max(var, MIN_VAR))


class GMMPixel:
    def __init__(self, init_value, K=3, alpha=0.01, init_var: float = 0.01, 
                 threshold : float = 2.5, bg_threshold : float = 0.7):
        self.K = K
        self.alpha = alpha
        self.threshold = threshold
        self.bg_threshold = bg_threshold

        # initialized 1st gaussian which is known at start
        self.gaussians = [
            Gaussian(init_value, init_var, 1.0)
        ]

        # initialized 2nd and 3rd gaussians, empty slots having weight 0.0
        for _ in range(K-1):
            self.gaussians.append(
                Gaussian(0.0, init_var, 0.0)
            )

    # def match(self, x, threshold = 2.5):
    #     for g in self.gaussians:
    #         if abs(x - g.mean) <= threshold * np.sqrt(g.var):
    #             return g
    #     return None
    # THE ABOVE MATCH RETURNS THE FIRST MATCH NOT THE BEST MATCH CAUSING WRONG GAUSSIAN UPDATE

    # Match the Gaussian with minimum Mahalanobis distance.
    def get_best_match(self, x):
        best = None
        best_dist = float("inf")
        x = float(x)
        for g in self.gaussians:
            std = safe_std(g.var)
            dist = float(abs(x - g.mean) / std)

            if dist < self.threshold and dist < best_dist:
                best = g
                best_dist = dist

        return best

    
    def update(self, x, matched):
        # Decay Weights
        for g in self.gaussians:
            g.weight *= (1-self.alpha)
        # HERE, What this means intuitively
        #     Every Gaussian slowly forgets
        #     The matched Gaussian gets a probability boost
        #     Over time:
        #         Dominant background modes emerge naturally
        #         Rare modes disappear smoothly

        if matched is not None:
            # update matched Gaussian
            diff = x - matched.mean
            matched.mean += self.alpha * diff
            matched.var = update_variance(matched.var, diff, self.alpha)
            matched.weight += self.alpha
        else:
            weakest = min(self.gaussians, key = lambda g: g.weight)
            weakest.mean = x
            weakest.var = max(0.02,MIN_VAR)
            weakest.weight = 0.05

    # IT'S BETTER TO DECAY WEIGHT FIRST THE REWARD
        # # Decay Weights
        # for g in self.gaussians:
        #     g.weight *= (1-self.alpha)

        # Normalize weights
        total = sum(g.weight for g in self.gaussians)
        for g in self.gaussians:
            g.weight /= total
    
    # --------------------Step 4--------------------

    # Objective:
    # Decide which Gaussian represent background

    # 1. sort Gaussian by: W_k/sigma_k
        #  why?
        #     High weight -> frequent
        #     Low variance -> stable

    # Background selection rule
    # let cumulative weight:
    #        sum(k=1 to B) w_k >= T_bg
    # Those first B Gaussians = background

    def is_background(self, matched):
        if matched is None:
            return False
        
        self.gaussians.sort(
            key=lambda g: g.weight / safe_std(g.var),
            reverse=True
        )

        cumulative = 0.0
        for g in self.gaussians:
            cumulative += g.weight
            if g is matched:
                return True
            if cumulative > self.bg_threshold:
                break

        return False


    




