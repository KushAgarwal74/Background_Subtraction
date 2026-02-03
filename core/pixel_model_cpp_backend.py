from background_subtraction_gmm import gmm_cpp

class GMMBackgroundCpp:
    def __init__(self, first_frame, k: int = 3, alpha:float = 0.01,
                 threshold : float = 2.5, bg_threshold: float = 0.7):
        self.cppModel = gmm_cpp.GMMModel(first_frame, k, alpha, threshold, bg_threshold)

    def apply(self, frame):
        return self.cppModel.apply(frame)