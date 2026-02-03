import time
import cv2
import numpy as np


class GMMBenchmarker:
    def __init__(self, model, resolution = (640,480)):
        self.model = model
        self.res = resolution
        self.is_vectorized = hasattr(self.model, 'pixel_logic')
        self.isCpp = hasattr(self.model, "cppModel")
        self.stats_vector = {
            "preprocess": [],
            "match": [],
            "classify": [],
            "update": [],
            "total": []
        }
        self.stats_loop = {
            "preprocess": [],
            "logic": [],              # This will be the total time for the GMM math
            "total": []
        }
        self.stats_cpp = {
            "preprocess": [],
            "total": []
        }

    def benchmark_frame(self, frame):
        t0 = time.perf_counter()

        # A. Preprocessing
        frame_resized = cv2.resize(frame, self.res)
        gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY).astype("float32")/255
        t1 = time.perf_counter()

        if self.is_vectorized:
            # --------- VECTOR LOGIC -----------
            # B. Match (The Logic we vectorized)
            matches, any_match, best_k, dist = self.model.pixel_logic.get_best_match(gray)
            t2 = time.perf_counter()

            # C. Classify
            fg_mask = self.model.pixel_logic.is_background(any_match, best_k)
            t3 = time.perf_counter()

            # D. Update (The innovation + learning)
            update_mask = np.ones_like(any_match, dtype=bool)
            self.model.pixel_logic.update(gray, matches, any_match, best_k, update_mask)
            t4 = time.perf_counter()

            # Record timings
            self.stats_vector["preprocess"].append(t1-t0)
            self.stats_vector["match"].append(t2 - t1)
            self.stats_vector["classify"].append(t3-t2)
            self.stats_vector["update"].append(t4 - t3)
            self.stats_vector["total"].append(t4 - t0)
        elif self.isCpp:
            # --------- CPP LOGIC ---------
            # Call the standard apply() for total time
            fg_mask = self.model.apply(gray)
            t2 = time.perf_counter()
            
            # Record Loop Stats
            self.stats_cpp["preprocess"].append(t1 - t0)
            self.stats_cpp["total"].append(t2 - t0)
        else:
            # --------- LOOP LOGIC ---------
            # Call the standard apply() which contains the internals loops
            fg_mask = self.model.apply(gray)
            t2 = time.perf_counter()
            
            # Record Loop Stats
            self.stats_loop["preprocess"].append(t1 - t0)
            self.stats_loop["logic"].append(t2 - t1)
            self.stats_loop["total"].append(t2 - t0)

        return fg_mask
    
    def get_report(self):
        # Choose which dictionay to report based on model type
        stats = {}
        if self.is_vectorized:
            stats = self.stats_vector
        elif self.isCpp:
            stats = self.stats_cpp
        else:
            stats = self.stats_loop

        total_time = sum(stats["total"])
        if total_time == 0: return {"Error": "No frames processed"}

        fps = len(stats["total"]) / total_time
        
        report = {k: np.mean(v) * 1000 for k, v in stats.items() if k != "total"}
        report["FPS"] = fps

        return report


