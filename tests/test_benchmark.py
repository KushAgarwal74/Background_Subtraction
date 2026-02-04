import cv2
import pandas as pd 
from core.benchmark import GMMBenchmarker
from core.gmm import GMMBackground as LoopModel, GMMBackgroundVectorized as VecModel
from core.pixel_model_cpp_backend import GMMBackgroundCpp as CppModel

# 1. Define Resolution to test
# we include very small to standard HD
resolutions = [
    (160, 120),   # QQVGA
    (320, 240),   # QVGA
    (640, 480),   # VGA
    (800, 600),   # SVGA
    (1024, 768),  # XGA (Missing)
    (1280, 720),  # HD
    # (1920, 1080), # FHD (Missing)
    # (2560, 1440), # QHD (Missing)
    # (3840, 2160)  # 4K (Missing)
]

def run_scaling_benchmark(video_path):
    results = []
    scale = 0.5

    for res in resolutions:
        print(f"\n------- Testing Resolution: {res[0]}x{res[1]} -------")
        
        # Test both models at this resolution
        for name, model_class in [("Loop", LoopModel), ("Vectorized", VecModel), ("CPP", CppModel)]:
            
            # Open fresh video
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise IOError(f"Cannot open video: {video_path}")
            
            # Read first frame for THIS Model and resolution
            ret,first_frame = cap.read()
            if not ret:
                cap.release()
                continue
        
            # Initialize with the resized first frame
            init_gray = cv2.cvtColor(cv2.resize(first_frame,res), cv2.COLOR_BGR2GRAY).astype("float32")/255
        
            downscale = cv2.resize(init_gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if res[0] > 600 else init_gray
            model = model_class(downscale, k =3)

            bench = GMMBenchmarker(model, resolution=res)

            # Run for a fixed number of frames (loops take long so we limit frames)
            # num_test_frame = 5 if name == "Loop" and res[0]>600 else 30
            num_test_frame = 50

            for _ in range(num_test_frame):
                ret, frame = cap.read()
                if not ret: break
                bench.benchmark_frame(frame)
            
            report = bench.get_report()
            report["Model"] = name
            report["Resolution"] = f"{res[0]}x{res[1]}"
            results.append(report)
            print(f"{name} model finished: {report['FPS']:.2f} FPS")

    cap.release()
    return pd.DataFrame(results)

# Run and Display
if __name__ == "__main__":
    df = run_scaling_benchmark("data/videos/bowling.mp4")
    print("\nFINAL SCALING REPORT")
    # print(df[['Resolution', 'Model', 'FPS', 'logic']])
    print(df)