import numpy as np
import cv2
import time
from core.gmm import GMMBackground as LoopModel, GMMBackgroundVectorized as VecModel
from core.pixel_model_cpp_backend import GMMBackgroundCpp as CppModel

def run_parity_test(video_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")
    
    ret, frame = cap.read()

    if not ret: 
        print("Error: could not read video frame.")
        return 
    
    # 1. Prepare Frame (small size recommended for the loop version)
    # Testing 780p with loops will take forever, resize to 100x100 for test
    test_size = (100,100)
    frame_small = cv2.resize(frame, test_size)
    gray = cv2.cvtColor(frame_small, cv2.COLOR_BGR2GRAY).astype("float32")/255

    # 2. Initialize both models with the same frame and parameters
    params = {"k":3, "alpha":0.01}
    loop_model = LoopModel(gray, **params)
    vec_model  = VecModel(gray, **params)

    print(f"--- Starting Parity Test ({test_size[0]}x{test_size[1]}) ---")

    # 3. Process the NEXT frame
    ret, frame = cap.read()
    frame_small = cv2.resize(frame, test_size)
    gray_next = cv2.cvtColor(frame_small, cv2.COLOR_BGR2GRAY).astype("float32")/255

    # Time the loop Version
    start = time.time()
    mask_loop = loop_model.apply(gray_next)
    loop_time = time.time() - start

    # Time the Vectorized Version
    start = time.time()
    mask_vec = vec_model.apply(gray_next)
    vec_time = time.time() - start

    # 4. Compare Results
    # Check what percentage of pixels are identical
    mismatch_count = np.sum(mask_loop != mask_vec)
    similarity = (1.0 - (mismatch_count / mask_loop.size)) * 100

    print(f"Loop Time: {loop_time:.4f}s")
    print(f"Vec Time:  {vec_time:.4f}s")
    print(f"Speedup:   {loop_time / vec_time:.1f}x")
    print(f"Pixel Similarity: {similarity:.2f}%")

    if similarity > 99.9:
        print("\n✅ SUCCESS: Vectorization changed performance, not behavior.")
    else:
        print("\n❌ FAILURE: Logic drift detected. Check your indexing.")

    cap.release()

def run_multi_frame_parity_test(video_path, test_frames = 100):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")
    
    ret, frame = cap.read()

    if not ret:
        print("Error: could not read video frame.")
        return 
    
    # 1. Setup
    test_size = (100,100)
    frame_small = cv2.resize(frame, test_size)
    gray = cv2.cvtColor(frame_small, cv2.COLOR_BGR2GRAY).astype("float32")/255

    # Initialize both with the same yaml-style params
    params = {"k":3, "alpha":0.01}
    loop_model = LoopModel(gray, **params)
    vec_model = VecModel(gray, **params)
    cpp_model = CppModel(gray, **params)

    total_time_loop = 0
    total_time_vec = 0
    total_time_cpp = 0

    similarities_LV = []
    similarities_VC = []
    similarities_LC = []
    
    print(f" --- Running Parity Test Over {test_frames} frames ---")

    for f in range(test_frames + 1):
        ret, frame = cap.read()

        if not ret: break

        # pre-process 
        gray_frame = cv2.cvtColor(cv2.resize(frame, test_size), cv2.COLOR_BGR2GRAY).astype("float32")/255

        # Time Loop
        t0 = time.time()
        mask_loop = loop_model.apply(gray_frame)
        total_time_loop += (time.time() - t0)

        # Time Vec
        t1 = time.time()
        mask_vec = vec_model.apply(gray_frame)
        total_time_vec += (time.time() - t1)

        # Time Cpp
        t2 = time.time()
        mask_cpp = cpp_model.apply(gray_frame)
        total_time_cpp += (time.time() - t2) 

        # Compare
        def similarity(a, b):
            return 100.0 * (1.0 - np.mean(a!=b))
        # mismatch = np.sum(mask_loop != mask_vec)
        # sim = (1 - (mismatch / mask_loop.size)) * 100
        # similarities.append(sim)

        similarities_LV.append(similarity(mask_loop, mask_vec))
        similarities_LC.append(similarity(mask_loop, mask_cpp))
        similarities_VC.append(similarity(mask_vec, mask_cpp))

        mismatch_mask = (mask_loop != mask_cpp).astype("uint8") * 255
        cv2.imshow("Loop vs Cpp Mismatch", mismatch_mask)

        if f % 10 == 0:
            print(f"Frame {f:03d} | Match_LV: {similarities_LV[-1]:.2f}% | Match_LC: {similarities_LC[-1]:.2f}% | "
                  f"Match_VC: {similarities_VC[-1]:.2f}%")

        if cv2.waitKey(100) & 0xFF == 27:   # ESC
            break
        
    # 2. Final Report
    avg_sim = np.mean(similarities_LC)
    print("\n" + "="*50)
    print(f"FINAL REPORT OVER                   {test_frames} FRAMES")
    print(f"Average Similarity Loop Vs Vec:     {np.mean(similarities_LV):.4f}%")
    print(f"Average Similarity Loop Vs Cpp:     {np.mean(similarities_LC):.4f}%")
    print(f"Average Similarity Vec vs Cpp:      {np.mean(similarities_VC):.4f}%")
    print(f"Total Loop Time:                    {total_time_loop:.2f}s")
    print(f"Total Vec Time:                     {total_time_vec:.2f}s")
    print(f"Total Cpp Time:                     {total_time_cpp:.2f}s")
    print(f"Total Speedup with Vec:             {total_time_loop / total_time_vec:.1f}x")
    print(f"Total Speedup_loop_vs_Cpp:          {total_time_loop / total_time_cpp:.1f}x")
    print(f"Total Speedup_Vec_vs_Cpp:           {total_time_vec / total_time_cpp:.1f}x")
    print("="*50)

    if avg_sim > 99.9:
        print("✅ STABILITY PROVEN: No behavioral drift over time.")
    else:
        print("❌ WARNING: Minor drift detected. Check floating point precision.")

    cap.release()


# if __name__ == "__main__":
#     run_parity_test("data/videos/bowling.mp4")

if __name__ == "__main__":
    run_multi_frame_parity_test("data/videos/nightout.mp4", test_frames= 100)

