import argparse
import os
import cv2
import yaml
from app.visualize import show_frame
from core.gmm import SingleGaussianBackground, GMMBackground, GMMBackgroundVectorized
from core.pixel_model_cpp_backend import GMMBackgroundCpp

def load_config(path):
    """
    The Reader: Loads parameter from the YAML file.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, 'r') as file:
        return yaml.safe_load(file)

def run_video(video_path: str, camera_index : int, scale : float, user_config: dict, mode : str):
    """
    The Logic: Processes the video using GMM model.
    """
    if camera_index is not None:
        cap = cv2.VideoCapture(camera_index)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        print(f"[INFO] Using camera index : {camera_index}")
    else:
        cap = cv2.VideoCapture(video_path)
        print(f"[INFO] Using video file: {video_path}")

    if not cap.isOpened():
        src = f"camera {camera_index}" if camera_index is not None else video_path
        raise IOError(f"Cannot open Source: {src}")
    
    #------------------Metadata---------------------
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)

    print(f"[INFO] Resolution: {width} x {height}")
    print(f"[INFO] FPS: {fps}")

    bg_model = None

    #----------Video Precessing Rule--------------
    while True:     # Loop the video frame by frame until the end of the video
        
        #---------reads the next frame from the video-----------
        ret, frame = cap.read()         
        # ret: boolean -> is True if a frame was read successfully
        # frame: numpy array -> representing the image (typically in BGR color format)
        
        if not ret:
            break

        # --------Frame Processing-------
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Converts the color frame (BGR format) into a single-channel grayscale image.

        # Normalize to [0, 1]
        gray_norm = gray.astype("float32") / 255.0

        # Pixel range check
        # print(f"Pixel range: min={gray_norm.min():.3f}, max = {gray_norm.max():.3f}", end="\r")

        # Downscaling for faster FPS
        downscale_frame = cv2.resize(gray_norm, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        # INTER_AREA
          # Best for downsampling
          # Reduces aliasing

        # --------------Background Masking - Initialize model (ONCE, at small resolution)-----------------
        if bg_model is None:
            if mode == "cpp":
                bg_model = GMMBackgroundCpp(downscale_frame, **user_config)
            elif mode == 'vec':
                bg_model = GMMBackgroundVectorized(downscale_frame, **user_config)
            else:
                bg_model = GMMBackground(downscale_frame, **user_config)
            continue

        # Process and Display : Run Background Subtraction
        # fg_mask = bg_model.apply(gray_norm)
        small_mask = bg_model.apply(downscale_frame)

        # Upscale mask back
        fg_mask = cv2.resize(small_mask, (gray_norm.shape[1], gray_norm.shape[0]), interpolation=cv2.INTER_NEAREST)
                # Inter_NEAREST)
                  # Preserves binary values
                  # No ghost pixels
                  # No gray edges

        #-------------Display and User Input--------------
        show_frame(gray_norm)       # dispaly the original grayscale frame
        cv2.imshow("Foreground Mask", fg_mask) # Dispaly Foreground mask frame

        if cv2.waitKey(30) & 0xFF == 27:   # ESC
            break
        # cv2.waitKey(30) waits for a key press for 30 milliseconds.
        # 0xFF == 27 checks if the pressed key was the ESC key (ASCII value 27).

        
    cap.release()   # Releases the video file handler, freeing up resources.
    cv2.destroyAllWindows() # Closes all the OpenCV windows that were opened during execution.


def main():
    # Argparse is the listner
    parser = argparse.ArgumentParser(description="GMM Background Subtraction Tool")

    # Path Arguments
    parser.add_argument("--video", type=str, default=None,
                        help="Path to the input video file")
    parser.add_argument("--config", type=str, default="configs/gmm.yaml",
                        help="Path to the YAML configuration file (default: %(default)s)")
    parser.add_argument("--camera", type = int, default= None,
                        help="Camera index (e.g. 0, 1, 2 ....)")
    parser.add_argument("--scale", type=float, default=1.0,
                        help="scale model for performance (e.g. 0.5, 0.3....)")
    
    # Execution Arguments
    parser.add_argument("--mode", type = str, choices = ['vec', 'loop', 'cpp'], default = 'vec',
                        help = "Execution mode: vectorized or loop (default: %(default)s)")
    parser.add_argument("--show", action = "store_true", default = True,
                        help = "Display the video windows")
    
    args = parser.parse_args()
    # Namespace(video='bowling.mp4', config='configs/gmm.yaml', mode='vec', show=True)

    if not (0 < args.scale <= 1.0):
        parser.error("--scale must be in (0, 1]")

    if args.video is None and args.camera is None:
        parser.error("Either -- video or --camera must be provided")
    if args.video is not None and args.camera is not None:
        print("[INFO] Camera Provided, Video will be ignored")

    # Config is the Reader
    # Load the dictionary using the path from terminal 
    user_Config = load_config(args.config)

    print(f"[INFO] Starting GMM in {args.mode} mode")
    # print(f"[INFO] Video: {args.video}")
    print(f"[INFO] Config: {user_Config}")

    # Pass the terminal inputs into the logic function
    run_video(video_path=args.video, camera_index = args.camera, scale = args.scale, user_config=user_Config, mode = args.mode)

if __name__ == "__main__":
    main()

# Run Video to proove maths

# def load_config(path = "configs/gmm.yaml"):
#     with open(path, 'r') as file:
#         return yaml.safe_load(file)

# def run_video(video_path: str):
#     user_input = load_config()

#     cap = cv2.VideoCapture(video_path)

#     if not cap.isOpened():
#         raise IOError(f"Cannot open video: {video_path}")
    
#     #------------------Metadata---------------------
#     width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     fps    = cap.get(cv2.CAP_PROP_FPS)

#     print(f"[INFO] Resolution: {width} x {height}")
#     print(f"[INFO] FPS: {fps}")

#     bg_model = None

#     #----------Video Precessing Rule--------------
#     while True:     # Loop the video frame by frame until the end of the video
        
#         #---------reads the next frame from the video-----------
#         ret, frame = cap.read()         
#         # ret: boolean -> is True if a frame was read successfully
#         # frame: numpy array -> representing the image (typically in BGR color format)
        
#         if not ret:
#             break

#         # --------Frame Processing-------
        
#         # Convert to grayscale
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         # Converts the color frame (BGR format) into a single-channel grayscale image.

#         # Normalize to [0, 1]
#         gray_norm = gray.astype("float32") / 255.0

#         # Pixel range check
#         # print(f"Pixel range: min={gray_norm.min():.3f}, max = {gray_norm.max():.3f}", end="\r")

#         #-------------Display and User Input--------------
#         show_frame(gray_norm)       # dispaly the normalized grayscale frame in a window

#         # --------------Background Masking-----------------
#         if bg_model is None:
#             # bg_model = SingleGaussianBackground(gray_norm)
#             # bg_model = GMMBackground(gray_norm)
#             bg_model = GMMBackgroundVectorized(gray_norm, **user_input)
#             continue

#         fg_mask = bg_model.apply(gray_norm)

#         cv2.imshow("Foreground Mask", fg_mask)


#         if cv2.waitKey(100) & 0xFF == 27:   # ESC
#             break
#         # cv2.waitKey(30) waits for a key press for 30 milliseconds.
#         # 0xFF == 27 checks if the pressed key was the ESC key (ASCII value 27).

        
#     cap.release()   # Releases the video file handler, freeing up resources.
#     cv2.destroyAllWindows() # Closes all the OpenCV windows that were opened during execution.


# if __name__ == "__main__":
#     # video path OUTSIDE computer_vision folder
#     run_video("data/videos/bowling.mp4")
#     # run_video("data/videos/nightout.mp4")

