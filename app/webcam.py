# app/webcam.py
import cv2

def list_cameras(max_index=10):
    print("[INFO] Scanning camera indices...")
    for i in range(max_index):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            print(f"[INFO] Camera index {i} is available")
            cap.release()
            break

if __name__ == "__main__":
    list_cameras()
