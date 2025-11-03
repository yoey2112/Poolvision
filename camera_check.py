import cv2

def find_working_camera(max_index: int = 5):
    """
    Try camera indices 0..max_index and return the first that opens.
    """
    for i in range(max_index + 1):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap is not None and cap.isOpened():
            cap.release()
            return i
    return None

def main():
    print("Searching for a working camera...")
    cam_index = find_working_camera()
    if cam_index is None:
        print("No camera found. Make sure your USB webcam is connected.")
        return

    print(f"Using camera index: {cam_index}")
    cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)

    # Try a reasonable starting resolution/FPS; we’ll refine later.
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cap.set(cv2.CAP_PROP_FPS, 30)

    if not cap.isOpened():
        print("Failed to open camera.")
        return

    print("Press 'q' to quit the preview window.")
    while True:
        ok, frame = cap.read()
        if not ok:
            print("Failed to read a frame from the camera.")
            break

        cv2.imshow("PoolVision - Camera Check", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
