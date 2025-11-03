import json
from pathlib import Path
from typing import List, Tuple, Optional

import cv2
import numpy as np

from pv.camera import Camera

CALIBRATION_PATH = Path(__file__).resolve().parent.parent / "config" / "calibration.json"

# ===== Utility =====

def order_corners(pts: np.ndarray) -> np.ndarray:
    """
    Order 4 points as TL, TR, BR, BL (clockwise starting at top-left).
    pts: shape (4,2)
    """
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).ravel()

    rect[0] = pts[np.argmin(s)]   # top-left
    rect[2] = pts[np.argmax(s)]   # bottom-right
    rect[1] = pts[np.argmin(diff)]  # top-right
    rect[3] = pts[np.argmax(diff)]  # bottom-left
    return rect

def largest_quad_contour(image: np.ndarray) -> Optional[np.ndarray]:
    """
    Find the largest 4-point contour in the image (by area).
    Returns a 4x2 array of points or None.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Mild blur + edge detect
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    # Dilate a bit to close gaps
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    best_area = 0.0

    for cnt in contours:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) != 4:
            continue
        area = cv2.contourArea(approx)
        if area < 10000:  # ignore tiny quads
            continue

        # Filter by aspect ratio near 2:1 (loose bounds)
        pts = approx.reshape(-1, 2).astype(np.float32)
        rect = order_corners(pts)
        w = np.linalg.norm(rect[1] - rect[0])  # top width
        h = np.linalg.norm(rect[3] - rect[0])  # left height
        if h == 0:
            continue
        ratio = w / h if w > h else h / w  # treat either orientation
        # Pool tables ~ 2:1 rectangle; allow loose range
        if 1.6 <= ratio <= 2.4 and area > best_area:
            best_area = area
            best = rect

    return best

def warp_perspective(image: np.ndarray, corners: np.ndarray, out_w=2000, out_h=1000) -> Tuple[np.ndarray, np.ndarray]:
    """
    Warp the image using the 4 ordered corners to a top-down rectangle of out_w x out_h.
    Returns (warped, H)
    """
    dst = np.array([[0, 0],
                    [out_w - 1, 0],
                    [out_w - 1, out_h - 1],
                    [0, out_h - 1]], dtype="float32")
    H = cv2.getPerspectiveTransform(corners.astype("float32"), dst)
    warped = cv2.warpPerspective(image, H, (out_w, out_h))
    return warped, H

def save_calibration(H: np.ndarray, out_size: Tuple[int, int], corners_src: np.ndarray):
    CALIBRATION_PATH.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "homography": H.tolist(),
        "output_size": {"width": int(out_size[0]), "height": int(out_size[1])},
        "source_corners": corners_src.tolist()
    }
    with open(CALIBRATION_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"[calibration] Saved to {CALIBRATION_PATH}")

# ===== Manual click helper =====

class ClickCollector:
    def __init__(self, image: np.ndarray):
        self.image = image.copy()
        self.clone = image.copy()
        self.points: List[Tuple[int, int]] = []

    def callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.points.append((x, y))
            cv2.circle(self.image, (x, y), 6, (0, 255, 255), -1)
            cv2.putText(self.image, str(len(self.points)), (x + 8, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

    def collect(self, window_name="Click 4 corners (TL, TR, BR, BL)"):
        print("[manual] Click corners in order: TL, TR, BR, BL. Press 'r' to reset, 'Enter' to accept, or 'q' to cancel.")
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(window_name, self.callback)
        while True:
            cv2.imshow(window_name, self.image)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('r'):
                self.image = self.clone.copy()
                self.points.clear()
                print("[manual] Reset clicks.")
            elif key in (13, 10):  # Enter
                if len(self.points) == 4:
                    cv2.destroyWindow(window_name)
                    return np.array(self.points, dtype="float32")
                else:
                    print("[manual] Need exactly 4 points.")
            elif key == ord('q'):
                cv2.destroyWindow(window_name)
                return None

# ===== Main flow =====

def main():
    # 1) Open camera and take a snapshot
    cam = Camera(index="auto", width=1920, height=1080, fps=30, backend_name="dshow", flip=False)
    cam.open()
    print("Preview: press 's' to capture snapshot for calibration, 'q' to quit.")
    cv2.namedWindow("Calibration Preview", cv2.WINDOW_NORMAL)

    snapshot = None
    while True:
        ok, frame = cam.read()
        if not ok: break
        display = frame.copy()
        cv2.putText(display, "Press 's' to snapshot, 'q' to quit", (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)
        cv2.imshow("Calibration Preview", display)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            cam.release()
            cv2.destroyAllWindows()
            return
        if key == ord('s'):
            snapshot = frame.copy()
            break

    cam.release()
    cv2.destroyAllWindows()

    if snapshot is None:
        print("No snapshot taken.")
        return

    # 2) Try automatic table detection
    print("[auto] Detecting table...")
    # Work on a downscaled copy for robustness
    scale = 1280.0 / snapshot.shape[1] if snapshot.shape[1] > 1280 else 1.0
    snap_small = cv2.resize(snapshot, (int(snapshot.shape[1]*scale), int(snapshot.shape[0]*scale)), interpolation=cv2.INTER_AREA)

    corners_small = largest_quad_contour(snap_small)
    corners_full = None
    if corners_small is not None:
        # Map back to full-res
        corners_full = (corners_small / scale).astype("float32")

        # Show overlay for review
        overlay = snapshot.copy()
        pts = corners_full.astype(int).reshape(-1, 1, 2)
        cv2.polylines(overlay, [pts], isClosed=True, color=(0, 255, 0), thickness=3)
        blend = cv2.addWeighted(snapshot, 0.6, overlay, 0.4, 0)
        cv2.imshow("[auto] Detected table (press 'a' accept, 'm' manual, 'r' retry, 'q' quit)", blend)
        while True:
            key = cv2.waitKey(0) & 0xFF
            if key == ord('a'):  # accept
                break
            elif key == ord('m'):
                corners_full = None  # force manual
                break
            elif key == ord('r'):
                cv2.destroyAllWindows()
                return main()  # restart flow
            elif key == ord('q'):
                cv2.destroyAllWindows()
                return
        cv2.destroyAllWindows()

    # 3) Manual fallback if needed
    if corners_full is None:
        collector = ClickCollector(snapshot)
        pts = collector.collect()
        if pts is None or len(pts) != 4:
            print("[manual] Cancelled.")
            return
        # User has provided TL, TR, BR, BL already
        corners_full = pts.astype("float32")

    # 4) Warp to top-down and save calibration
    OUT_W, OUT_H = 2000, 1000  # 2:1 canvas for standardized analyses
    warped, H = warp_perspective(snapshot, corners_full, OUT_W, OUT_H)

    cv2.imshow("Top-down (preview)", warped)
    print("[calibration] Press 'Enter' to save, 'r' to retry, or 'q' to abort.")
    while True:
        key = cv2.waitKey(0) & 0xFF
        if key in (13, 10):  # Enter
            save_calibration(H, (OUT_W, OUT_H), corners_full)
            break
        elif key == ord('r'):
            cv2.destroyAllWindows()
            return main()
        elif key == ord('q'):
            cv2.destroyAllWindows()
            return

    cv2.destroyAllWindows()
    print("[calibration] Done.")

if __name__ == "__main__":
    main()
