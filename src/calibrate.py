import json
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import cv2
import numpy as np

from pv.camera import Camera

OUT_DIR = Path(__file__).resolve().parent.parent / "config"
CALIBRATION_PATH = OUT_DIR / "calibration.json"

# ---------- Geometry helpers ----------

def line_from_points(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    """
    Return line in homogeneous form ax + by + c = 0 through two points [x,y].
    """
    x1, y1 = float(p1[0]), float(p1[1])
    x2, y2 = float(p2[0]), float(p2[1])
    a = y1 - y2
    b = x2 - x1
    c = x1 * y2 - x2 * y1
    v = np.array([a, b, c], dtype=np.float64)
    n = np.linalg.norm(v[:2]) or 1.0
    return v / n

def intersect_lines(l1: np.ndarray, l2: np.ndarray) -> np.ndarray:
    """
    Intersect two homogeneous lines. Returns [x,y] in Euclidean coords.
    """
    p = np.cross(l1, l2)
    if abs(p[2]) < 1e-9:
        raise ValueError("Parallel or nearly parallel lines.")
    return (p[:2] / p[2]).astype(np.float64)

def fit_line(points: List[Tuple[float, float]]) -> np.ndarray:
    """
    Robust line fit in ax+by+c=0 form from >=2 points via least squares.
    """
    pts = np.array(points, dtype=np.float64)
    # Fit y = m x + b -> ax + by + c = 0 with a=-m, b=1, c=-b
    X = np.vstack([pts[:,0], np.ones(len(pts))]).T
    m, b = np.linalg.lstsq(X, pts[:,1], rcond=None)[0]
    a = -m; bb = 1.0; c = -b
    v = np.array([a, bb, c], dtype=np.float64)
    n = np.linalg.norm(v[:2]) or 1.0
    return v / n

def order_corners_anyorder(pts4: np.ndarray) -> np.ndarray:
    """
    Order 4 points (any order) -> TL, TR, BR, BL.
    """
    pts = pts4.astype(np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    rect = np.zeros((4,2), dtype=np.float32)
    rect[0] = pts[np.argmin(s)]     # TL
    rect[2] = pts[np.argmax(s)]     # BR
    rect[1] = pts[np.argmin(d)]     # TR
    rect[3] = pts[np.argmax(d)]     # BL
    return rect

def compute_homography(src_rect: np.ndarray, out_w: int, out_h: int) -> np.ndarray:
    dst = np.array([[0,0],[out_w-1,0],[out_w-1,out_h-1],[0,out_h-1]], dtype=np.float32)
    H = cv2.getPerspectiveTransform(src_rect.astype(np.float32), dst)
    return H

def transform_points(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    pts = pts.astype(np.float64)
    ones = np.ones((pts.shape[0],1), dtype=np.float64)
    ph = np.hstack([pts, ones])
    tp = (ph @ H.T)
    tp = tp[:,:2] / tp[:,2:3]
    return tp

# ---------- UI helpers ----------

def resize_with_scale(img, max_w=1600):
    h, w = img.shape[:2]
    if w <= max_w:
        return img.copy(), 1.0
    s = max_w / float(w)
    return cv2.resize(img, (int(w*s), int(h*s)), interpolation=cv2.INTER_AREA), s

class ClickUI:
    """
    Collects clicked points with proper mapping from displayed (resized) coords back to full-res.
    """
    def __init__(self, image: np.ndarray, title: str, max_w: int = 1600):
        self.base_full = image.copy()
        self.overlay_full = image.copy()
        self.title = title
        self.max_w = max_w
        self.view, self.scale = resize_with_scale(self.overlay_full, self.max_w)
        self.points: List[Tuple[int,int]] = []

    def _rebuild(self):
        self.view, self.scale = resize_with_scale(self.overlay_full, self.max_w)

    def _view_to_full(self, x, y):
        fx = int(round(x / self.scale))
        fy = int(round(y / self.scale))
        h, w = self.base_full.shape[:2]
        fx = max(0, min(w-1, fx))
        fy = max(0, min(h-1, fy))
        return fx, fy

    def _draw_marker_full(self, pt_full: Tuple[int,int], idx: int, color=(0,255,255)):
        cv2.circle(self.overlay_full, pt_full, 8, color, -1)
        cv2.putText(self.overlay_full, str(idx), (pt_full[0]+10, pt_full[1]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)

    def _mouse(self, ev, x, y, flags, param):
        if ev == cv2.EVENT_LBUTTONDOWN:
            p = self._view_to_full(x,y)
            self.points.append(p)
            self._draw_marker_full(p, len(self.points))
            self._rebuild()

    def collect(self, need: int, instructions: str, color=(0,255,255)) -> List[Tuple[int,int]]:
        cv2.namedWindow(self.title, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.title, self._mouse)
        while True:
            im = self.view.copy()
            cv2.putText(im, instructions, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)
            cv2.putText(im, f"{len(self.points)}/{need}", (12, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)
            cv2.putText(im, "Enter=OK  r=reset  q=cancel", (12, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)
            cv2.imshow(self.title, im)
            k = cv2.waitKey(1) & 0xFF
            if k == ord('r'):
                self.overlay_full = self.base_full.copy()
                self.points.clear()
                self._rebuild()
            elif k in (13,10):  # Enter
                if len(self.points) == need:
                    cv2.destroyWindow(self.title)
                    return list(self.points)
            elif k == ord('q'):
                cv2.destroyWindow(self.title)
                return []

# ---------- Calibration flow ----------

def main():
    # 1) Snapshot
    cam = Camera(index="auto", width=1920, height=1080, fps=30, backend_name="dshow", flip=False)
    cam.open()
    print("Preview: s=snapshot, q=quit")
    cv2.namedWindow("Calibrate - Snapshot", cv2.WINDOW_NORMAL)

    snap = None
    while True:
        ok, frame = cam.read()
        if not ok: break
        v,_ = resize_with_scale(frame)
        cv2.putText(v, "Press 's' to snapshot, 'q' to quit", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)
        cv2.imshow("Calibrate - Snapshot", v)
        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            cam.release(); cv2.destroyAllWindows(); return
        if k == ord('s'):
            snap = frame.copy(); break
    cam.release(); cv2.destroyAllWindows()
    if snap is None:
        print("No snapshot taken."); return

    # 2) Collect 12 pocket rail-stop points (clockwise)
    """
    We need 2 points per pocket where the rail stops at the pocket mouth:
    Order (clockwise facing the table from the break side):
      1-2:  Top-Left pocket (point on TOP rail, then point on LEFT rail)
      3-4:  Top-Right pocket (TOP rail, then RIGHT rail)
      5-6:  Right-Middle pocket (point on RIGHT rail towards TOP, then RIGHT rail towards BOTTOM)
      7-8:  Bottom-Right pocket (RIGHT rail, then BOTTOM rail)
      9-10: Bottom-Left pocket (BOTTOM rail, then LEFT rail)
      11-12: Left-Middle pocket (LEFT rail towards BOTTOM, then LEFT rail towards TOP)
    """
    pocket_labels = [
        "TL pocket: TOP-rail point", "TL pocket: LEFT-rail point",
        "TR pocket: TOP-rail point", "TR pocket: RIGHT-rail point",
        "Right-MIDDLE: RIGHT-rail (toward TOP)", "Right-MIDDLE: RIGHT-rail (toward BOTTOM)",
        "BR pocket: RIGHT-rail point", "BR pocket: BOTTOM-rail point",
        "BL pocket: BOTTOM-rail point", "BL pocket: LEFT-rail point",
        "Left-MIDDLE: LEFT-rail (toward BOTTOM)", "Left-MIDDLE: LEFT-rail (toward TOP)",
    ]
    ui = ClickUI(snap, "Calibrate - Pockets")
    pocket_pts = []
    for label in pocket_labels:
        pts = ui.collect(1, f"Click {label}", color=(0,255,255))
        if not pts:
            print("Cancelled."); return
        pocket_pts.extend(pts)

    pocket_pts = np.array(pocket_pts, dtype=np.float64).reshape(12,2)

    # Assign to rails for line fitting
    # Top rail: TL-top (0), TR-top (2)
    # NOTE: side pockets live on long rails; they do NOT touch left/right rails.
    # We'll also add top-side helper points by projecting from mid pockets if helpful,
    # but for a stable fit, two points suffice because they are far apart.
    top_points = [tuple(pocket_pts[0]), tuple(pocket_pts[2])]
    bottom_points = [tuple(pocket_pts[8]), tuple(pocket_pts[7])]  # BL-bottom (8), BR-bottom (7)
    left_points = [tuple(pocket_pts[1]), tuple(pocket_pts[9]), tuple(pocket_pts[11])]   # TL-left (1), BL-left (9), L-mid (11)
    right_points = [tuple(pocket_pts[3]), tuple(pocket_pts[5]), tuple(pocket_pts[6])]   # TR-right (3), R-mid top (5), BR-right (6)

    L_top = fit_line(top_points)
    L_bottom = fit_line(bottom_points)
    L_left = fit_line(left_points)
    L_right = fit_line(right_points)

    # Intersections -> inner cloth corners TL, TR, BR, BL
    TL = intersect_lines(L_top, L_left)
    TR = intersect_lines(L_top, L_right)
    BR = intersect_lines(L_bottom, L_right)
    BL = intersect_lines(L_bottom, L_left)
    play_corners = np.array([TL, TR, BR, BL], dtype=np.float32)

    # 3) Collect 4 OUTER table corners (any order)
    ui2 = ClickUI(snap, "Calibrate - Outer corners")
    oc = ui2.collect(4, "Click 4 OUTER wooden corners (any order). Enter when done.")
    if not oc or len(oc) != 4:
        print("Cancelled."); return
    outer_corners = order_corners_anyorder(np.array(oc, dtype=np.float32))

    # 4) Build homography for play surface
    # Pick play size using measured aspect ratio
    wA = np.linalg.norm(TR - TL)
    wB = np.linalg.norm(BR - BL)
    hA = np.linalg.norm(TR - BR)
    hB = np.linalg.norm(TL - BL)
    avg_w = (wA + wB) / 2.0
    avg_h = (hA + hB) / 2.0
    aspect = max(1e-6, avg_w / avg_h)
    out_w = 2000
    out_h = int(round(out_w / aspect))
    H_play = compute_homography(play_corners, out_w, out_h)

    # 5) Expand canvas to include rails using outer corners
    outer_proj = transform_points(H_play, outer_corners)  # shape (4,2) in play canvas coords
    all_pts = np.vstack([outer_proj, np.array([[0,0],[out_w-1,0],[out_w-1,out_h-1],[0,out_h-1]], dtype=np.float64)])
    min_xy = np.floor(all_pts.min(axis=0) - 10)  # small padding
    max_xy = np.ceil(all_pts.max(axis=0) + 10)

    shift_x = -min_xy[0]
    shift_y = -min_xy[1]
    T = np.array([[1,0,shift_x],
                  [0,1,shift_y],
                  [0,0,1]], dtype=np.float64)

    out_full_w = int(max_xy[0] - min_xy[0])
    out_full_h = int(max_xy[1] - min_xy[1])

    H_full = (T @ H_play).astype(np.float32)  # warp whole image so rails are visible

    # 6) Preview & approve
    warped_preview = cv2.warpPerspective(snap, H_full, (out_full_w, out_full_h))
    display = warped_preview.copy()
    # Draw play-surface rectangle on preview
    ps = transform_points(H_full, play_corners)  # already includes T
    ps = ps.astype(int)
    cv2.polylines(display, [ps.reshape(-1,1,2)], True, (0,255,0), 2, cv2.LINE_AA)
    v,_ = resize_with_scale(display)
    cv2.imshow("Calibrate - Preview", v)
    print("Preview: Enter=save, r=redo, q=cancel")
    while True:
        k = cv2.waitKey(0) & 0xFF
        if k in (13,10):
            break
        elif k == ord('r'):
            cv2.destroyAllWindows()
            return main()
        elif k == ord('q'):
            cv2.destroyAllWindows()
            return
    cv2.destroyAllWindows()

    # 7) Save calibration
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = {
        "homography_play": H_play.tolist(),
        "homography_full": H_full.tolist(),  # includes translation so rails fit in frame
        "play_size": {"width": int(out_w), "height": int(out_h)},
        "full_size": {"width": int(out_full_w), "height": int(out_full_h)},
        "play_corners_src": play_corners.tolist(),
        "outer_corners_src": outer_corners.tolist(),
        "pocket_points": pocket_pts.tolist()
    }
    with open(CALIBRATION_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"[calibration] Saved -> {CALIBRATION_PATH}")

if __name__ == "__main__":
    main()
