import json
from pathlib import Path
from typing import Tuple, List, Optional
import cv2
import numpy as np

from pv.camera import Camera
from pv.calib import load_calibration

# ---------------- config ----------------
def load_config() -> dict:
    cfg_path = Path(__file__).resolve().parent.parent / "config" / "config.json"
    with open(cfg_path, "r", encoding="utf-8") as f:
        return json.load(f)

# ---------------- warp helpers ----------------
def warp_points(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    ones = np.ones((pts.shape[0], 1), dtype=np.float32)
    ph = np.hstack([pts.astype(np.float32), ones])
    wp = (ph @ H.T)
    return wp[:, :2] / wp[:, 2:3]

# ---------------- masks ----------------
def make_play_mask(w: int, h: int, poly: np.ndarray, shrink: int = 0) -> np.ndarray:
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [poly.reshape(-1, 1, 2).astype(np.int32)], 255)
    if shrink > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (shrink*2+1, shrink*2+1))
        mask = cv2.erode(mask, k, 1)
    return mask

def make_pocket_mask(w: int, h: int, pocket_pairs_src: np.ndarray, H_runtime: np.ndarray, radius_px: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=np.uint8)
    for i in range(6):
        p1, p2 = pocket_pairs_src[i, 0], pocket_pairs_src[i, 1]
        mid = ((p1 + p2) / 2.0).astype(np.float32)[None, ...]
        mid_w = warp_points(H_runtime, mid)[0]
        cv2.circle(mask, (int(round(mid_w[0])), int(round(mid_w[1]))), int(radius_px), 255, -1)
    return mask

# ---------------- preprocessing ----------------
def preprocess(bgr: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(2.0, (8, 8)).apply(l)
    lab = cv2.merge((l, a, b))
    norm = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    gray = cv2.cvtColor(cv2.GaussianBlur(norm, (5, 5), 0), cv2.COLOR_BGR2GRAY)
    return gray

# ---------------- detection (scored + NMS) ----------------
def rim_edge_score(gray: np.ndarray, x: int, y: int, r: int) -> float:
    h, w = gray.shape
    if r < 5 or x-r-3 < 0 or y-r-3 < 0 or x+r+3 >= w or y+r+3 >= h:
        return 0.0
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, 3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, 3)
    mag = cv2.magnitude(gx, gy)
    inner = np.zeros_like(gray, dtype=np.uint8)
    outer = np.zeros_like(gray, dtype=np.uint8)
    cv2.circle(inner, (x, y), max(1, r-2), 255, -1)
    cv2.circle(outer, (x, y), r+2, 255, -1)
    ring = cv2.bitwise_and(outer, cv2.bitwise_not(inner))
    vals = mag[ring > 0]
    return float(vals.mean()) if vals.size else 0.0

def nms_keep_best(cands: List[tuple], scores: List[float], dist_thr: float) -> List[int]:
    if not cands:
        return []
    order = sorted(range(len(cands)), key=lambda i: scores[i], reverse=True)
    kept = []
    for i in order:
        xi, yi, ri = cands[i]
        keep = True
        for j in kept:
            xj, yj, rj = cands[j]
            if (xi-xj)**2 + (yi-yj)**2 <= dist_thr**2:
                keep = False
                break
        if keep:
            kept.append(i)
    return kept

def detect_balls(warped_bgr, gray, allowed_mask, min_r, max_r, dp, param1, param2,
                 nms_dist, min_texture_var) -> List[tuple]:
    circles = cv2.HoughCircles(
        gray, cv2.HOUGH_GRADIENT, dp,
        minDist=max(10, int(min_r * 1.3)),
        param1=param1, param2=param2,
        minRadius=min_r, maxRadius=max_r
    )
    cands = [] if circles is None else np.round(circles[0, :]).astype(int).tolist()

    scored = []
    for (x, y, r) in cands:
        if r < min_r: 
            continue
        circle_mask = np.zeros_like(allowed_mask)
        cv2.circle(circle_mask, (x, y), r, 255, -1)
        inside = cv2.countNonZero(cv2.bitwise_and(circle_mask, allowed_mask))
        if inside < np.pi * (r ** 2) * 0.7:
            continue
        patch = gray[max(0, y-r):y+r, max(0, x-r):x+r]
        if patch.size == 0 or patch.std() < min_texture_var:
            continue
        score = rim_edge_score(gray, x, y, r)
        scored.append(((x, y, r), score))

    if not scored:
        return []
    cands2, scores = zip(*scored)
    keep_idx = nms_keep_best(list(cands2), list(scores), dist_thr=float(nms_dist))
    return [cands2[i] for i in keep_idx]

# ---------------- tracker ----------------
class Track:
    def __init__(self, tid: int, x: float, y: float, r: float):
        self.id = tid
        self.x = x; self.y = y; self.r = r
        self.vx = 0.0; self.vy = 0.0
        self.hits = 1
        self.missed = 0
        self.is_cue = False

    def predict(self):
        self.x += self.vx
        self.y += self.vy

    def update(self, x: float, y: float, r: float, alpha_pos=0.6, alpha_r=0.5):
        # velocity from residual
        vx_new = x - self.x
        vy_new = y - self.y
        self.vx = 0.5 * self.vx + 0.5 * vx_new
        # EMA smoothing
        self.x = alpha_pos * x + (1 - alpha_pos) * self.x
        self.y = alpha_pos * y + (1 - alpha_pos) * self.y
        self.r = alpha_r * r + (1 - alpha_r) * self.r
        self.hits += 1
        self.missed = 0

class Tracker:
    def __init__(self, max_missed=8, gate_px=50):
        self.tracks: List[Track] = []
        self.next_id = 1
        self.max_missed = max_missed
        self.gate_px = gate_px
        self.cue_id: Optional[int] = None

    def _distance(self, t: Track, det: tuple) -> float:
        x, y, r = det
        return float(np.hypot(t.x - x, t.y - y))

    def step(self, detections: List[tuple]):
        # predict
        for t in self.tracks:
            t.predict()

        used_det = set()
        # greedy nearest-neighbor with gate
        for t in self.tracks:
            best = None; best_d = 1e9; best_i = -1
            for i, det in enumerate(detections):
                if i in used_det: continue
                d = self._distance(t, det)
                if d < best_d:
                    best_d = d; best = det; best_i = i
            if best is not None and best_d <= self.gate_px:
                t.update(*best)
                used_det.add(best_i)
            else:
                t.missed += 1

        # new tracks for unmatched detections
        for i, det in enumerate(detections):
            if i in used_det: 
                continue
            x, y, r = det
            nt = Track(self.next_id, float(x), float(y), float(r))
            self.next_id += 1
            self.tracks.append(nt)

        # drop long-missed tracks
        self.tracks = [t for t in self.tracks if t.missed <= self.max_missed]

        # maintain cue id (brightest heuristic handled outside; we only keep the flag)
        if self.cue_id is not None and not any(t.id == self.cue_id for t in self.tracks):
            self.cue_id = None

    def assign_cue_by_brightness(self, bgr: np.ndarray):
        # find brightest track region
        best_id, best_val = None, -1.0
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        for t in self.tracks:
            x, y, r = int(round(t.x)), int(round(t.y)), int(round(t.r))
            patch = gray[max(0,y-r):y+r, max(0,x-r):x+r]
            if patch.size == 0: 
                continue
            val = float(patch.mean())
            if val > best_val:
                best_val = val; best_id = t.id
        # stickiness: keep previous cue unless new one is much brighter
        if self.cue_id is None:
            self.cue_id = best_id
        else:
            # simple stickiness rule
            cur = next((t for t in self.tracks if t.id == self.cue_id), None)
            if cur is None:
                self.cue_id = best_id
            else:
                x, y, r = int(round(cur.x)), int(round(cur.y)), int(round(cur.r))
                cur_val = float(gray[max(0,y-r):y+r, max(0,x-r):x+r].mean()) if r>0 else -1
                if best_val - cur_val > 10:   # require meaningful jump to swap
                    self.cue_id = best_id
        # set flags
        for t in self.tracks:
            t.is_cue = (t.id == self.cue_id)

# ---------------- main app ----------------
def main():
    cfg = load_config()
    cam_cfg = cfg.get("camera", {})
    ui_cfg = cfg.get("ui", {})
    runtime = cfg.get("runtime", {})
    s = float(runtime.get("downscale", 1.0))

    # calibration
    raw, H_full, (FULL_W, FULL_H), H_play, (PLAY_W, PLAY_H) = load_calibration()
    S = np.array([[s,0,0],[0,s,0],[0,0,1]], dtype=np.float32)
    H_runtime = (S @ H_full).astype(np.float32)
    OUT_W = max(1, int(round(FULL_W * s)))
    OUT_H = max(1, int(round(FULL_H * s)))

    play_src = np.array(raw["play_corners_src"], dtype=np.float32)
    play_poly = warp_points(H_runtime, play_src)

    pp = np.array(raw["pocket_points"], dtype=np.float32).reshape(12,2)
    pocket_pairs_src = np.stack([pp[0:2], pp[2:4], pp[4:6], pp[6:8], pp[8:10], pp[10:12]], axis=0)

    cam = Camera(index=cam_cfg.get("index","auto"),
                 width=int(cam_cfg.get("width",1920)),
                 height=int(cam_cfg.get("height",1080)),
                 fps=20,
                 backend_name=str(cam_cfg.get("backend","dshow")),
                 flip=bool(cam_cfg.get("flip",False)),
                 prefer_mjpg=bool(cam_cfg.get("prefer_mjpg",True)))
    cam.open()

    # detection params
    base = int(min(OUT_W, OUT_H))
    min_r = max(10, base // 70)
    max_r = max(min_r + 6, base // 40)
    edge_shrink = max(6, base // 180)
    pocket_excl = max(20, base // 35)
    nms_dist = max(10, base // 150)
    min_texture_var = 8.0
    dp, param1, param2 = 1.2, 120, 20

    play_mask = make_play_mask(OUT_W, OUT_H, play_poly, edge_shrink)
    pocket_mask = make_pocket_mask(OUT_W, OUT_H, pocket_pairs_src, H_runtime, pocket_excl)
    allowed = cv2.bitwise_and(play_mask, cv2.bitwise_not(pocket_mask))

    # tracker params (gate in pixels on runtime canvas)
    tracker = Tracker(max_missed=10, gate_px=max(24, base // 30))

    show_grid, show_masks = True, False
    win = ui_cfg.get("window_name","PoolVision - Live")
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    print("Controls: q quit | s save | g grid | m masks | [/] minR | ;/' maxR | -/= sens | ,/. edge | 9/0 pocketR")

    while True:
        ok, frame = cam.read()
        if not ok: break

        warped = cv2.warpPerspective(frame, H_runtime, (OUT_W, OUT_H))
        gray = preprocess(warped)

        # detect
        detections = detect_balls(
            warped, gray, allowed_mask=allowed,
            min_r=min_r, max_r=max_r, dp=dp, param1=param1, param2=param2,
            nms_dist=nms_dist, min_texture_var=min_texture_var
        )

        # track
        tracker.step(detections)
        tracker.assign_cue_by_brightness(warped)

        # draw
        display = warped.copy()

        if show_grid:
            step = max(50, int(min(OUT_W, OUT_H) / 20))
            poly_mask = np.zeros((OUT_H, OUT_W), dtype=np.uint8)
            cv2.fillPoly(poly_mask, [play_poly.reshape(-1,1,2).astype(np.int32)], 255)
            for x in range(0, OUT_W, step):
                line = np.zeros_like(poly_mask); cv2.line(line,(x,0),(x,OUT_H),255,1,cv2.LINE_AA)
                display[line & poly_mask > 0] = (255,255,255)
            for y in range(0, OUT_H, step):
                line = np.zeros_like(poly_mask); cv2.line(line,(0,y),(OUT_W,y),255,1,cv2.LINE_AA)
                display[line & poly_mask > 0] = (255,255,255)

        if show_masks:
            dbg = np.zeros_like(display)
            dbg[...,1] = play_mask
            dbg[...,2] = pocket_mask
            display = cv2.addWeighted(display, 0.7, dbg, 0.3, 0)

        for t in tracker.tracks:
            color = (255,255,0) if t.is_cue else (0,255,0)
            cv2.circle(display, (int(round(t.x)), int(round(t.y))), int(round(t.r)), color, 2, cv2.LINE_AA)
            cv2.putText(display, f"ID {t.id}", (int(t.x)+int(t.r)+4, int(t.y)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
            if t.missed > 0:
                cv2.putText(display, f"({t.missed})", (int(t.x)+int(t.r)+4, int(t.y)+18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1, cv2.LINE_AA)

        hud = f"FPS:{cam.fps:4.1f} tracks:{len(tracker.tracks)}  r:[{min_r},{max_r}]  p2:{param2}  gate:{tracker.gate_px}  {OUT_W}x{OUT_H}"
        cv2.putText(display, hud, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)

        cv2.imshow(win, display)
        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'): break
        elif k == ord('s'):
            out = Path("snapshots") / f"track_{cv2.getTickCount()}.jpg"
            out.parent.mkdir(exist_ok=True); cv2.imwrite(str(out), display); print("[save]", out)
        elif k == ord('g'): show_grid = not show_grid
        elif k == ord('m'): show_masks = not show_masks
        elif k == ord('['): min_r = max(6, min_r-1)
        elif k == ord(']'): min_r = min(min_r+1, max_r-2)
        elif k == ord(';'): max_r = max(max_r-1, min_r+2)
        elif k == ord("'"): max_r += 1
        elif k == ord('-'): param2 = max(8, param2-1)
        elif k == ord('='): param2 = min(80, param2+1)
        elif k == ord(','):
            edge_shrink = max(0, edge_shrink-1)
            play_mask = make_play_mask(OUT_W, OUT_H, play_poly, edge_shrink)
            allowed = cv2.bitwise_and(play_mask, cv2.bitwise_not(pocket_mask))
        elif k == ord('.'):
            edge_shrink = min(80, edge_shrink+1)
            play_mask = make_play_mask(OUT_W, OUT_H, play_poly, edge_shrink)
            allowed = cv2.bitwise_and(play_mask, cv2.bitwise_not(pocket_mask))
        elif k == ord('9'):
            pocket_excl = max(6, pocket_excl-2)
            pocket_mask = make_pocket_mask(OUT_W, OUT_H, pocket_pairs_src, H_runtime, pocket_excl)
            allowed = cv2.bitwise_and(play_mask, cv2.bitwise_not(pocket_mask))
        elif k == ord('0'):
            pocket_excl = min(180, pocket_excl+2)
            pocket_mask = make_pocket_mask(OUT_W, OUT_H, pocket_pairs_src, H_runtime, pocket_excl)
            allowed = cv2.bitwise_and(play_mask, cv2.bitwise_not(pocket_mask))

    cam.release(); cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
