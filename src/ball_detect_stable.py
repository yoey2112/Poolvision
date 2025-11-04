import json
from pathlib import Path
from typing import Tuple, List
import cv2
import numpy as np

from pv.camera import Camera
from pv.calib import load_calibration

# ----- config -----
def load_config() -> dict:
    cfg_path = Path(__file__).resolve().parent.parent / "config" / "config.json"
    with open(cfg_path, "r", encoding="utf-8") as f:
        return json.load(f)

# ----- warp helpers -----
def warp_points(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    ones = np.ones((pts.shape[0], 1), dtype=np.float32)
    ph = np.hstack([pts.astype(np.float32), ones])
    wp = (ph @ H.T)
    return wp[:, :2] / wp[:, 2:3]

# ----- masks -----
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

# ----- preprocessing -----
def preprocess(bgr: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l,a,b = cv2.split(lab)
    l = cv2.createCLAHE(2.0, (8,8)).apply(l)
    lab = cv2.merge((l,a,b))
    norm = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    gray = cv2.cvtColor(cv2.GaussianBlur(norm, (5,5), 0), cv2.COLOR_BGR2GRAY)
    return gray

# ----- scoring + NMS -----
def rim_edge_score(gray: np.ndarray, x: int, y: int, r: int) -> float:
    """Edge strength in an annulus around the circle rim."""
    h, w = gray.shape
    if r < 5 or x-r-3 < 0 or y-r-3 < 0 or x+r+3 >= w or y+r+3 >= h:
        return 0.0
    # gradient magnitude
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)

    inner = np.zeros_like(gray, dtype=np.uint8)
    outer = np.zeros_like(gray, dtype=np.uint8)
    cv2.circle(inner, (x, y), max(1, r-2), 255, -1)
    cv2.circle(outer, (x, y), r+2, 255, -1)
    ring = cv2.bitwise_and(outer, cv2.bitwise_not(inner))
    vals = mag[ring > 0]
    if vals.size == 0:
        return 0.0
    return float(vals.mean())

def nms_keep_best(cands: List[tuple], scores: List[float], dist_thr: float) -> List[int]:
    """Return indices of kept candidates."""
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

# ----- main -----
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

    # params
    base = int(min(OUT_W, OUT_H))
    min_r = max(10, base // 70)
    max_r = max(min_r + 6, base // 40)
    edge_shrink = max(6, base // 180)
    pocket_excl = max(20, base // 35)
    nms_dist = max(10, base // 150)
    min_texture_var = 8.0

    play_mask = make_play_mask(OUT_W, OUT_H, play_poly, edge_shrink)
    pocket_mask = make_pocket_mask(OUT_W, OUT_H, pocket_pairs_src, H_runtime, pocket_excl)

    dp, param1, param2 = 1.2, 120, 20
    show_grid, show_masks = True, False

    win = ui_cfg.get("window_name","PoolVision - Live")
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    print("q quit | s save | g grid | m masks | [/] minR | ;/' maxR | ,/. edge | -/= sens | 9/0 pocketR")

    # simple EMA smoothing for radius
    r_ema = None
    alpha = 0.6

    while True:
        ok, frame = cam.read()
        if not ok: break

        warped = cv2.warpPerspective(frame, H_runtime, (OUT_W, OUT_H))
        gray = preprocess(warped)

        # base allowed area
        allowed = cv2.bitwise_and(play_mask, cv2.bitwise_not(pocket_mask))

        circles = cv2.HoughCircles(
            gray, cv2.HOUGH_GRADIENT, dp,
            minDist=max(10, int(min_r*1.3)),
            param1=param1, param2=param2,
            minRadius=min_r, maxRadius=max_r
        )
        cands = [] if circles is None else np.round(circles[0,:]).astype(int).tolist()

        # per-candidate score (rim edge) and prune by mask/texture
        scored = []
        for (x,y,r) in cands:
            if r < min_r: continue
            circle_mask = np.zeros_like(allowed)
            cv2.circle(circle_mask,(x,y),r,255,-1)
            inside = cv2.countNonZero(cv2.bitwise_and(circle_mask, allowed))
            if inside < np.pi*(r**2)*0.7:  # enough inside play & away from pockets
                continue
            # texture variance (reject flat little dots)
            patch = gray[max(0,y-r):y+r, max(0,x-r):x+r]
            if patch.size==0 or patch.std() < min_texture_var:
                continue
            score = rim_edge_score(gray, x, y, r)
            scored.append(((x,y,r), score))

        if scored:
            cands2, scores = zip(*scored)
            keep_idx = nms_keep_best(list(cands2), list(scores), dist_thr=float(nms_dist))
            final = [cands2[i] for i in keep_idx]
        else:
            final = []

        # optional smoothing of typical radius
        if final:
            r_avg = float(np.mean([r for _,_,r in final]))
            r_ema = r_avg if r_ema is None else (alpha*r_avg + (1-alpha)*(r_ema))
        display = warped.copy()

        # grid only on play
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

        # draw circles (one per ball)
        for (x,y,r) in final:
            if r_ema is not None:
                r = int(0.5*r + 0.5*r_ema)  # gentle radius stabilize
            cv2.circle(display, (x,y), r, (0,255,0), 2, cv2.LINE_AA)

        hud = f"FPS:{cam.fps:4.1f} balls:{len(final)}  r:[{min_r},{max_r}]  p2:{param2}  edge:{edge_shrink}  pocketR:{pocket_excl}  {OUT_W}x{OUT_H}"
        cv2.putText(display, hud, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)

        cv2.imshow(win, display)
        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'): break
        elif k == ord('s'):
            out = Path("snapshots") / f"stable_{cv2.getTickCount()}.jpg"
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
        elif k == ord('.'):
            edge_shrink = min(80, edge_shrink+1)
            play_mask = make_play_mask(OUT_W, OUT_H, play_poly, edge_shrink)
        elif k == ord('9'):
            pocket_excl = max(6, pocket_excl-2)
            pocket_mask = make_pocket_mask(OUT_W, OUT_H, pocket_pairs_src, H_runtime, pocket_excl)
        elif k == ord('0'):
            pocket_excl = min(180, pocket_excl+2)
            pocket_mask = make_pocket_mask(OUT_W, OUT_H, pocket_pairs_src, H_runtime, pocket_excl)

    cam.release(); cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
