import json
from pathlib import Path
import cv2
import numpy as np

from pv.camera import Camera
from pv.calib import load_calibration

def load_config() -> dict:
    cfg_path = Path(__file__).resolve().parent.parent / "config" / "config.json"
    with open(cfg_path, "r", encoding="utf-8") as f:
        return json.load(f)

def draw_hud(img, text: str, org=(12, 28)):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)

def draw_grid_in_rect(img, rect_pts: np.ndarray, step=100):
    h, w = img.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [rect_pts.reshape(-1,1,2).astype(np.int32)], 255)
    for x in range(0, w, step):
        line = np.zeros_like(mask); cv2.line(line, (x, 0), (x, h), 255, 1, cv2.LINE_AA)
        line = cv2.bitwise_and(line, mask); img[line > 0] = (255,255,255)
    for y in range(0, h, step):
        line = np.zeros_like(mask); cv2.line(line, (0, y), (w, y), 255, 1, cv2.LINE_AA)
        line = cv2.bitwise_and(line, mask); img[line > 0] = (255,255,255)

def main():
    cfg = load_config()
    cam_cfg = cfg.get("camera", {})
    ui_cfg = cfg.get("ui", {})
    runtime = cfg.get("runtime", {})
    s = float(runtime.get("downscale", 1.0))  # runtime scale

    # Load calibration
    try:
        raw, H_full, (full_w, full_h), H_play, (play_w, play_h) = load_calibration()
    except Exception as e:
        print(f"[error] {e}")
        print("Tip: Run `python .\\src\\calibrate.py` first.")
        return

    # Apply runtime downscale via homography scale
    S = np.array([[s, 0, 0],[0, s, 0],[0, 0, 1]], dtype=np.float32)
    H_runtime = (S @ H_full).astype(np.float32)
    out_w = max(1, int(round(full_w * s)))
    out_h = max(1, int(round(full_h * s)))

    # Compute play polygon in runtime space
    play_src = np.array(raw["play_corners_src"], dtype=np.float32)
    ones = np.ones((4,1), dtype=np.float32)
    ps = np.hstack([play_src, ones]) @ H_runtime.T
    ps = ps[:,:2] / ps[:,2:3]

    # Camera
    cam = Camera(
        index=cam_cfg.get("index","auto"),
        width=int(cam_cfg.get("width",1920)),
        height=int(cam_cfg.get("height",1080)),
        fps=int(cam_cfg.get("fps",30)),
        backend_name=str(cam_cfg.get("backend","dshow")),
        flip=bool(cam_cfg.get("flip",False)),
        prefer_mjpg=bool(cam_cfg.get("prefer_mjpg",True)),
    )
    cam.open()

    window_name = ui_cfg.get("window_name","PoolVision - Live")
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    show_grid = True

    print("Controls: 'q' quit | 's' save warped snapshot | 'g' toggle grid")
    while True:
        ok, frame = cam.read()
        if not ok: break

        warped = cv2.warpPerspective(frame, H_runtime, (out_w, out_h))
        display = warped.copy()
        if show_grid:
            draw_grid_in_rect(display, ps.copy(), step=max(50, int(min(out_w, out_h) / 20)))

        draw_hud(display, f"Top-down (rails)  {out_w}x{out_h}   FPS: {cam.fps:.1f}")
        cv2.imshow(window_name, display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('g'):
            show_grid = not show_grid
        elif key == ord('s'):
            out = Path("snapshots") / f"rails_{cv2.getTickCount()}.jpg"
            out.parent.mkdir(exist_ok=True)
            cv2.imwrite(str(out), warped)
            print(f"[save] {out}")

    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
