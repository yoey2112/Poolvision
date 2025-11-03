import json
from pathlib import Path
import cv2
from pv.camera import Camera

def load_config() -> dict:
    cfg_path = Path(__file__).resolve().parent.parent / "config" / "config.json"
    with open(cfg_path, "r", encoding="utf-8") as f:
        return json.load(f)

def draw_hud(frame, text: str, org=(12, 28)):
    cv2.putText(
        frame,
        text,
        org,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        lineType=cv2.LINE_AA
    )

def resize_for_display(frame, max_width: int):
    h, w = frame.shape[:2]
    if w <= max_width:
        return frame
    scale = max_width / float(w)
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

def main():
    cfg = load_config()
    cam_cfg = cfg.get("camera", {})
    ui_cfg = cfg.get("ui", {})

    cam = Camera(
        index=cam_cfg.get("index", "auto"),
        width=int(cam_cfg.get("width", 1920)),
        height=int(cam_cfg.get("height", 1080)),
        fps=int(cam_cfg.get("fps", 30)),
        backend_name=str(cam_cfg.get("backend", "dshow")),
        flip=bool(cam_cfg.get("flip", False)),
    )
    cam.open()

    actual_w, actual_h, actual_fps = cam.actual_props()
    print(f"Requested: {cam_cfg.get('width')}x{cam_cfg.get('height')}@{cam_cfg.get('fps')} ({cam_cfg.get('backend')})")
    print(f"Actual:    {actual_w}x{actual_h}@{actual_fps:.1f}")

    window_name = ui_cfg.get("window_name", "PoolVision - Live")
    show_fps = bool(ui_cfg.get("show_fps", True))
    max_display_width = int(ui_cfg.get("max_display_width", 1600))

    Path("snapshots").mkdir(exist_ok=True)

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    print("Controls: 'q' = quit, 's' = save snapshot to ./snapshots")

    while True:
        ok, frame = cam.read()
        if not ok or frame is None:
            print("Frame read failed.")
            break

        display = frame
        if show_fps:
            draw_hud(display, f"FPS: {cam.fps:.1f}  {display.shape[1]}x{display.shape[0]}")

        display = resize_for_display(display, max_display_width)
        cv2.imshow(window_name, display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            out = Path("snapshots") / f"frame_{cv2.getTickCount()}.jpg"
            cv2.imwrite(str(out), frame)  # save full-res
            print(f"Saved snapshot: {out}")

    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
