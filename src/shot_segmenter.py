# src/shot_segmenter.py
from __future__ import annotations
import json, time
from pathlib import Path
from typing import List, Optional, Dict, Tuple
import cv2, np as numpy_alias  # quick alias to avoid accidental shadowing
import numpy as np

from pv.camera import ThreadedCamera
from pv.calib import load_calibration

# ---------------- config ----------------
def load_config() -> dict:
    cfg_path = Path(__file__).resolve().parent.parent / "config" / "config.json"
    with open(cfg_path, "r", encoding="utf-8") as f:
        return json.load(f)

# ---------------- helpers ----------------
def warp_points(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    ones = np.ones((pts.shape[0], 1), dtype=np.float32)
    ph = np.hstack([pts.astype(np.float32), ones])
    wp = (ph @ H.T)
    return wp[:, :2] / wp[:, 2:3]

def make_play_mask(w: int, h: int, poly: np.ndarray, shrink: int = 0) -> np.ndarray:
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [poly.reshape(-1,1,2).astype(np.int32)], 255)
    if shrink > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (shrink*2+1, shrink*2+1))
        mask = cv2.erode(mask, k, 1)
    return mask

def make_pocket_mask(w: int, h: int, pocket_pairs_src: np.ndarray, H_runtime: np.ndarray, radius_px: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=np.uint8)
    for i in range(6):
        p1, p2 = pocket_pairs_src[i,0], pocket_pairs_src[i,1]
        mid = ((p1+p2)/2.0).astype(np.float32)[None,...]
        mid_w = warp_points(H_runtime, mid)[0]
        cv2.circle(mask, (int(round(mid_w[0])), int(round(mid_w[1]))), int(radius_px), 255, -1)
    return mask

def preprocess_fast(bgr: np.ndarray) -> np.ndarray:
    ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    y = ycrcb[...,0]
    y = cv2.GaussianBlur(y, (3,3), 0)
    return y

# ---------------- detection on downscaled image ----------------
def rim_edge_score(gray: np.ndarray, x: int, y: int, r: int) -> float:
    h,w = gray.shape
    if r < 5 or x-r-3 < 0 or y-r-3 < 0 or x+r+3 >= w or y+r+3 >= h:
        return 0.0
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, 3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, 3)
    mag = cv2.magnitude(gx, gy)
    inner = np.zeros_like(gray, np.uint8)
    outer = np.zeros_like(gray, np.uint8)
    cv2.circle(inner, (x,y), max(1,r-2), 255, -1)
    cv2.circle(outer, (x,y), r+2, 255, -1)
    ring = cv2.bitwise_and(outer, cv2.bitwise_not(inner))
    vals = mag[ring>0]
    return float(vals.mean()) if vals.size else 0.0

def nms_keep_best(cands: List[tuple], scores: List[float], dist_thr: float) -> List[int]:
    if not cands: return []
    order = sorted(range(len(cands)), key=lambda i: scores[i], reverse=True)
    kept = []
    for i in order:
        xi,yi,ri = cands[i]
        ok = True
        for j in kept:
            xj,yj,rj = cands[j]
            if (xi-xj)**2 + (yi-yj)**2 <= dist_thr**2:
                ok = False; break
        if ok: kept.append(i)
    return kept

def detect_balls_scaled(gray_full: np.ndarray, allowed_full: np.ndarray,
                        min_r_full: int, max_r_full: int,
                        dp=1.2, param1=110, param2=18,
                        nms_dist_full=12, min_texture_var=7.5,
                        det_scale: float = 0.6) -> List[tuple]:
    if det_scale != 1.0:
        gray = cv2.resize(gray_full, None, fx=det_scale, fy=det_scale, interpolation=cv2.INTER_AREA)
        allowed = cv2.resize(allowed_full, None, fx=det_scale, fy=det_scale, interpolation=cv2.INTER_NEAREST)
        s = det_scale
        min_r = max(5, int(round(min_r_full * s)))
        max_r = max(min_r+4, int(round(max_r_full * s)))
        nms_dist = max(6, int(round(nms_dist_full * s)))
    else:
        gray, allowed = gray_full, allowed_full
        min_r, max_r, nms_dist = min_r_full, max_r_full, nms_dist_full

    circles = cv2.HoughCircles(
        gray, cv2.HOUGH_GRADIENT, dp,
        minDist=max(10, int(min_r*1.3)),
        param1=param1, param2=param2,
        minRadius=min_r, maxRadius=max_r
    )
    cands = [] if circles is None else np.round(circles[0,:]).astype(int).tolist()

    scored=[]
    for (x,y,r) in cands:
        if r < min_r: continue
        cm = np.zeros_like(allowed); cv2.circle(cm,(x,y),r,255,-1)
        inside = cv2.countNonZero(cv2.bitwise_and(cm, allowed))
        if inside < np.pi*(r**2)*0.7: continue
        patch = gray[max(0,y-r):y+r, max(0,x-r):x+r]
        if patch.size==0 or patch.std()<min_texture_var: continue
        score = rim_edge_score(gray, x, y, r)
        scored.append(((x,y,r), score))

    if not scored: return []

    c2, scores = zip(*scored)
    keep = nms_keep_best(list(c2), list(scores), dist_thr=float(nms_dist))
    dets = [c2[i] for i in keep]

    if det_scale != 1.0:
        inv = 1.0 / det_scale
        dets = [(int(round(x*inv)), int(round(y*inv)), int(round(r*inv))) for (x,y,r) in dets]
    return dets

# ---------------- tracker with jitter clamp ----------------
class Track:
    def __init__(self, tid:int, x:float,y:float,r:float):
        self.id=tid; self.x=x; self.y=y; self.r=r
        self.vx=0.0; self.vy=0.0; self.hits=1; self.missed=0; self.is_cue=False
        self._avgx=x; self._avgy=y

    def predict(self, decay=0.85):
        self.x += self.vx
        self.y += self.vy
        self.vx *= decay
        self.vy *= decay

    def update(self,x:float,y:float,r:float,
               a=0.6, ar=0.5, snap_px=1.2):
        if abs(x-self.x) < snap_px and abs(y-self.y) < snap_px:
            x, y = self.x, self.y
        vx,vy = x-self.x, y-self.y
        self.vx = 0.5*self.vx + 0.5*vx
        self.vy = 0.5*self.vy + 0.5*vy
        self.x  = a*x + (1-a)*self.x
        self.y  = a*y + (1-a)*self.y
        self.r  = ar*r + (1-ar)*self.r
        self.hits += 1; self.missed = 0
        self._avgx = 0.8*self._avgx + 0.2*self.x
        self._avgy = 0.8*self._avgy + 0.2*self.y

    @property
    def speed(self)->float: return float(np.hypot(self.vx,self.vy))

class Tracker:
    def __init__(self,max_missed=10,gate_px=45):
        self.tracks: List[Track]=[]; self.next_id=1
        self.max_missed=max_missed; self.gate_px=gate_px; self.cue_id:Optional[int]=None

    def _dist(self,t:Track,det:tuple)->float:
        x,y,_=det; return float(np.hypot(t.x-x,t.y-y))

    def step(self,dets: Optional[List[tuple]], locked: bool):
        if locked:
            return
        for t in self.tracks: t.predict()

        if dets is None:
            for t in self.tracks: t.missed=min(self.max_missed, t.missed+1)
            self.tracks=[t for t in self.tracks if t.missed<=self.max_missed]; return

        used=set()
        for t in self.tracks:
            best=None; bd=1e9; bi=-1
            for i,d in enumerate(dets):
                if i in used: continue
                d2=self._dist(t,d)
                if d2<bd: bd=d2; best=d; bi=i
            if best is not None and bd<=self.gate_px:
                t.update(*best)
                used.add(bi)
            else:
                t.missed += 1

        for i,d in enumerate(dets):
            if i in used: continue
            x,y,r=d; nt=Track(self.next_id,float(x),float(y),float(r))
            self.next_id+=1; self.tracks.append(nt)

        self.tracks=[t for t in self.tracks if t.missed<=self.max_missed]

    def assign_cue_by_brightness(self,bgr:np.ndarray):
        if not self.tracks: return
        gray=cv2.cvtColor(bgr,cv2.COLOR_BGR2GRAY)
        best_id=None; best_v=-1.0
        for t in self.tracks:
            x,y,r=int(round(t.x)),int(round(t.y)),int(round(t.r))
            p=gray[max(0,y-r):y+r, max(0,x-r):x+r]
            if p.size==0: continue
            v=float(p.mean())
            if v>best_v: best_v=v; best_id=t.id
        self.cue_id=best_id
        for t in self.tracks: t.is_cue=(t.id==self.cue_id)

# ---------------- stability lock ----------------
class StabilityLock:
    """
    Locks tracker when the table is still and unlocks on real motion.
    """
    def __init__(self, play_mask: np.ndarray,
                 still_thresh_ratio=0.0015,
                 diff_threshold=14,
                 still_frames_to_lock=6,
                 unlock_motion_ratio=0.003,
                 unlock_cue_speed=6.0):
        # ✅ ensure mask is uint8 0/255 (fixes your crash)
        if play_mask.dtype != np.uint8:
            play_mask = (play_mask.astype(np.uint8) * 255) if play_mask.dtype == np.bool_ else play_mask.astype(np.uint8)
        self.play_mask = play_mask

        self.diff_threshold = int(diff_threshold)
        self.still_thresh_ratio = float(still_thresh_ratio)
        self.unlock_motion_ratio = float(unlock_motion_ratio)
        self.still_frames_to_lock = int(still_frames_to_lock)
        self.unlock_cue_speed = float(unlock_cue_speed)

        self.prev: Optional[np.ndarray] = None
        self.static_run = 0
        self.locked = False

    def update(self, gray: np.ndarray, cue_speed: float) -> bool:
        if self.prev is None:
            self.prev = gray.copy()
            self.locked = False
            self.static_run = 0
            return self.locked

        diff = cv2.absdiff(gray, self.prev)
        self.prev = gray.copy()
        _, th = cv2.threshold(diff, self.diff_threshold, 255, cv2.THRESH_BINARY)
        moving = cv2.countNonZero(cv2.bitwise_and(th, self.play_mask))
        total  = cv2.countNonZero(self.play_mask)
        ratio = (moving / max(1, total))

        if not self.locked:
            if ratio < self.still_thresh_ratio:
                self.static_run += 1
                if self.static_run >= self.still_frames_to_lock:
                    self.locked = True
            else:
                self.static_run = 0
        else:
            if cue_speed >= self.unlock_cue_speed or ratio >= self.unlock_motion_ratio:
                self.locked = False
                self.static_run = 0

        return self.locked

# ---------------- shot recorder (unchanged) ----------------
class ShotRecorder:
    def __init__(self, pocket_mask: np.ndarray, fps: float,
                 speed_start_px=8.0, speed_end_px=1.2, settle_frames=18,
                 start_confirm_frames=2, max_duration_s=20.0,
                 min_track_frames=5, min_track_displacement=6.0):
        self.fps=float(max(fps,1.0)); self.active=False
        self.start_frame_idx=0; self.end_frame_idx=0
        self.traj:Dict[int,List[Tuple[float,float]]]={}; self._first_pos:Dict[int,Tuple[float,float]]={}
        self.pocketed:Dict[int,bool]={}
        self.speed_start_px=float(speed_start_px); self.speed_end_px=float(speed_end_px)
        self.settle_frames=int(settle_frames); self.start_confirm_frames=int(start_confirm_frames)
        self.max_duration_s=float(max_duration_s)
        self._start_arm=0; self._still_counter=0; self._first_contact_logged=False
        self.first_contact:Optional[Tuple[int,int]]=None
        self.pocket_mask=pocket_mask
        self.min_track_frames=int(min_track_frames); self.min_track_displacement=float(min_track_displacement)
    def update_fps(self,v:float):
        if v>0.1: self.fps=float(v)
    def _in_pocket(self,x:float,y:float)->bool:
        xi,yi=int(round(x)),int(round(y))
        if yi<0 or yi>=self.pocket_mask.shape[0] or xi<0 or xi>=self.pocket_mask.shape[1]: return False
        return self.pocket_mask[yi,xi]>0
    def maybe_start(self,tracks:List[Track], frame_idx:int):
        cue=next((t for t in tracks if t.is_cue),None)
        if cue is None: self._start_arm=0; return
        self._start_arm = self._start_arm+1 if cue.speed>=self.speed_start_px else 0
        if not self.active and self._start_arm>=self.start_confirm_frames:
            self.active=True; self.start_frame_idx=frame_idx; self.end_frame_idx=frame_idx
            self.traj.clear(); self._first_pos.clear(); self.pocketed.clear()
            self._still_counter=0; self._first_contact_logged=False; self.first_contact=None
    def update(self,tracks:List[Track], frame_idx:int):
        if not self.active: return
        any_moving=False
        for t in tracks:
            self.traj.setdefault(t.id,[]).append((float(t.x),float(t.y)))
            self._first_pos.setdefault(t.id,(float(t.x),float(t.y)))
            if self._in_pocket(t.x,t.y): self.pocketed[t.id]=True
            if t.speed>self.speed_end_px: any_moving=True
        self._still_counter = 0 if any_moving else self._still_counter+1
        self.end_frame_idx=frame_idx
        duration_frames=self.end_frame_idx-self.start_frame_idx
        if self._still_counter>=self.settle_frames or (duration_frames/max(self.fps,0.1))>=self.max_duration_s:
            self.active=False
    def dump_summary(self)->dict:
        filtered:Dict[str,List[Tuple[float,float]]]={}
        for tid,pts in self.traj.items():
            if len(pts)<self.min_track_frames: continue
            (x0,y0)=self._first_pos.get(tid,pts[0]); (x1,y1)=pts[-1]
            if float(np.hypot(x1-x0,y1-y0))<self.min_track_displacement: continue
            filtered[str(tid)]=pts
        duration_frames=max(0,self.end_frame_idx-self.start_frame_idx)
        return {
            "fps":float(self.fps),
            "start_frame":int(self.start_frame_idx),
            "end_frame":int(self.end_frame_idx),
            "duration_frames":int(duration_frames),
            "duration_s":float(duration_frames/max(self.fps,0.1)),
            "trajectories":filtered,
            "pocketed":{str(k):bool(v) for k,v in self.pocketed.items() if str(k) in filtered},
            "first_contact":None
        }

# ---------------- main ----------------
def main():
    cfg = load_config()
    cam_cfg = cfg.get("camera", {})
    ui_cfg  = cfg.get("ui", {})
    runtime = cfg.get("runtime", {})
    s = float(runtime.get("downscale", 0.6))

    raw, H_full, (FULL_W, FULL_H), H_play, (PLAY_W, PLAY_H) = load_calibration()
    S = np.array([[s,0,0],[0,s,0],[0,0,1]], np.float32)
    H_runtime = (S @ H_full).astype(np.float32)
    OUT_W, OUT_H = int(round(FULL_W*s)), int(round(FULL_H*s))

    play_src = np.array(raw["play_corners_src"], np.float32)
    play_poly = warp_points(H_runtime, play_src)
    pp = np.array(raw["pocket_points"], np.float32).reshape(12,2)
    pocket_pairs_src = np.stack([pp[0:2], pp[2:4], pp[4:6], pp[6:8], pp[8:10], pp[10:12]], axis=0)

    base = int(min(OUT_W, OUT_H))
    edge_shrink = max(6, base//180)
    pocket_excl = max(20, base//35)
    play_mask   = make_play_mask(OUT_W, OUT_H, play_poly, edge_shrink)
    pocket_mask = make_pocket_mask(OUT_W, OUT_H, pocket_pairs_src, H_runtime, pocket_excl)
    allowed     = cv2.bitwise_and(play_mask, cv2.bitwise_not(pocket_mask))

    grid = np.zeros((OUT_H, OUT_W, 3), np.uint8)
    step = max(50, int(min(OUT_W, OUT_H)/20))
    for x in range(0, OUT_W, step): cv2.line(grid,(x,0),(x,OUT_H),(255,255,255),1,cv2.LINE_AA)
    for y in range(0, OUT_H, step): cv2.line(grid,(0,y),(OUT_W,y),(255,255,255),1,cv2.LINE_AA)
    grid_mask = np.zeros((OUT_H, OUT_W), np.uint8)
    cv2.fillPoly(grid_mask, [play_poly.reshape(-1,1,2).astype(np.int32)], 255)
    grid = cv2.bitwise_and(grid, grid, mask=grid_mask)

    cam = ThreadedCamera(index=cam_cfg.get("index","auto"),
                         width=int(cam_cfg.get("width",1280)),
                         height=int(cam_cfg.get("height",720)),
                         fps=int(cam_cfg.get("fps",20)),
                         backend_name=str(cam_cfg.get("backend","dshow")),
                         flip=bool(cam_cfg.get("flip",False)),
                         prefer_mjpg=bool(cam_cfg.get("prefer_mjpg",True)),
                         set_buffer_sz=1)
    cam.open()

    min_r_full = max(10, base//70)
    max_r_full = max(min_r_full+6, base//40)
    nms_dist_full = max(10, base//150)
    det_scale = 0.6
    detect_every = 3

    tracker = Tracker(max_missed=10, gate_px=max(28, base//30))
    shot = ShotRecorder(pocket_mask=pocket_mask, fps=max(cam.fps,1.0))
    # pass mask as-is; class converts dtype properly
    guard = StabilityLock(play_mask=(allowed>0),
                          still_thresh_ratio=0.0015,
                          diff_threshold=14,
                          still_frames_to_lock=6,
                          unlock_motion_ratio=0.003,
                          unlock_cue_speed=6.0)

    win = ui_cfg.get("window_name","PoolVision - Live")
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    out_dir = Path("shots"); out_dir.mkdir(exist_ok=True)

    frame_idx=0; fps_ema=0.0; alpha=0.15; t_prev=time.perf_counter()
    print("Controls: q quit | s save | space force-start")

    while True:
        ok, frame = cam.read()
        if not ok or frame is None:
            time.sleep(0.001); continue

        now=time.perf_counter(); dt=now-t_prev; t_prev=now
        if dt>0:
            inst=1.0/dt
            fps_ema = inst if fps_ema==0 else alpha*inst + (1-alpha)*fps_ema

        warped = cv2.warpPerspective(frame, H_runtime, (OUT_W, OUT_H))
        gray   = preprocess_fast(warped)

        cue_speed = 0.0
        if tracker.tracks:
            cue = next((t for t in tracker.tracks if t.is_cue), None)
            if cue is not None:
                cue_speed = cue.speed

        locked = guard.update(gray, cue_speed)

        detections=None
        if (frame_idx % detect_every == 0) and not locked:
            detections = detect_balls_scaled(
                gray, allowed, min_r_full, max_r_full,
                dp=1.2, param1=110, param2=18,
                nms_dist_full=nms_dist_full, min_texture_var=7.5,
                det_scale=det_scale
            )

        tracker.step(detections, locked=locked)
        tracker.assign_cue_by_brightness(warped)

        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):
            shot.active=True; shot.start_frame_idx=frame_idx; shot.end_frame_idx=frame_idx
            shot.traj.clear(); shot._first_pos.clear(); shot.pocketed.clear()
            shot._still_counter=0; shot._first_contact_logged=False; shot.first_contact=None
        else:
            shot.maybe_start(tracker.tracks, frame_idx)

        shot.update_fps(fps_ema if fps_ema>0.1 else cam.fps)
        shot.update(tracker.tracks, frame_idx)

        display = warped.copy()
        display = cv2.addWeighted(display, 1.0, grid, 0.8, 0)
        for t in tracker.tracks:
            color = (255,255,0) if t.is_cue else (0,255,0)
            cv2.circle(display,(int(round(t.x)),int(round(t.y))),int(round(t.r)),color,2,cv2.LINE_AA)
            cv2.putText(display,f"ID {t.id}",(int(t.x)+int(t.r)+4,int(t.y)),
                        cv2.FONT_HERSHEY_SIMPLEX,0.6,color,2,cv2.LINE_AA)

        just_finished = (not shot.active) and (shot.start_frame_idx != shot.end_frame_idx)
        if just_finished:
            summary = shot.dump_summary()
            ts = int(time.time()*1000)
            (out_dir / f"shot_{ts}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
            cv2.imwrite(str(out_dir / f"shot_{ts}.png"), display)
            print(f"[shot] END frames:{summary['duration_frames']} secs:{summary['duration_s']:.2f} fps:{summary['fps']:.1f}")

        hud = f"FPS(loop):{(fps_ema or 0):4.1f}  FPS(cam):{cam.fps:4.1f}  tracks:{len(tracker.tracks)}  lock:{locked}  active:{shot.active}"
        cv2.putText(display, hud, (10,24), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)

        if key == ord('q'): break
        if key == ord('s'):
            out = Path('snapshots')/f"seg_{cv2.getTickCount()}.jpg"
            out.parent.mkdir(exist_ok=True); cv2.imwrite(str(out), display); print("[save]", out)

        cv2.imshow(win, display)
        frame_idx += 1

    cam.release(); cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
