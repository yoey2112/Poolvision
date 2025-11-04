import json
from pathlib import Path
import cv2
import numpy as np
from typing import Tuple
from pv.camera import Camera
from pv.calib import load_calibration

def load_config() -> dict:
    cfg_path = Path(__file__).resolve().parent.parent / "config" / "config.json"
    with open(cfg_path, "r", encoding="utf-8") as f:
        return json.load(f)

def make_play_mask(full_shape: Tuple[int, int], play_poly_full: np.ndarray, edge_margin: int = 0) -> np.ndarray:
    h, w = full_shape[1], full_shape[0]
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [play_poly_full.reshape(-1,1,2).astype(np.int32)], 255)
    if edge_margin > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (edge_margin*2+1, edge_margin*2+1))
        mask = cv2.erode(mask, k, iterations=1)
    return mask

def preprocess(img_bgr: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l,a,b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    lab = cv2.merge((l,a,b))
    norm = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    gray = cv2.cvtColor(cv2.GaussianBlur(norm,(5,5),0), cv2.COLOR_BGR2GRAY)
    return gray

def merge_overlaps(circles, thresh=10):
    merged=[]
    for (x,y,r) in circles:
        found=False
        for i,(mx,my,mr) in enumerate(merged):
            if np.hypot(mx-x,my-y)<thresh:
                merged[i]=((mx+x)//2,(my+y)//2,int((mr+r)/2))
                found=True;break
        if not found:
            merged.append((x,y,r))
    return merged

def filter_circles(gray,bgr,circles,mask):
    out=[]
    for (x,y,r) in circles:
        if r<5 or x<r or y<r or x+r>=gray.shape[1] or y+r>=gray.shape[0]: continue
        roi_mask=np.zeros_like(mask); cv2.circle(roi_mask,(x,y),r,255,-1)
        if cv2.countNonZero(cv2.bitwise_and(mask,roi_mask))<np.pi*r*r*0.6: continue
        mean_int=cv2.mean(gray,mask=roi_mask)[0]
        if mean_int<40 or mean_int>240: continue
        out.append((x,y,r))
    return out

def identify_cue(bgr,circles):
    cue=None; brightest=-1
    for (x,y,r) in circles:
        patch=bgr[max(0,y-r):y+r, max(0,x-r):x+r]
        if patch.size==0: continue
        mean=np.mean(cv2.cvtColor(patch,cv2.COLOR_BGR2GRAY))
        if mean>brightest:
            brightest=mean; cue=(x,y,r)
    return cue

def main():
    cfg=load_config()
    cam_cfg=cfg.get("camera",{})
    ui_cfg=cfg.get("ui",{})
    try:
        raw,H_full,(fw,fh),H_play,(pw,ph)=load_calibration()
    except Exception as e:
        print(f"[error] {e}"); return
    play_src=np.array(raw["play_corners_src"],dtype=np.float32)
    ones=np.ones((4,1),dtype=np.float32)
    ps_full=(np.hstack([play_src,ones])@H_full.T); ps_full=ps_full[:,:2]/ps_full[:,2:3]
    cam=Camera(index=cam_cfg.get("index","auto"),
               width=int(cam_cfg.get("width",1920)),
               height=int(cam_cfg.get("height",1080)),
               fps=int(cam_cfg.get("fps",30)),
               backend_name=str(cam_cfg.get("backend","dshow")),
               flip=bool(cam_cfg.get("flip",False)),
               prefer_mjpg=bool(cam_cfg.get("prefer_mjpg",True)))
    cam.open()
    min_r,max_r,edge_margin,param1,param2=8,55,8,120,18
    dp=1.2; show_grid=True
    mask=make_play_mask((fw,fh),ps_full,edge_margin)
    cv2.namedWindow("BallDetect",cv2.WINDOW_NORMAL)
    print("q quit | s save | [/] minR | ;/' maxR | ,/. margin | -/= sens | 1/2 canny")
    while True:
        ok,frm=cam.read()
        if not ok: break
        warp=cv2.warpPerspective(frm,H_full,(fw,fh))
        gray=preprocess(warp)
        circ=cv2.HoughCircles(gray,cv2.HOUGH_GRADIENT,dp, max(8,int(min_r*1.2)),
                              param1=param1,param2=param2,minRadius=min_r,maxRadius=max_r)
        circles=[]
        if circ is not None:
            circles=np.round(circ[0,:]).astype(int).tolist()
            circles=merge_overlaps(circles)
            circles=filter_circles(gray,warp,circles,mask)
        cue=identify_cue(warp,circles)
        disp=warp.copy()
        if show_grid:
            step=100
            gridmask=np.zeros((fh,fw),dtype=np.uint8)
            cv2.fillPoly(gridmask,[ps_full.reshape(-1,1,2).astype(np.int32)],255)
            for x in range(0,fw,step):
                line=np.zeros_like(gridmask);cv2.line(line,(x,0),(x,fh),255,1)
                disp[line>0]=(255,255,255)
            for y in range(0,fh,step):
                line=np.zeros_like(gridmask);cv2.line(line,(0,y),(fw,y),255,1)
                disp[line>0]=(255,255,255)
        for (x,y,r) in circles:
            color=(0,255,0)
            if cue and (x,y,r)==cue: color=(255,255,0)
            cv2.circle(disp,(x,y),r,color,2)
        cv2.putText(disp,f"FPS:{cam.fps:4.1f} balls:{len(circles)} cue:{'yes' if cue else 'no'}",(12,28),
                    cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
        cv2.imshow("BallDetect",disp)
        k=cv2.waitKey(1)&0xFF
        if k==ord('q'):break
        elif k==ord('s'):
            out=Path("snapshots")/f"detect_{cv2.getTickCount()}.jpg";out.parent.mkdir(exist_ok=True)
            cv2.imwrite(str(out),disp);print("[save]",out)
        elif k==ord('['):min_r=max(4,min_r-1)
        elif k==ord(']'):min_r=min(min_r+1,max_r-2)
        elif k==ord(';'):max_r=max(max_r-1,min_r+2)
        elif k==ord("'"):max_r+=1
        elif k==ord('-'):param2=max(5,param2-1)
        elif k==ord('='):param2=min(80,param2+1)
        elif k==ord(','):edge_margin=max(0,edge_margin-1);mask=make_play_mask((fw,fh),ps_full,edge_margin)
        elif k==ord('.'):edge_margin=min(60,edge_margin+1);mask=make_play_mask((fw,fh),ps_full,edge_margin)
        elif k==ord('1'):param1=max(50,param1-5)
        elif k==ord('2'):param1=min(250,param1+5)
    cam.release();cv2.destroyAllWindows()

if __name__=="__main__":
    main()
