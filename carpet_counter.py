"""
HALI SAYICI v8 - OBB + ByteTrack
===================================
- YOLO OBB ile döndürülmüş kutu tespiti
- ByteTrack ile stabil ID takibi
- Basit ve temiz

Kullanım:
  python carpet_counter.py --video video.mp4 --model best.pt

Kontroller:
  Fare     : Çizgi çiz (2 nokta)
  SPACE    : Onayla
  1 / 2    : Giriş yönü
  R        : Sıfırla
  P        : Duraklat
  Q / ESC  : Çıkış
  S        : Ekran görüntüsü
  +/-      : Eşik ayarla
"""

import cv2
import numpy as np
import argparse
import sys
import os
import time
import zipfile
from dataclasses import dataclass, field


class UIScale:
    def __init__(self, fw, fh):
        self.s = max(fw / 1280, 1.0)
    def font(self, b=0.6): return b * self.s
    def thick(self, b=1): return max(1, int(b * self.s))
    def px(self, b=10): return int(b * self.s)


@dataclass
class TrackInfo:
    positions: list = field(default_factory=list)
    first_seen: int = 0
    last_seen: int = 0
    frames_seen: int = 0
    confidence_sum: float = 0.0
    class_name: str = ""
    color: tuple = (0, 255, 0)
    last_side: int = 0
    can_cross: bool = False
    cross_count: int = 0
    last_cross_frame: int = 0
    last_cross_direction: int = 0
    obb_points: object = None
    bbox: tuple = (0, 0, 0, 0)

    @property
    def center(self):
        return self.positions[-1] if self.positions else None
    @property
    def avg_confidence(self):
        return self.confidence_sum / self.frames_seen if self.frames_seen > 0 else 0
    @property
    def total_travel(self):
        if len(self.positions) < 2: return 0
        t = 0
        for i in range(1, len(self.positions)):
            dx = self.positions[i][0] - self.positions[i-1][0]
            dy = self.positions[i][1] - self.positions[i-1][1]
            t += np.sqrt(dx*dx + dy*dy)
        return t


def get_side_of_line(point, ls, le):
    if point is None: return 0
    lv = (le[0]-ls[0], le[1]-ls[1]); n = (-lv[1], lv[0])
    tp = (point[0]-ls[0], point[1]-ls[1])
    dot = n[0]*tp[0] + n[1]*tp[1]
    return 1 if dot > 0 else (-1 if dot < 0 else 0)

def point_to_line_distance(point, ls, le):
    if point is None: return 9999
    num = abs((le[1]-ls[1])*point[0]-(le[0]-ls[0])*point[1]+le[0]*ls[1]-le[1]*ls[0])
    den = np.sqrt((le[1]-ls[1])**2+(le[0]-ls[0])**2)
    return num/den if den>0 else 0


def draw_arrow(frame, ls, le, d, label, color, sc):
    mid = ((ls[0]+le[0])//2, (ls[1]+le[1])//2)
    lv = np.array([le[0]-ls[0], le[1]-ls[1]], dtype=float)
    ll = np.linalg.norm(lv)
    if ll == 0: return
    n = np.array([-lv[1], lv[0]]) / ll * sc.px(50) * d
    end = (int(mid[0]+n[0]), int(mid[1]+n[1]))
    cv2.arrowedLine(frame, mid, end, color, sc.thick(3), tipLength=0.35)
    cv2.putText(frame, label, (end[0]+sc.px(8), end[1]-sc.px(8)),
                cv2.FONT_HERSHEY_SIMPLEX, sc.font(0.7), color, sc.thick(2))

def draw_trail(frame, pos, color, sc, ml=25):
    pts = pos[-ml:]
    for i in range(1, len(pts)):
        a = i/len(pts); t = max(1, int(a*sc.thick(3)))
        c = tuple(int(v*a) for v in color)
        cv2.line(frame, (int(pts[i-1][0]),int(pts[i-1][1])),
                 (int(pts[i][0]),int(pts[i][1])), c, t)

def draw_obb(frame, obb_pts, color, thickness=2):
    pts = np.int32(obb_pts).reshape((-1, 1, 2))
    cv2.polylines(frame, [pts], True, color, thickness)

def draw_counts(frame, ci, co, net, sc):
    x, y = sc.px(20), sc.px(80); pad = sc.px(15)
    text = f"NET GIRIS: {net}"
    fs = sc.font(1.5); th = sc.thick(3)
    (tw, tht), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fs, th)
    cv2.rectangle(frame, (x-pad,y-tht-pad), (x+tw+pad*2,y+pad), (0,0,0), -1)
    cv2.rectangle(frame, (x-pad,y-tht-pad), (x+tw+pad*2,y+pad), (0,255,100), sc.thick(2))
    cv2.putText(frame, text, (x,y), cv2.FONT_HERSHEY_SIMPLEX, fs, (0,255,100), th)
    y2 = y+sc.px(55); fs2 = sc.font(0.8); th2 = sc.thick(2); gap = sc.px(35)
    t_in = f"Giris: {ci}"; t_out = f"Cikis: {co}"
    (tw1,_),_ = cv2.getTextSize(t_in, cv2.FONT_HERSHEY_SIMPLEX, fs2, th2)
    (tw2,_),_ = cv2.getTextSize(t_out, cv2.FONT_HERSHEY_SIMPLEX, fs2, th2)
    mw = max(tw1, tw2)
    cv2.rectangle(frame, (x-pad,y2-sc.px(25)), (x+mw+pad*2,y2+gap+sc.px(10)), (0,0,0), -1)
    cv2.putText(frame, t_in, (x,y2), cv2.FONT_HERSHEY_SIMPLEX, fs2, (100,255,100), th2)
    cv2.putText(frame, t_out, (x,y2+gap), cv2.FONT_HERSHEY_SIMPLEX, fs2, (100,100,255), th2)

def draw_info(frame, lines, sc, sy=None):
    if sy is None: sy = sc.px(250)
    x = sc.px(20); pad = sc.px(8); fs = sc.font(0.6); th = sc.thick(1); gap = sc.px(35); y = sy
    for txt in lines:
        (tw, tht), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, fs, th)
        cv2.rectangle(frame, (x-pad,y-tht-pad), (x+tw+pad*2,y+pad), (0,0,0), -1)
        cv2.putText(frame, txt, (x,y), cv2.FONT_HERSHEY_SIMPLEX, fs, (200,200,200), th)
        y += gap


def create_tracker_config(conf_dir):
    p = os.path.join(conf_dir, "bytetrack_v8.yaml")
    with open(p, "w") as f:
        f.write(
            "tracker_type: bytetrack\n"
            "track_high_thresh: 0.25\n"
            "track_low_thresh: 0.05\n"
            "new_track_thresh: 0.2\n"
            "track_buffer: 120\n"
            "match_thresh: 0.9\n"
            "fuse_score: True\n"
        )
    return p


def resolve_model_path(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model bulunamadi: {model_path}")

    if os.path.isfile(model_path):
        return model_path

    if not os.path.isdir(model_path):
        raise ValueError(f"Gecersiz model yolu: {model_path}")

    # Klasore acilmis PyTorch checkpoint yapisini tekrar .pt dosyasina paketle.
    required_files = ["data.pkl", "version", "byteorder"]
    is_extracted_torch = all(
        os.path.isfile(os.path.join(model_path, name)) for name in required_files
    ) and os.path.isdir(os.path.join(model_path, "data"))

    if is_extracted_torch:
        repacked_path = model_path.rstrip("/\\") + "_repacked.pt"
        needs_repack = True
        if os.path.exists(repacked_path) and os.path.getsize(repacked_path) >= 1024 and zipfile.is_zipfile(repacked_path):
            try:
                with zipfile.ZipFile(repacked_path, "r") as zf:
                    names = [n for n in zf.namelist() if n and not n.endswith("/")]
                    needs_repack = (not names) or any("/" not in n for n in names)
            except Exception:
                needs_repack = True

        if needs_repack:
            print(f"Model klasor olarak bulundu, tekrar paketleniyor: {repacked_path}")
            root_name = os.path.basename(model_path.rstrip("/\\"))
            with zipfile.ZipFile(
                repacked_path,
                "w",
                compression=zipfile.ZIP_STORED,
                strict_timestamps=False,
            ) as zf:
                for root, _, files in os.walk(model_path):
                    for fname in sorted(files):
                        fpath = os.path.join(root, fname)
                        relname = os.path.relpath(fpath, model_path).replace("\\", "/")
                        arcname = f"{root_name}/{relname}"
                        zf.write(fpath, arcname)
        return repacked_path

    pt_files = [f for f in os.listdir(model_path) if f.endswith(".pt")]
    if len(pt_files) == 1:
        return os.path.join(model_path, pt_files[0])

    raise IsADirectoryError(
        f"Model dosyasi yerine klasor verildi: {model_path}. "
        f"Bir .pt dosya yolu verin."
    )


def main():
    ap = argparse.ArgumentParser(description="YOLO Halı Sayıcı v8 (OBB + ByteTrack)")
    ap.add_argument("--video", required=True)
    ap.add_argument("--model", default="best.pt")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.10)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--output", default=None)
    ap.add_argument("--min-size", type=int, default=200)
    ap.add_argument("--min-frames", type=int, default=5)
    ap.add_argument("--min-travel", type=int, default=30)
    ap.add_argument("--recount-dist", type=int, default=60)
    ap.add_argument("--cooldown-frames", type=int, default=15)
    ap.add_argument("--scale", type=float, default=0.5,
                    help="Çizgi çizme ekranı ölçeği (varsayılan: 0.5)")
    args = ap.parse_args()

    if not os.path.exists(args.video):
        print(f"HATA: Video bulunamadı: {args.video}"); sys.exit(1)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"HATA: Video açılamadı"); sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    dur = total_frames / fps
    sc = UIScale(fw, fh)
    scale = args.scale

    try:
        model_path = resolve_model_path(args.model)
    except Exception as e:
        print(f"HATA: Model yuklenemedi: {e}")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"  HALI SAYICI v8 - OBB + ByteTrack")
    print(f"{'='*60}")
    print(f"  Video : {args.video} ({fw}x{fh}, {fps:.0f}fps, {dur:.0f}s)")
    print(f"  Model : {model_path}")
    print(f"  Conf  : {args.conf}  |  IoU: {args.iou}  |  imgsz: {args.imgsz}")
    print(f"{'='*60}\n")

    from ultralytics import YOLO
    print("Model yükleniyor...")
    model = YOLO(model_path)
    print(f"Model hazır. Sınıflar: {list(model.names.values())}\n")

    conf_dir = os.path.dirname(os.path.abspath(__file__))
    tracker_yaml = create_tracker_config(conf_dir)

    ret, first_frame = cap.read()
    if not ret: print("HATA: Video okunamadı"); sys.exit(1)

    # ═══ ADIM 1: SAYIM ÇİZGİSİ ═══
    line_pts = []
    win_l = "Sayim Cizgisi Ciz"
    cv2.namedWindow(win_l, cv2.WINDOW_AUTOSIZE)

    def line_cb(ev, x, y, fl, p):
        if ev == cv2.EVENT_LBUTTONDOWN and len(line_pts) < 2:
            line_pts.append((x, y))
            # Ölçeklenmiş koordinatı orijinale çevir
            print(f"  P{len(line_pts)}: ({x}, {y}) -> orijinal: ({int(x/scale)}, {int(y/scale)})")

    cv2.setMouseCallback(win_l, line_cb)

    print("  ADIM 1: Sayım çizgisi çizin (2 nokta tıklayın)")
    print("  SPACE=Onayla | R=Sıfırla | Q=Çıkış\n")

    while True:
        small_w = int(fw * scale); small_h = int(fh * scale)
        small = cv2.resize(first_frame, (small_w, small_h), interpolation=cv2.INTER_AREA)

        # Kılavuz
        cv2.line(small, (0, small_h//2), (small_w, small_h//2), (60,60,60), 1)
        cv2.line(small, (small_w//2, 0), (small_w//2, small_h), (60,60,60), 1)

        for i, pt in enumerate(line_pts):
            cv2.circle(small, (pt[0], pt[1]), 8, (0,255,255), -1)
            cv2.putText(small, f"P{i+1}", (pt[0]+12, pt[1]-8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)

        if len(line_pts) == 2:
            cv2.line(small, line_pts[0], line_pts[1], (0,0,255), 2)
            cv2.putText(small, "SPACE=Onayla | R=Sifirla", (10, small_h-15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        else:
            cv2.putText(small, f"{2-len(line_pts)} nokta tikla", (10, small_h-15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        cv2.imshow(win_l, small)
        k = cv2.waitKey(30) & 0xFF
        if k == ord(' ') and len(line_pts) == 2: break
        elif k == ord('r'): line_pts = []
        elif k in (ord('q'), 27):
            cap.release(); cv2.destroyAllWindows(); sys.exit(0)

    cv2.destroyWindow(win_l)

    # Küçük ekran koordinatlarını orijinale çevir
    ls = (int(line_pts[0][0] / scale), int(line_pts[0][1] / scale))
    le = (int(line_pts[1][0] / scale), int(line_pts[1][1] / scale))

    # ═══ ADIM 2: GİRİŞ YÖNÜ ═══
    ed = 1
    win_d = "Giris Yonu"
    cv2.namedWindow(win_d, cv2.WINDOW_AUTOSIZE)

    print(f"\n  ADIM 2: Giriş yönünü seçin (1 veya 2)")

    while True:
        small = cv2.resize(first_frame, (int(fw*scale), int(fh*scale)), interpolation=cv2.INTER_AREA)
        # Çizgiyi küçük ekranda göster
        ls_s = (int(ls[0]*scale), int(ls[1]*scale))
        le_s = (int(le[0]*scale), int(le[1]*scale))
        cv2.line(small, ls_s, le_s, (0,0,255), 2)
        ssc = UIScale(int(fw*scale), int(fh*scale))
        draw_arrow(small, ls_s, le_s, 1, "1: GIRIS", (0,255,0), ssc)
        draw_arrow(small, ls_s, le_s, -1, "2: GIRIS", (0,165,255), ssc)
        cv2.putText(small, "1 veya 2 basin", (10, int(fh*scale)-15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        cv2.imshow(win_d, small)
        k = cv2.waitKey(30) & 0xFF
        if k == ord('1'): ed = 1; break
        elif k == ord('2'): ed = -1; break
        elif k in (ord('q'), 27):
            cap.release(); cv2.destroyAllWindows(); sys.exit(0)

    cv2.destroyWindow(win_d)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    print(f"  Çizgi: {ls} -> {le}, Yön: {'1' if ed==1 else '2'}\n")

    # ═══ SAYIM DÖNGÜSÜ ═══
    win = "Hali Sayici v8"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    disp_sc = min(1920/fw, 1080/fh, 1.0)
    cv2.resizeWindow(win, int(fw*disp_sc), int(fh*disp_sc))

    tracks = {}
    ci = co = fn = 0
    details = []; ct = args.conf
    paused = False; flash = {}
    t0 = time.time()

    writer = None
    if args.output:
        writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"), fps, (fw, fh))

    print(f"  SAYIM BAŞLADI | P=duraklat Q=çık +/-=eşik S=screenshot\n")

    while True:
        if paused:
            k = cv2.waitKey(50) & 0xFF
            if k == ord("p"): paused = False
            elif k in (ord("q"), 27): break
            continue

        ret, frame = cap.read()
        if not ret: break
        fn += 1

        # ── YOLO OBB + ByteTrack ──
        try:
            results = model.track(
                frame, conf=ct, iou=args.iou, imgsz=args.imgsz,
                persist=True, tracker=tracker_yaml, verbose=False
            )
        except:
            results = model.track(
                frame, conf=ct, iou=args.iou, imgsz=args.imgsz,
                persist=True, verbose=False
            )

        active_ids = set()

        for r in results:
            # OBB model
            if hasattr(r, 'obb') and r.obb is not None and len(r.obb) > 0:
                ids = r.obb.id
                if ids is None: continue
                for i in range(len(r.obb)):
                    tid = int(ids[i])
                    cls_name = model.names.get(int(r.obb.cls[i]), "?")
                    conf = float(r.obb.conf[i])
                    xywhr = r.obb.xywhr[i].cpu().numpy()
                    cx, cy, w, h = float(xywhr[0]), float(xywhr[1]), float(xywhr[2]), float(xywhr[3])
                    if w * h < args.min_size: continue
                    obb_pts = r.obb.xyxyxyxy[i].cpu().numpy()
                    xc = obb_pts[:, 0]; yc = obb_pts[:, 1]

                    active_ids.add(tid)
                    if tid not in tracks:
                        clr = tuple(int(c) for c in np.random.randint(100, 255, 3))
                        tracks[tid] = TrackInfo(first_seen=fn, color=clr)

                    ti = tracks[tid]
                    ti.positions.append((cx, cy))
                    ti.last_seen = fn; ti.frames_seen += 1
                    ti.confidence_sum += conf; ti.class_name = cls_name
                    ti.obb_points = obb_pts
                    ti.bbox = (float(xc.min()), float(yc.min()), float(xc.max()), float(yc.max()))
                    if len(ti.positions) > 100: ti.positions = ti.positions[-100:]

            # Normal detection fallback
            elif r.boxes is not None:
                ids = r.boxes.id
                if ids is None: continue
                for i in range(len(r.boxes)):
                    tid = int(ids[i])
                    cls_name = model.names.get(int(r.boxes.cls[i]), "?")
                    conf = float(r.boxes.conf[i])
                    x1, y1, x2, y2 = r.boxes.xyxy[i].cpu().numpy()
                    w, h = x2-x1, y2-y1
                    if w * h < args.min_size: continue
                    cx, cy = float((x1+x2)/2), float((y1+y2)/2)

                    active_ids.add(tid)
                    if tid not in tracks:
                        clr = tuple(int(c) for c in np.random.randint(100, 255, 3))
                        tracks[tid] = TrackInfo(first_seen=fn, color=clr)

                    ti = tracks[tid]
                    ti.positions.append((cx, cy))
                    ti.last_seen = fn; ti.frames_seen += 1
                    ti.confidence_sum += conf; ti.class_name = cls_name
                    ti.obb_points = None
                    ti.bbox = (float(x1), float(y1), float(x2), float(y2))
                    if len(ti.positions) > 100: ti.positions = ti.positions[-100:]

        # ── Çizgi geçiş ──
        for tid in active_ids:
            ti = tracks[tid]
            if ti.center is None: continue
            cs = get_side_of_line(ti.center, ls, le)
            dl = point_to_line_distance(ti.center, ls, le)

            if ti.last_side == 0 and cs != 0: ti.last_side = cs; continue
            if ti.frames_seen < args.min_frames:
                if cs != 0: ti.last_side = cs; continue
            if ti.cross_count == 0 and ti.total_travel < args.min_travel:
                if cs != 0: ti.last_side = cs; continue
            if dl > args.recount_dist: ti.can_cross = True
            if ti.last_cross_frame > 0 and (fn-ti.last_cross_frame) < args.cooldown_frames:
                if cs != 0: ti.last_side = cs; continue

            if cs != 0 and ti.last_side != 0 and cs != ti.last_side and ti.can_cross:
                dr = cs; ts = fn / fps
                if dr == ed: ci += 1; et = "GIRIS"; flash[fn] = 1
                else: co += 1; et = "CIKIS"; flash[fn] = -1
                ti.cross_count += 1; ti.last_cross_frame = fn
                ti.last_cross_direction = dr; ti.can_cross = False
                net = ci - co
                rc = f" (#{ti.cross_count})" if ti.cross_count > 1 else ""
                details.append((fn, tid, et, ts, ti.avg_confidence, ti.cross_count))
                print(f"  [{et:5s}] Net:{net:3d} | F{fn} t={ts:.1f}s | T{tid} conf={ti.avg_confidence:.2f}{rc}")
            if cs != 0: ti.last_side = cs

        # Temizle
        for tid in [t for t in tracks if fn-tracks[t].last_seen > 150 and t not in active_ids]:
            del tracks[tid]

        # ── Çizim ──
        vis = frame.copy()

        # Flaş
        for ff2, fd in list(flash.items()):
            age = fn - ff2
            if age < 12:
                ov = vis.copy()
                c = (0,255,0) if fd == 1 else (0,0,255)
                cv2.line(ov, ls, le, c, sc.thick(18))
                cv2.addWeighted(ov, 0.35*(1-age/12), vis, 1-0.35*(1-age/12), 0, vis)
            else: del flash[ff2]

        # Çizgi
        cv2.line(vis, ls, le, (0,0,255), sc.thick(3))
        draw_arrow(vis, ls, le, ed, "GIRIS", (0,255,0), sc)

        # Kutular + izler
        for tid in active_ids:
            ti = tracks.get(tid)
            if not ti or not ti.center: continue

            if ti.cross_count > 0:
                bc = (0,255,0) if ti.last_cross_direction == ed else (0,0,255)
            elif ti.frames_seen >= args.min_frames:
                bc = (255,200,0)
            else:
                bc = (150,150,150)

            # OBB veya normal kutu
            if ti.obb_points is not None:
                draw_obb(vis, ti.obb_points, bc, sc.thick(2))
            else:
                x1,y1,x2,y2 = [int(v) for v in ti.bbox]
                cv2.rectangle(vis, (x1,y1), (x2,y2), bc, sc.thick(2))

            cx, cy = int(ti.center[0]), int(ti.center[1])
            draw_trail(vis, ti.positions, bc, sc)
            cv2.circle(vis, (cx, cy), sc.px(5), bc, -1)

            ci2 = "" if ti.cross_count == 0 else f" x{ti.cross_count}"
            lb = f"T{tid} {ti.avg_confidence:.2f}{ci2}"
            lfs = sc.font(0.5); lth = sc.thick(2)
            (ltw, lht), _ = cv2.getTextSize(lb, cv2.FONT_HERSHEY_SIMPLEX, lfs, lth)
            cv2.rectangle(vis, (cx-sc.px(2), cy-lht-sc.px(14)), (cx+ltw+sc.px(6), cy-sc.px(2)), bc, -1)
            cv2.putText(vis, lb, (cx+sc.px(2), cy-sc.px(5)),
                        cv2.FONT_HERSHEY_SIMPLEX, lfs, (0,0,0), lth)

        # Sayaçlar
        net = ci - co
        draw_counts(vis, ci, co, net, sc)

        # Bilgi
        el = time.time() - t0
        prog = fn/total_frames*100 if total_frames > 0 else 0
        draw_info(vis, [
            f"Frame: {fn}/{total_frames}",
            f"Video: {fn/fps:.1f}s / {dur:.1f}s | %{prog:.1f}",
            f"Aktif: {len(active_ids)} | ByteTrack + OBB",
            f"Conf: {ct:.2f} | IoU: {args.iou}",
        ], sc)

        # Progress bar
        bh = sc.px(6); bw = int(fw * prog / 100)
        cv2.rectangle(vis, (0,fh-bh), (fw,fh), (30,30,30), -1)
        cv2.rectangle(vis, (0,fh-bh), (bw,fh), (0,255,100), -1)

        cv2.imshow(win, vis)
        if writer: writer.write(vis)

        k = cv2.waitKey(1) & 0xFF
        if k in (ord("q"), 27): break
        elif k == ord("p"): paused = True; print("  Duraklatıldı")
        elif k == ord("s"):
            sfn = f"screenshot_{fn}.png"; cv2.imwrite(sfn, vis); print(f"  Kaydedildi: {sfn}")
        elif k in (ord("+"), ord("=")):
            ct = min(0.95, ct+0.05); print(f"  Eşik: {ct:.2f}")
        elif k == ord("-"):
            ct = max(0.05, ct-0.05); print(f"  Eşik: {ct:.2f}")

    cap.release()
    if writer: writer.release()
    cv2.destroyAllWindows()
    try: os.remove(tracker_yaml)
    except: pass

    el = time.time() - t0; net = ci - co
    print(f"\n{'='*60}\n  SONUÇLAR\n{'='*60}")
    print(f"  Giriş: {ci}\n  Çıkış: {co}\n  NET: {net}")
    if el > 0: print(f"  Süre: {el:.1f}s ({fn/el:.1f} fps)")
    print(f"{'='*60}")

    if details:
        print(f"\n  {'#':<4} {'Tür':<6} {'Frame':<7} {'Zaman':<8} {'TrkID':<7} {'Güven':<6}")
        print(f"  {'-'*42}")
        for i, (f2, t2, e2, ts2, cf2, cc2) in enumerate(details, 1):
            print(f"  {i:<4} {e2:<6} {f2:<7} {ts2:<8.1f}s {t2:<7} {cf2:<6.2f}")

    rpt = os.path.splitext(args.video)[0] + "_sayim_raporu.txt"
    with open(rpt, "w", encoding="utf-8") as f:
        f.write(f"HALI SAYIM RAPORU v8 (OBB + ByteTrack)\n{'='*40}\n")
        f.write(f"Video: {args.video}\nModel: {args.model}\n")
        f.write(f"Çizgi: {ls} -> {le}\nConf: {ct} | IoU: {args.iou}\n\n")
        f.write(f"Giriş: {ci}\nÇıkış: {co}\nNET: {net}\n\n")
        for i, (f2, t2, e2, ts2, cf2, cc2) in enumerate(details, 1):
            f.write(f"{i}. [{e2}] Frame {f2}, t={ts2:.1f}s, T{t2}, conf={cf2:.2f}\n")
    print(f"\n  Rapor: {rpt}\n")


if __name__ == "__main__":
    main()
