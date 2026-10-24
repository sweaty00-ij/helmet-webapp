"""Detection: YOLO11 (rider, with helmet) -> Helmet / No Helmet ราย rider

ย้ายมาจาก HelmetAppW2.ipynb (Week 2)
"""

import math
from dataclasses import dataclass, field

import cv2
import numpy as np

HEAD_PART = 0.5    # ครึ่งบนของกล่อง rider = บริเวณศีรษะ
TOP_MARGIN = 0.1   # ยอมให้หมวกโผล่เหนือกล่อง rider ได้ 10% ของความสูง
HEAD_Y = 0.15      # ตำแหน่งศีรษะโดยประมาณ = 15% จากขอบบนของกล่อง rider

IMAGE_EXT = ["jpg", "jpeg", "png", "bmp", "webp"]
VIDEO_EXT = ["mp4", "avi", "mov", "mkv", "m4v", "wmv"]


@dataclass
class Detection:
    box: tuple      # (x1, y1, x2, y2) พิกัด pixel
    conf: float

    @property
    def center(self):
        x1, y1, x2, y2 = self.box
        return (x1 + x2) / 2, (y1 + y2) / 2


@dataclass
class Rider(Detection):
    helmet: Detection | None = None   # หมวกที่จับคู่ได้ (None = ไม่สวมหมวก)

    @property
    def has_helmet(self):
        return self.helmet is not None


@dataclass
class FrameResult:
    riders: list = field(default_factory=list)
    unmatched_helmets: list = field(default_factory=list)   # หมวกที่ไม่มี rider จับคู่
    inference_ms: float = 0.0

    @property
    def n_riders(self):
        return len(self.riders)

    @property
    def n_helmet(self):
        return sum(r.has_helmet for r in self.riders)

    @property
    def n_no_helmet(self):
        return self.n_riders - self.n_helmet


def class_ids(model):
    """คืน (rider_id, helmet_id) จากชื่อคลาสของโมเดล"""

    ids = {name: i for i, name in model.names.items()}
    return ids["rider"], ids["with helmet"]


def match_helmets(riders, helmets):
    """จับคู่หมวกให้ rider (ใส่ค่าใน rider.helmet) คืนรายการหมวกที่จับคู่ไม่ได้

    หมวกเป็นของ rider ถ้าจุดกึ่งกลางหมวกอยู่ในครึ่งบนของกล่อง rider
    หมวก 1 ใบเป็นของ rider ได้คนเดียว ถ้าเข้าเงื่อนไขหลายคน เลือกคนที่ศีรษะอยู่ใกล้หมวกที่สุด
    """

    unmatched = []

    for helmet in sorted(helmets, key=lambda d: d.conf, reverse=True):

        hx, hy = helmet.center
        best, best_dist = None, math.inf

        for rider in riders:
            if rider.has_helmet:
                continue

            x1, y1, x2, y2 = rider.box
            h = y2 - y1
            if not (x1 <= hx <= x2 and y1 - TOP_MARGIN * h <= hy <= y1 + HEAD_PART * h):
                continue

            # ระยะจากหมวกถึงศีรษะ หารด้วยความสูง rider เพื่อให้เทียบกันได้ทุกขนาด
            dist = math.hypot(hx - (x1 + x2) / 2, hy - (y1 + HEAD_Y * h)) / h
            if dist < best_dist:
                best, best_dist = rider, dist

        if best is None:
            unmatched.append(helmet)
        else:
            best.helmet = helmet

    return unmatched


def detect(model, image, rider_conf=0.25, helmet_conf=0.25, nms_iou=0.5):
    """ตรวจภาพ BGR 1 ภาพ คืนค่า FrameResult"""

    rider_id, helmet_id = class_ids(model)

    result = model.predict(
        image,
        conf=min(rider_conf, helmet_conf),
        iou=nms_iou,
        classes=[rider_id, helmet_id],
        verbose=False,
    )[0]

    riders, helmets = [], []

    for box, cls, conf in zip(result.boxes.xyxy.tolist(), result.boxes.cls.tolist(), result.boxes.conf.tolist()):
        if cls == rider_id and conf >= rider_conf:
            riders.append(Rider(tuple(box), conf))
        elif cls == helmet_id and conf >= helmet_conf:
            helmets.append(Detection(tuple(box), conf))

    unmatched = match_helmets(riders, helmets)

    return FrameResult(riders, unmatched, inference_ms=sum(result.speed.values()))


def extract_frames(video_path, n=5):
    """แบ่งวิดีโอออกมา n เฟรม กระจายเท่า ๆ กัน (ไม่เอาเฟรมแรก/สุดท้ายสุด ซึ่งมักเป็นภาพดำ)

    คืน (frames, names, info)
    """

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError("cannot open video")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30          # บางไฟล์ไม่บอก FPS (ได้ 0) ใช้ 30 แทน
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if total <= 0:
        cap.release()
        raise ValueError("cannot read frame count")

    frames, names = [], []
    for idx in np.linspace(0, total - 1, n + 2)[1:-1].astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if ok:
            frames.append(frame)
            names.append(f"frame {idx} ({idx / fps:.1f}s)")

    cap.release()
    info = {"width": width, "height": height, "fps": fps, "frames": total, "seconds": total / fps}

    return frames, names, info


# ==============================
# วาดผล (OpenCV)
# ==============================

GREEN = (0, 200, 0)
RED = (0, 0, 230)
LIGHT_RED = (90, 90, 255)
YELLOW = (0, 220, 255)
WHITE = (255, 255, 255)
FONT = cv2.FONT_HERSHEY_SIMPLEX


def draw_detections(image, result):
    """วาดกรอบ rider (เขียว = สวมหมวก / แดง = ไม่สวมหมวก) และกรอบหมวก (เหลือง)"""

    out = image.copy()
    s = max(out.shape[:2]) / 1000                      # ปรับขนาดเส้น/ตัวอักษรตามขนาดภาพ
    thick = max(2, round(2.5 * s))
    font_scale, font_thick = max(0.45, 0.6 * s), max(1, round(1.5 * s))

    for rider in result.riders:
        x1, y1, x2, y2 = map(int, rider.box)
        color = GREEN if rider.has_helmet else RED
        text = f"{'Helmet' if rider.has_helmet else 'No Helmet'} {rider.conf:.2f}"

        cv2.rectangle(out, (x1, y1), (x2, y2), color, thick)

        # ป้ายข้อความพื้นทึบเหนือกรอบ
        (tw, th), base = cv2.getTextSize(text, FONT, font_scale, font_thick)
        ty = max(y1, th + base + 4)
        cv2.rectangle(out, (x1, ty - th - base - 4), (x1 + tw + 6, ty), color, -1)
        cv2.putText(out, text, (x1 + 3, ty - base - 2), FONT, font_scale, WHITE, font_thick, cv2.LINE_AA)

    helmets = [r.helmet for r in result.riders if r.has_helmet] + result.unmatched_helmets
    for helmet in helmets:
        hx1, hy1, hx2, hy2 = map(int, helmet.box)
        cv2.rectangle(out, (hx1, hy1), (hx2, hy2), YELLOW, max(1, thick - 1))

    return out


def add_header(image, result):
    """ต่อแถบสถิติไว้ด้านบนของภาพ"""

    w = image.shape[1]
    scale = min(max(w / 1300, 0.45), 1.0)
    thickness = max(1, round(2 * scale))
    rate = f"{result.n_helmet / result.n_riders:.0%}" if result.n_riders else "-"
    items = [
        (f"Riders: {result.n_riders}", WHITE),
        (f"Helmet: {result.n_helmet}", GREEN),
        (f"No helmet: {result.n_no_helmet}", LIGHT_RED),
        (f"Compliance: {rate}", WHITE),
    ]

    header = np.full((int(55 * scale), w, 3), (35, 35, 35), np.uint8)
    x = int(14 * scale)
    for text, color in items:
        cv2.putText(header, text, (x, int(38 * scale)), FONT, scale, color, thickness, cv2.LINE_AA)
        x += cv2.getTextSize(text, FONT, scale, thickness)[0][0] + int(28 * scale)

    return np.vstack([header, image])


def annotate(image, result):
    """กรอบตรวจจับ + แถบสถิติ"""

    return add_header(draw_detections(image, result), result)
