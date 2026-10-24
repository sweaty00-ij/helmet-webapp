"""Image Processing: Contrast Enhancement (เทคนิคเดียวกับที่ใช้ตอนเทรนใน Week 1)"""

import cv2
import numpy as np


def contrast_enhance(image, factor=2.0):
    """เพิ่ม/ลด Contrast ของภาพ BGR

    ทำแบบเดียวกับ PIL ImageEnhance.Contrast ที่ใช้ใน Week 1:
    ดึงค่าสีแต่ละ pixel ให้ห่างจากค่าความสว่างเฉลี่ยของภาพ factor เท่า
    factor > 1 = contrast มากขึ้น, factor < 1 = contrast น้อยลง, 1 = ภาพเดิม
    """

    mean = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).mean()
    out = mean + factor * (image.astype(np.float32) - mean)

    return np.clip(out, 0, 255).astype(np.uint8)
