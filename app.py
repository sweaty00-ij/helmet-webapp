"""Smart Helmet Detection — Streamlit Web Application

รัน:  streamlit run app.py
"""

import tempfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from ultralytics import YOLO

from src.detection import IMAGE_EXT, VIDEO_EXT, annotate, detect, extract_frames
from src.preprocessing import contrast_enhance

ROOT = Path(__file__).parent
MODEL_PATH = ROOT / "yolo_10epoch_best.pt"

st.set_page_config(page_title="Smart Helmet Detection", page_icon="🏍️", layout="wide")


@st.cache_resource
def load_model():
    return YOLO(str(MODEL_PATH))


def read_image(uploaded):
    """ไฟล์ที่อัปโหลด -> ภาพ BGR (OpenCV) หรือ None ถ้าอ่านไม่ได้"""

    data = np.frombuffer(uploaded.getvalue(), np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def to_rgb(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def rate(helmet, total):
    return helmet / total if total else None


# ==============================
# Input: รับ Image / Video (ช่องเดียว แยกจากนามสกุลไฟล์)
# ==============================

RIDER_CONF = 0.25    # confidence ขั้นต่ำของ rider
HELMET_CONF = 0.25   # confidence ขั้นต่ำของหมวก
N_FRAMES = 5         # วิดีโอ: จำนวนเฟรมที่แบ่งออกมาตรวจ
CONTRAST = 1.5      # Image Processing: Contrast Enhancement factor (เทียบผลก่อน/หลัง)


def read_video(upload):
    """วิดีโอที่อัปโหลด -> (frames, names) แบ่งออกมา N_FRAMES เฟรม"""

    # OpenCV อ่านวิดีโอจาก path เท่านั้น จึงเขียนลงไฟล์ชั่วคราวก่อน
    with tempfile.NamedTemporaryFile(suffix=Path(upload.name).suffix, delete=False) as tmp:
        tmp.write(upload.getvalue())
    try:
        frames, names, info = extract_frames(tmp.name, N_FRAMES)
    except ValueError as e:
        st.error(f"อ่านวิดีโอไม่ได้: {upload.name} ({e})")
        return [], []
    finally:
        Path(tmp.name).unlink(missing_ok=True)

    st.video(upload)
    st.caption(f"{upload.name}: {info['width']}x{info['height']}, {info['fps']:.0f} FPS, "
               f"{info['seconds']:.1f} s → ตรวจ {len(frames)} เฟรม")
    return frames, [f"{upload.name} {n}" for n in names]


def get_inputs():
    """แสดงช่องอัปโหลด คืน (images, names) — ภาพ BGR, วิดีโอจะถูกแบ่งเป็นเฟรม"""

    uploads = st.file_uploader("Upload image(s) or video", type=IMAGE_EXT + VIDEO_EXT,
                               accept_multiple_files=True)
    images, names = [], []

    for f in uploads or []:
        if Path(f.name).suffix.lower().lstrip(".") in VIDEO_EXT:
            frames, frame_names = read_video(f)
            images += frames
            names += frame_names
            continue

        image = read_image(f)
        if image is None:
            st.warning(f"อ่านไฟล์ไม่ได้: {f.name}")
            continue
        images.append(image)
        names.append(f.name)

    return images, names


# ==============================
# Process -> แสดงผล + สถิติ
# ==============================

def summarize(results):
    riders = sum(r.n_riders for r in results)
    helmet = sum(r.n_helmet for r in results)
    return riders, helmet, riders - helmet, rate(helmet, riders)


def show_detection(images, names):

    model = load_model()

    # ตรวจ 2 แบบ: ภาพต้นฉบับ และภาพหลัง Contrast Enhancement
    with st.spinner("Detecting..."):
        enhanced = [contrast_enhance(img, CONTRAST) for img in images]
        results = [detect(model, img, RIDER_CONF, HELMET_CONF) for img in images]
        results_ce = [detect(model, img, RIDER_CONF, HELMET_CONF) for img in enhanced]

    # ---------- สถิติรวม (ภาพต้นฉบับ) ----------
    riders, helmet, no_helmet, compliance = summarize(results)

    st.subheader("Summary")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Images / Frames", len(results))
    c2.metric("Riders", riders)
    c3.metric("With helmet", helmet)
    c4.metric("No helmet", no_helmet)
    c5.metric("Compliance", f"{compliance:.0%}" if compliance is not None else "-")

    # ---------- เทียบก่อน/หลัง Contrast Enhancement ----------
    st.subheader("Contrast Enhancement: before vs after")
    st.caption(f"Image Processing — เพิ่ม Contrast ×{CONTRAST} ก่อนส่งเข้า YOLO แล้วเทียบผลกับภาพต้นฉบับ")

    compare_df = pd.DataFrame(
        [summarize(results), summarize(results_ce)],
        index=["Original", f"Contrast Enhancement ×{CONTRAST}"],
        columns=["riders", "with_helmet", "no_helmet", "compliance_rate"],
    )
    st.dataframe(compare_df, column_config={"compliance_rate": st.column_config.NumberColumn(format="percent")})

    # ---------- ตารางรายภาพ ----------
    stats_df = pd.DataFrame([{
        "input": name,
        "riders": r.n_riders,
        "with_helmet": r.n_helmet,
        "no_helmet": r.n_no_helmet,
        "riders (CE)": rc.n_riders,
        "with_helmet (CE)": rc.n_helmet,
        "no_helmet (CE)": rc.n_no_helmet,
        "inference_ms": round(r.inference_ms, 1),
    } for name, r, rc in zip(names, results, results_ce)])

    left, right = st.columns(2)
    with left:
        st.bar_chart(stats_df, x="input", y=["with_helmet", "no_helmet"],
                     color=["#2ca02c", "#d62728"], x_label="", y_label="riders (original)")
    with right:
        st.dataframe(stats_df, hide_index=True)
        st.download_button("⬇️ Download CSV", stats_df.to_csv(index=False), "helmet_stats.csv", "text/csv")

    # ---------- ผลรายภาพ ----------
    st.subheader("Results")
    st.caption("🟩 สวมหมวก  🟥 ไม่สวมหมวก  🟨 หมวกที่ตรวจพบ")

    for name, original, ce, r, rc in zip(names, images, enhanced, results, results_ce):

        st.markdown(f"**{name}** — original: helmet {r.n_helmet}/{r.n_riders} | "
                    f"contrast ×{CONTRAST}: helmet {rc.n_helmet}/{rc.n_riders}")

        cols = st.columns(4)
        cols[0].image(to_rgb(original), caption="Original")
        cols[1].image(to_rgb(annotate(original, r)), caption="Result (original)")
        cols[2].image(to_rgb(ce), caption=f"Contrast Enhancement ×{CONTRAST}")
        cols[3].image(to_rgb(annotate(ce, rc)), caption="Result (enhanced)")


# ==============================
# Page
# ==============================

st.title("🏍️ Smart Helmet Detection")
st.caption("ตรวจว่าผู้ขับขี่มอเตอร์ไซค์สวมหมวกกันน็อกหรือไม่ — YOLO11 + OpenCV")

images, names = get_inputs()
if images:
    show_detection(images, names)
else:
    st.info("อัปโหลดภาพหรือวิดีโอเพื่อเริ่มตรวจ")
