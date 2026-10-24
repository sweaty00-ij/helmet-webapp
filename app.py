"""Smart Helmet Detection — Streamlit Web Application

รัน:  streamlit run app.py
"""

import json
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
RESULTS_DIR = ROOT / "results"

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
# Sidebar: ตั้งค่า
# ==============================

st.sidebar.header("⚙️ Settings")

input_type = st.sidebar.radio("Input", ["Image", "Video"], horizontal=True)
n_frames = st.sidebar.slider("Frames from video", 1, 20, 5) if input_type == "Video" else None

st.sidebar.subheader("Detection")
rider_conf = st.sidebar.slider("Rider confidence", 0.05, 0.95, 0.25, 0.05)
helmet_conf = st.sidebar.slider("Helmet confidence", 0.05, 0.95, 0.25, 0.05,
                                help="โมเดลมักให้ confidence ของหมวกต่ำกว่า rider")

st.sidebar.subheader("Image Processing")
use_contrast = st.sidebar.checkbox("Contrast Enhancement")
contrast_factor = st.sidebar.slider("Contrast factor", 1.0, 3.0, 1.5, 0.1, disabled=not use_contrast)


# ==============================
# Input: รับ Image / Video
# ==============================

def get_inputs():
    """แสดงช่องอัปโหลด คืน (images, names) — ภาพ BGR, วิดีโอจะถูกแบ่งเป็นเฟรม"""

    if input_type == "Image":
        images, names = [], []
        uploads = st.file_uploader("Upload image(s)", type=IMAGE_EXT, accept_multiple_files=True)
        for f in uploads or []:
            image = read_image(f)
            if image is None:
                st.warning(f"อ่านไฟล์ไม่ได้: {f.name}")
                continue
            images.append(image)
            names.append(f.name)
        return images, names

    upload = st.file_uploader("Upload video", type=VIDEO_EXT)
    if not upload:
        return [], []

    st.video(upload)

    # OpenCV อ่านวิดีโอจาก path เท่านั้น จึงเขียนลงไฟล์ชั่วคราวก่อน
    with tempfile.NamedTemporaryFile(suffix=Path(upload.name).suffix, delete=False) as tmp:
        tmp.write(upload.getvalue())
    try:
        frames, names, info = extract_frames(tmp.name, n_frames)
    except ValueError as e:
        st.error(f"อ่านวิดีโอไม่ได้: {e}")
        return [], []
    finally:
        Path(tmp.name).unlink(missing_ok=True)

    st.caption(f"{info['width']}x{info['height']}, {info['fps']:.0f} FPS, "
               f"{info['frames']} frames ({info['seconds']:.1f} s) → ตรวจ {len(frames)} เฟรม")
    return frames, names


# ==============================
# Process -> แสดงผล + สถิติ
# ==============================

def show_detection(images, names):

    model = load_model()

    with st.spinner("Detecting..."):
        processed = [contrast_enhance(img, contrast_factor) if use_contrast else img for img in images]
        results = [detect(model, img, rider_conf, helmet_conf) for img in processed]
        annotated = [annotate(img, res) for img, res in zip(processed, results)]

    # ---------- สถิติรวม ----------
    riders = sum(r.n_riders for r in results)
    helmet = sum(r.n_helmet for r in results)
    compliance = rate(helmet, riders)

    st.subheader("Summary")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Images / Frames", len(results))
    c2.metric("Riders", riders)
    c3.metric("With helmet", helmet)
    c4.metric("No helmet", riders - helmet)
    c5.metric("Compliance", f"{compliance:.0%}" if compliance is not None else "-")

    stats_df = pd.DataFrame([{
        "input": name,
        "riders": r.n_riders,
        "with_helmet": r.n_helmet,
        "no_helmet": r.n_no_helmet,
        "unmatched_helmets": len(r.unmatched_helmets),
        "compliance_rate": rate(r.n_helmet, r.n_riders),
        "inference_ms": round(r.inference_ms, 1),
    } for name, r in zip(names, results)])

    left, right = st.columns(2)
    with left:
        st.bar_chart(stats_df, x="input", y=["with_helmet", "no_helmet"],
                     color=["#2ca02c", "#d62728"], x_label="", y_label="riders")
    with right:
        st.dataframe(stats_df, hide_index=True)
        st.download_button("⬇️ Download CSV", stats_df.to_csv(index=False), "helmet_stats.csv", "text/csv")

    # ---------- ผลรายภาพ ----------
    st.subheader("Results")
    st.caption("🟩 สวมหมวก  🟥 ไม่สวมหมวก  🟨 หมวกที่ตรวจพบ")

    for name, original, enhanced, image, res in zip(names, images, processed, annotated, results):

        st.markdown(f"**{name}** — riders {res.n_riders}, helmet {res.n_helmet}, "
                    f"no helmet {res.n_no_helmet}, {res.inference_ms:.0f} ms")

        if use_contrast:
            cols = st.columns(3)
            cols[0].image(to_rgb(original), caption="Original")
            cols[1].image(to_rgb(enhanced), caption=f"Contrast Enhancement (×{contrast_factor})")
            cols[2].image(to_rgb(image), caption="Result")
        else:
            cols = st.columns(2)
            cols[0].image(to_rgb(original), caption="Original")
            cols[1].image(to_rgb(image), caption="Result")


# ==============================
# Evaluation (ผลจาก evaluate.py)
# ==============================

def show_evaluation():

    metrics_file = RESULTS_DIR / "metrics.json"
    if not metrics_file.exists():
        st.info("ยังไม่มีผลประเมิน — รัน `python evaluate.py --test path/to/test` ก่อน")
        return

    m = json.loads(metrics_file.read_text(encoding="utf-8"))

    st.subheader(f"Test set: {m['test_images']} images — model `{m['model']}`")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("mAP50", f"{m['mAP50']:.3f}")
    c2.metric("mAP50-95", f"{m['mAP50-95']:.3f}")
    c3.metric("Precision", f"{m['precision']:.3f}")
    c4.metric("Recall", f"{m['recall']:.3f}")

    st.markdown("**Per class**")
    st.dataframe(pd.DataFrame(m["per_class"]).T.round(3))
    st.caption("ค่าเฉลี่ยรวมด้านบนถูกดึงลงโดยคลาส motorcycle ซึ่งมีใน Test set แค่ 1 กล่อง "
               "— คลาสที่แอปใช้จริงคือ rider และ with helmet")

    plots = [
        ("confusion_matrix_normalized.png", "Confusion Matrix (normalized)"),
        ("BoxPR_curve.png", "Precision-Recall Curve"),
        ("confusion_matrix.png", "Confusion Matrix"),
        ("BoxF1_curve.png", "F1-Confidence Curve"),
    ]
    plots = [(RESULTS_DIR / f, title) for f, title in plots if (RESULTS_DIR / f).exists()]
    for i in range(0, len(plots), 2):
        for col, (path, title) in zip(st.columns(2), plots[i:i + 2]):
            col.image(str(path), caption=title)


# ==============================
# Page
# ==============================

st.title("🏍️ Smart Helmet Detection")
st.caption("ตรวจว่าผู้ขับขี่มอเตอร์ไซค์สวมหมวกกันน็อกหรือไม่ — YOLO11 + OpenCV")

tab_detect, tab_eval = st.tabs(["Detect", "Model Evaluation"])

with tab_detect:
    images, names = get_inputs()
    if images:
        show_detection(images, names)
    else:
        st.info("อัปโหลดภาพหรือวิดีโอเพื่อเริ่มตรวจ")

with tab_eval:
    show_evaluation()
