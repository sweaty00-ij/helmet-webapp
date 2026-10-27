import tempfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from ultralytics import YOLO

from src.detection import IMAGE_EXT, VIDEO_EXT, annotate, detect, extract_frames

ROOT = Path(__file__).parent
MODEL_PATH = ROOT / "yolo_10epoch_best.pt"

st.set_page_config(page_title="Smart Helmet Detection", page_icon="🏍️", layout="wide")


@st.cache_resource
def load_model():
    return YOLO(str(MODEL_PATH))


def read_image(uploaded):
    data = np.frombuffer(uploaded.getvalue(), np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def to_rgb(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def adjust_contrast(image, factor):
    mean = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).mean()
    out = mean + factor * (image.astype(np.float32) - mean)
    return np.clip(out, 0, 255).astype(np.uint8)


def equalize(image):
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)



def rate(helmet, total):
    return helmet / total if total else None

RIDER_CONF = 0.25
HELMET_CONF = 0.25
N_FRAMES = 5
LOW_CONTRAST = 0.3


def read_video(upload):
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
    if "uploader_key" not in st.session_state:
        st.session_state.uploader_key = 0

    uploads = st.file_uploader("Upload image(s) or video", type=IMAGE_EXT + VIDEO_EXT,
                               accept_multiple_files=True,
                               key=f"uploader_{st.session_state.uploader_key}")

    if uploads and st.button("🗑️ Clear all"):
        st.session_state.uploader_key += 1
        st.rerun()

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


def summarize(results):
    riders = sum(r.n_riders for r in results)
    helmet = sum(r.n_helmet for r in results)
    return riders, helmet, riders - helmet, rate(helmet, riders)


def show_detection(images, names):
    model = load_model()

    with st.spinner("Detecting..."):
        degraded = [adjust_contrast(img, LOW_CONTRAST) for img in images]
        equalized = [equalize(img) for img in degraded]
        results = [detect(model, img, RIDER_CONF, HELMET_CONF) for img in images]
        results_low = [detect(model, img, RIDER_CONF, HELMET_CONF) for img in degraded]
        results_eq = [detect(model, img, RIDER_CONF, HELMET_CONF) for img in equalized]

    riders, helmet, no_helmet, compliance = summarize(results)

    st.subheader("Summary")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Images / Frames", len(results))
    c2.metric("Riders", riders)
    c3.metric("With helmet", helmet)
    c4.metric("No helmet", no_helmet)
    c5.metric("Compliance", f"{compliance:.0%}" if compliance is not None else "-")

    st.subheader("Histogram Equalization: before vs after")
    st.caption(f"Image Processing — ลด Contrast ×{LOW_CONTRAST} จำลองภาพแย่ แล้วแก้ด้วย Histogram Equalization")

    compare_df = pd.DataFrame(
        [summarize(results), summarize(results_low), summarize(results_eq)],
        index=["Original", f"Low contrast ×{LOW_CONTRAST}", "Histogram Equalization"],
        columns=["riders", "with_helmet", "no_helmet", "compliance_rate"],
    )

    st.dataframe(compare_df, column_config={"compliance_rate": st.column_config.NumberColumn(format="percent")})

    stats_df = pd.DataFrame([{
        "input": name,
        "riders": r.n_riders,
        "with_helmet": r.n_helmet,
        "no_helmet": r.n_no_helmet,
        "with_helmet (low)": rl.n_helmet,
        "no_helmet (low)": rl.n_no_helmet,
        "with_helmet (HE)": rh.n_helmet,
        "no_helmet (HE)": rh.n_no_helmet,
        "inference_ms": round(r.inference_ms, 1),
    } for name, r, rl, rh in zip(names, results, results_low, results_eq)])

    left, right = st.columns(2)
    with left:
        st.bar_chart(stats_df, x="input", y=["with_helmet", "no_helmet"],
                     color=["#2ca02c", "#d62728"], x_label="", y_label="riders (original)")
    with right:
        st.dataframe(stats_df, hide_index=True)
        st.download_button("⬇️ Download CSV", stats_df.to_csv(index=False), "helmet_stats.csv", "text/csv")

    st.subheader("Results")
    st.caption("🟩 สวมหมวก  🟥 ไม่สวมหมวก  🟨 หมวกที่ตรวจพบ")

    for name, original, low, eq, r, rl, rh in zip(names, images, degraded, equalized,
                                                    results, results_low, results_eq):

        st.markdown(f"**{name}** — original: helmet {r.n_helmet}/{r.n_riders} | "
                    f"low ×{LOW_CONTRAST}: helmet {rl.n_helmet}/{rl.n_riders} | "
                    f"HE: helmet {rh.n_helmet}/{rh.n_riders}")

        cols = st.columns(3)
        cols[0].image(to_rgb(annotate(original, r)), caption="Original")
        cols[1].image(to_rgb(annotate(low, rl)), caption=f"Low contrast ×{LOW_CONTRAST}")
        cols[2].image(to_rgb(annotate(eq, rh)), caption="Histogram Equalization")


st.title("🏍️ Smart Helmet Detection")
st.caption("ตรวจว่าผู้ขับขี่มอเตอร์ไซค์สวมหมวกกันน็อกหรือไม่ — YOLO11 + OpenCV")

images, names = get_inputs()
if images:
    show_detection(images, names)
else:
    st.info("อัปโหลดภาพหรือวิดีโอเพื่อเริ่มตรวจ")
