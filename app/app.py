# =========================================================
# 🚗 DETR Self-Driving Object Detection Dashboard (Final)
# =========================================================

import os
import torch
import streamlit as st
from PIL import Image, ImageDraw
from transformers import DetrForObjectDetection, DetrImageProcessor
import cv2
import tempfile
import time

# ----------------------------
# STREAMLIT PAGE CONFIG
# ----------------------------
st.set_page_config(page_title="🚗 DETR Self-Driving Object Detection", layout="wide")
st.title("🚘 DETR — Self-Driving Object Detection Dashboard")

# ----------------------------
# CONFIGURATION
# ----------------------------
MODEL_PATH = "pavankalyan123456/selfdriving-detr"  # your Hugging Face model repo

# ----------------------------
# LOAD MODEL + PROCESSOR
# ----------------------------
@st.cache_resource
def load_model():
    st.info("⏳ Loading DETR model from Hugging Face Hub...")
    model = DetrForObjectDetection.from_pretrained(MODEL_PATH)
    processor = DetrImageProcessor.from_pretrained(MODEL_PATH)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    st.success(f"✅ Model loaded successfully on **{device.upper()}**")
    return model, processor, device

model, processor, device = load_model()

# ----------------------------
# OBJECT DETECTION FUNCTION
# ----------------------------
def detect_objects(image: Image.Image):
    inputs = processor(images=image, return_tensors="pt").to(device)
    outputs = model(**inputs)
    target_sizes = torch.tensor([image.size[::-1]]).to(device)
    results = processor.post_process_object_detection(
        outputs, target_sizes=target_sizes, threshold=0.6
    )[0]

    draw = ImageDraw.Draw(image)
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        box = [round(i, 2) for i in box.tolist()]
        draw.rectangle(box, outline="lime", width=3)
        draw.text(
            (box[0], box[1]),
            f"{model.config.id2label[label.item()]}: {round(score.item(), 2)}",
            fill="white",
        )
    return image

# =========================================================
# 🖼️ IMAGE UPLOAD SECTION
# =========================================================
st.header("🖼️ Upload an Image")
image_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if image_file is not None:
    img = Image.open(image_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_container_width=True)
    if st.button("🔍 Run Detection on Image"):
        with st.spinner("Detecting objects..."):
            output_img = detect_objects(img)
            st.image(output_img, caption="Detections", use_container_width=True)
            st.success("✅ Detection Complete!")

# =========================================================
# 🎥 VIDEO UPLOAD SECTION (SAFE FOR STREAMLIT CLOUD)
# =========================================================
st.header("🎞️ Upload a Video for Detection (Streamlit Cloud Safe Mode)")

video_file = st.file_uploader("Upload a video...", type=["mp4", "avi", "mov"])

if video_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(video_file.read())
    cap = cv2.VideoCapture(tfile.name)
    cv2.setNumThreads(0)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    st.info(f"Processing video... ({total_frames} frames, {fps} fps)")

    frame_skip = st.slider("Skip every N frames (higher = faster)", 1, 20, 5)
    stframe = st.empty()
    progress = st.progress(0)

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_skip == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image_pil = Image.fromarray(frame_rgb)

            inputs = processor(images=image_pil, return_tensors="pt").to(device)
            outputs = model(**inputs)
            target_sizes = torch.tensor([image_pil.size[::-1]]).to(device)
            results = processor.post_process_object_detection(
                outputs, target_sizes=target_sizes, threshold=0.7
            )[0]

            draw = ImageDraw.Draw(image_pil)
            for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
                box = [round(i, 2) for i in box.tolist()]
                draw.rectangle(box, outline="red", width=3)
                draw.text(
                    (box[0], box[1]),
                    f"{model.config.id2label[label.item()]} {round(score.item(), 2)}",
                    fill="white",
                )

            stframe.image(image_pil, caption=f"Frame {frame_count}", use_container_width=True)

        frame_count += 1
        if frame_count % 50 == 0:
            progress.progress(min(frame_count / total_frames, 1.0))

    cap.release()
    st.success("✅ Video processing complete!")

# =========================================================
# 📷 LIVE CAMERA CAPTURE SECTION
# =========================================================
st.header("📷 Live Camera Detection")

start_cam = st.button("🎬 Start Live Camera")

if start_cam:
    cap = cv2.VideoCapture(0)
    cv2.setNumThreads(0)
    stframe = st.empty()
    st.info("Press **Stop Live Camera** to end stream.")

    stop_button = st.button("🛑 Stop Live Camera")
    while cap.isOpened() and not stop_button:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(frame_rgb)

        inputs = processor(images=image_pil, return_tensors="pt").to(device)
        outputs = model(**inputs)
        target_sizes = torch.tensor([image_pil.size[::-1]]).to(device)
        results = processor.post_process_object_detection(
            outputs, target_sizes=target_sizes, threshold=0.6
        )[0]

        draw = ImageDraw.Draw(image_pil)
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            box = [round(i, 2) for i in box.tolist()]
            draw.rectangle(box, outline="cyan", width=3)
            draw.text(
                (box[0], box[1]),
                f"{model.config.id2label[label.item()]} {round(score.item(), 2)}",
                fill="white",
            )

        stframe.image(image_pil, caption="Live Feed", use_container_width=True)
        time.sleep(0.05)  # controls FPS

    cap.release()
    st.success("🛑 Live camera stopped.")

# =========================================================
# FOOTER
# =========================================================
st.caption("🚀 Built with Hugging Face DETR + Streamlit by Pavan Kalyan")
