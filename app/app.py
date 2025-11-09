# ===============================
# 🚗 DETR Self-Driving Object Detection App
# with Debug Info for Streamlit Cloud
# ===============================

import sys
import os
import torch
import streamlit as st
from PIL import Image, ImageDraw
import cv2
import tempfile
import threading

# --- DEBUG BLOCK (checks imports & versions) ---
st.set_page_config(page_title="🚗 DETR Self-Driving Object Detection", layout="wide")
st.title("🚘 DETR — Self-Driving Object Detection Dashboard")

st.write("Python:", sys.version)
try:
    import transformers
    st.write("torch:", torch.__version__, "transformers:", transformers.__version__)
except Exception as e:
    st.error(f"Import error for torch/transformers: {e}")
    raise

try:
    from transformers import DetrForObjectDetection, DetrImageProcessor
    st.success("✅ Detr classes import OK")
except Exception as e:
    st.error(f"❌ Detr import failed: {e}")
    raise
# --- END DEBUG BLOCK ---

# ========================
# CONFIGURATION
# ========================
MODEL_PATH = "pavankalyan123456/selfdriving-detr"   # from Hugging Face Hub

# ========================
# LOAD MODEL + PROCESSOR
# ========================
@st.cache_resource
def load_model():
    st.info("⏳ Loading DETR model from Hugging Face Hub...")
    model = DetrForObjectDetection.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True
    )
    processor = DetrImageProcessor.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    st.success(f"✅ Model loaded successfully on **{device.upper()}**")
    return model, processor, device

model, processor, device = load_model()

# ========================
# HELPER — RUN DETECTION
# ========================
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

# ========================
# IMAGE UPLOAD
# ========================
st.header("🖼️ Upload an Image")
image_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if image_file is not None:
    img = Image.open(image_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)
    if st.button("🔍 Run Detection on Image"):
        with st.spinner("Detecting objects..."):
            output_img = detect_objects(img)
            st.image(output_img, caption="Detections", use_column_width=True)
            st.success("✅ Detection Complete!")

# ========================
# LONG VIDEO UPLOAD — THREAD SAFE
# ========================
st.header("🎥 Upload a Long Video (Optimized for 10k+ frames)")

video_file = st.file_uploader("Upload a video (up to ~10 min)...", type=["mp4", "avi", "mov"])

if video_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(video_file.read())
    cv2.setNumThreads(1)
    cap = cv2.VideoCapture(tfile.name)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    frame_skip = st.slider(
        "Skip every N frames (higher = faster, lower = more accurate)", 1, 20, 5
    )

    st.info(f"Processing video... ({total_frames} frames, {fps} fps)")
    progress = st.progress(0)
    stframe = st.empty()
    status = st.empty()

    def process_video():
        frame_count = 0
        processed = 0
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_count % frame_skip == 0:
                    processed += 1
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

                    stframe.image(image_pil, caption=f"Frame {frame_count}", use_column_width=True)
                    torch.cuda.empty_cache()

                frame_count += 1
                if frame_count % 50 == 0:
                    progress.progress(min(frame_count / total_frames, 1.0))
                    status.text(f"Processed {frame_count}/{total_frames} frames...")

            cap.release()
            progress.progress(1.0)
            status.text("✅ Video processing complete!")

        except Exception as e:
            st.error(f"⚠️ Error: {str(e)}")
        finally:
            cap.release()

    thread = threading.Thread(target=process_video)
    thread.start()
    cap.release()
    st.success("✅ Video processing complete!")

st.caption("🚀 Built with Hugging Face DETR + Streamlit by Pavan Kalyan")
