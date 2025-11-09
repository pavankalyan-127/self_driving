# =========================================================
# 🚗 DETR Self-Driving Object Detection (Final with Live Camera)
# =========================================================

import os
import time
import uuid
import torch
import streamlit as st
from PIL import Image, ImageDraw
from transformers import DetrForObjectDetection, DetrImageProcessor
import cv2
import tempfile
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import av

# =========================================================
# STREAMLIT CONFIG
# =========================================================
st.set_page_config(page_title="🚗 Self-Driving Object Detection", layout="wide")
st.title("🚘 DETR — Self-Driving Object Detection Dashboard")

# =========================================================
# MODEL CONFIGURATION
# =========================================================
MODEL_PATH = "pavankalyan123456/selfdriving-detr"  # your Hugging Face repo

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

# =========================================================
# DETECTION FUNCTION
# =========================================================
def detect_objects(image: Image.Image):
    """Run DETR object detection and draw boxes."""
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
# 📸 MODE SELECTION
# =========================================================
st.header("📸 Choose Input Source")
option = st.radio(
    "Select Mode:",
    ["Capture from Camera", "Upload Image", "Upload Video (MP4)", "Live Camera Detection"],
    index=1
)

# =========================================================
# 1️⃣ CAPTURE FROM CAMERA
# =========================================================
if option == "Capture from Camera":
    st.info("Use your webcam or phone camera to capture an image.")
    camera_image = st.camera_input("Take a photo", key="selfdriving_cam")

    if camera_image is not None:
        try:
            image = Image.open(camera_image).convert("RGB")
            st.image(image, caption="Captured Frame", use_container_width=True)
            with st.spinner("Detecting objects..."):
                output_img = detect_objects(image)
                st.image(output_img, caption="Detections", use_container_width=True)
                st.success("✅ Detection Complete!")
        except Exception as e:
            st.error(f"❌ Error processing camera image: {e}")

# =========================================================
# 2️⃣ UPLOAD IMAGE
# =========================================================
elif option == "Upload Image":
    image_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"], key="upload_image_2")
    if image_file is not None:
        try:
            img = Image.open(image_file).convert("RGB")
            st.image(img, caption="Uploaded Image", use_container_width=True)
            with st.spinner("Detecting objects..."):
                output_img = detect_objects(img)
                st.image(output_img, caption="Detections", use_container_width=True)
                st.success("✅ Detection Complete!")
        except Exception as e:
            st.error(f"❌ Error processing uploaded image: {e}")

# =========================================================
# 3️⃣ VIDEO UPLOAD (FRAME-BY-FRAME DETECTION)
# =========================================================
elif option == "Upload Video (MP4)":
    video_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"], key="upload_video_2")

    if video_file is not None:
        temp_uuid = str(uuid.uuid4())[:8]
        temp_path = f"temp_selfdriving_{temp_uuid}.mp4"
        try:
            with open(temp_path, "wb") as f:
                f.write(video_file.read())

            cap = cv2.VideoCapture(temp_path)
            if not cap.isOpened():
                st.error("❌ Could not open uploaded video.")
            else:
                st.info("🎥 Processing video... showing detections live.")
                stframe = st.empty()
                frame_count = 0
                success_frames = 0
                start_time = time.time()

                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    frame_count += 1
                    if frame_count % 3 != 0:
                        continue

                    try:
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        image_pil = Image.fromarray(frame_rgb)
                        output_img = detect_objects(image_pil)

                        display_frame = np.array(output_img)
                        stframe.image(display_frame, caption=f"Frame {frame_count}", use_container_width=True)
                        success_frames += 1
                        time.sleep(0.05)
                    except Exception as frame_err:
                        st.warning(f"⚠️ Error at frame {frame_count}: {frame_err}")
                        continue

                cap.release()
                fps = success_frames / (time.time() - start_time)
                st.success(f"✅ Finished processing {success_frames} frames. Avg FPS: {fps:.2f}")

        except Exception as e:
            st.error(f"❌ Error reading or processing uploaded video: {e}")
        finally:
            try:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            except Exception:
                pass

# =========================================================
# 4️⃣ TRUE LIVE CAMERA DETECTION (Continuous Stream)
# =========================================================
elif option == "Live Camera Detection":
    st.info("🎥 Starting live detection — allow camera access in your browser.")

    class LiveVideoProcessor(VideoProcessorBase):
        def __init__(self):
            self.model = model
            self.processor = processor

        def recv(self, frame):
            img = frame.to_ndarray(format="bgr24")
            image_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

            # DETR inference
            inputs = self.processor(images=image_pil, return_tensors="pt").to(device)
            outputs = self.model(**inputs)
            target_sizes = torch.tensor([image_pil.size[::-1]]).to(device)
            results = self.processor.post_process_object_detection(
                outputs, target_sizes=target_sizes, threshold=0.6
            )[0]

            # Draw boxes
            draw = ImageDraw.Draw(image_pil)
            for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
                box = [round(i, 2) for i in box.tolist()]
                draw.rectangle(box, outline="red", width=3)
                draw.text(
                    (box[0], box[1]),
                    f"{self.model.config.id2label[label.item()]} {round(score.item(), 2)}",
                    fill="white",
                )

            # Return processed frame
            return av.VideoFrame.from_ndarray(cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR), format="bgr24")

    webrtc_streamer(
        key="live_selfdriving",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=LiveVideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
    )

# =========================================================
# FOOTER
# =========================================================
st.markdown("---")
st.caption("🚀 Built with Hugging Face DETR + Streamlit + WebRTC by Pavan Kalyan")
