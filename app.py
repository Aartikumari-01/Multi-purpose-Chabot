"""
Multi-Purpose Vision Chatbot
----------------------------
An interactive Streamlit chatbot that:
  • Accepts text and image inputs
  • Uses a pre-trained YOLOv8 model to detect objects in images
  • Answers questions about the detected objects (and general conversation)
"""

import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import io
from typing import List, Dict, Any, Optional
import os

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Vision Chatbot",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
YOLO_MODEL = "yolov8n.pt"  # nano version – fast & good accuracy

# ---------------------------------------------------------------------------
# Cached model loading
# ---------------------------------------------------------------------------
@st.cache_resource
def load_yolo_model():
    """Load YOLOv8 model (downloaded automatically on first run)."""
    model = YOLO(YOLO_MODEL)
    return model


def detect_objects(image: Image.Image, conf_threshold: float = 0.25) -> Dict[str, Any]:
    """
    Run object detection on a PIL image.
    Returns annotated image (RGB numpy), list of detections, and a summary string.
    """
    model = load_yolo_model()

    # Convert PIL → OpenCV BGR
    img_array = np.array(image.convert("RGB"))
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    results = model.predict(img_bgr, conf=conf_threshold, verbose=False)
    result = results[0]

    # Annotated image (BGR → RGB)
    annotated_bgr = result.plot()
    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

    detections = []
    for box in result.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        label = result.names[cls_id]
        detections.append({
            "label": label,
            "confidence": round(conf, 3),
            "class_id": cls_id,
        })

    # Build human-readable summary
    if not detections:
        summary = "I couldn't detect any objects in this image with the current confidence threshold."
    else:
        # Count occurrences
        counts: Dict[str, int] = {}
        for d in detections:
            counts[d["label"]] = counts.get(d["label"], 0) + 1

        parts = [f"{count} {label}{'s' if count > 1 else ''}" for label, count in counts.items()]
        summary = "I detected the following objects: " + ", ".join(parts) + "."

    return {
        "annotated_image": annotated_rgb,
        "detections": detections,
        "summary": summary,
        "counts": counts if detections else {},
    }


def build_context_prompt(detections: List[Dict], user_question: str) -> str:
    """Create a context-aware prompt for the conversational model."""
    if not detections:
        return (
            "The user uploaded an image but no objects were detected. "
            f"User question: {user_question}"
        )

    counts = {}
    for d in detections:
        counts[d["label"]] = counts.get(d["label"], 0) + 1

    objects_str = ", ".join([f"{v} {k}" for k, v in counts.items()])
    details = "; ".join([f"{d['label']} ({d['confidence']*100:.0f}%)" for d in detections])

    prompt = f"""You are a helpful vision assistant. The user has uploaded an image and the object detection model found:

Objects detected: {objects_str}
Detailed detections: {details}

Answer the user's question based on these detection results. Be concise, friendly and accurate.
If the question cannot be answered from the detections alone, say so politely.

User question: {user_question}
"""
    return prompt


def get_llm_response(prompt: str, api_key: Optional[str] = None) -> str:
    """
    Get a response from Groq (recommended) or fall back to a simple rule-based reply.
    """
    if api_key:
        try:
            from groq import Groq
            client = Groq(api_key=api_key)
            chat_completion = client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a friendly and helpful vision chatbot. "
                            "You answer questions about objects detected in images. "
                            "Keep answers short and clear."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                model="llama-3.1-8b-instant",
                temperature=0.4,
                max_tokens=300,
            )
            return chat_completion.choices[0].message.content
        except Exception as e:
            return f"(LLM error: {e})\n\nFalling back to basic response based on detections."

    # Fallback – simple rule-based answers when no API key
    return (
        "I can see the detected objects listed above. "
        "For richer conversational answers, please add a free Groq API key in the sidebar.\n\n"
        "You can still ask questions like:\n"
        "• What objects did you find?\n"
        "• How many people / cars / dogs are there?\n"
        "• What is the most confident detection?"
    )


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "Hello! 👋 I'm your Vision Chatbot.\n\n"
                "You can:\n"
                "1. Upload an image and I'll detect the objects in it\n"
                "2. Ask me questions about the image (e.g. \"How many people are there?\")\n"
                "3. Chat with me about anything else\n\n"
                "Upload an image from the sidebar to get started!"
            ),
        }
    ]

if "last_detections" not in st.session_state:
    st.session_state.last_detections = []

if "last_summary" not in st.session_state:
    st.session_state.last_summary = ""

if "annotated_image" not in st.session_state:
    st.session_state.annotated_image = None

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("👁️ Vision Chatbot")
    st.markdown("---")

    st.subheader("📷 Upload Image")
    uploaded_file = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png", "webp", "bmp"],
        help="Supported formats: JPG, PNG, WebP, BMP",
    )

    conf_threshold = st.slider(
        "Detection confidence",
        min_value=0.1,
        max_value=0.9,
        value=0.35,
        step=0.05,
        help="Higher value = fewer but more confident detections",
    )

    st.markdown("---")
    st.subheader("🔑 Optional: Groq API Key")
    st.markdown(
        "For smarter conversational answers, get a **free** key at "
        "[console.groq.com](https://console.groq.com/keys)"
    )
    groq_key = st.text_input(
        "Groq API Key",
        type="password",
        placeholder="gsk_...",
        help="Leave empty to use basic detection-only mode",
    )

    st.markdown("---")
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Chat cleared! Upload a new image or ask me something.",
            }
        ]
        st.session_state.last_detections = []
        st.session_state.last_summary = ""
        st.session_state.annotated_image = None
        st.rerun()

    st.markdown("---")
    st.caption("Powered by YOLOv8 + Streamlit")

# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------
st.title("👁️ Multi-Purpose Vision Chatbot")
st.caption("Upload an image → detect objects → ask questions about what I see")

# Process uploaded image
if uploaded_file is not None:
    image = Image.open(uploaded_file)

    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Original Image", use_container_width=True)

    with st.spinner("Detecting objects with YOLOv8..."):
        result = detect_objects(image, conf_threshold=conf_threshold)

    st.session_state.annotated_image = result["annotated_image"]
    st.session_state.last_detections = result["detections"]
    st.session_state.last_summary = result["summary"]

    with col2:
        st.image(result["annotated_image"], caption="Detected Objects", use_container_width=True)

    # Show detection summary
    st.success(result["summary"])

    if result["detections"]:
        with st.expander("📊 Detection details"):
            import pandas as pd
            df = pd.DataFrame(result["detections"])
            st.dataframe(df, use_container_width=True)

# ---------------------------------------------------------------------------
# Chat interface
# ---------------------------------------------------------------------------
st.markdown("### 💬 Chat")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
if prompt := st.chat_input("Ask about the image or chat with me..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # If we have detections, make the prompt context-aware
            if st.session_state.last_detections:
                context_prompt = build_context_prompt(
                    st.session_state.last_detections, prompt
                )
            else:
                context_prompt = (
                    "No image has been analyzed yet. "
                    f"User said: {prompt}\n"
                    "Reply helpfully. If they seem to be asking about an image, "
                    "remind them to upload one first."
                )

            response = get_llm_response(context_prompt, api_key=groq_key or None)

            # Also append the detection summary if relevant
            if st.session_state.last_summary and any(
                word in prompt.lower()
                for word in ["what", "detect", "see", "object", "how many", "count"]
            ):
                response = f"{st.session_state.last_summary}\n\n{response}"

            st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
