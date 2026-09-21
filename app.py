"""
Multi-Purpose Vision Chatbot
----------------------------
An interactive Streamlit chatbot that:
  • Accepts text and image inputs
  • Uses a pre-trained DETR model (Hugging Face) to detect objects in images
  • Answers questions about the detected objects (and general conversation)
"""

import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from typing import List, Dict, Any, Optional
import io
import torch

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
# Model loading (cached)
# ---------------------------------------------------------------------------
@st.cache_resource
def load_detection_model():
    """Load DETR object detection model from Hugging Face."""
    from transformers import DetrImageProcessor, DetrForObjectDetection

    processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50", revision="no_timm")
    model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50", revision="no_timm")
    model.eval()
    return processor, model


def detect_objects(image: Image.Image, conf_threshold: float = 0.7) -> Dict[str, Any]:
    """
    Run object detection using DETR.
    Returns annotated image, list of detections, and a summary string.
    """
    processor, model = load_detection_model()

    # Prepare image
    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    # Convert outputs to scores, labels, boxes
    target_sizes = torch.tensor([image.size[::-1]])  # (height, width)
    results = processor.post_process_object_detection(
        outputs, target_sizes=target_sizes, threshold=conf_threshold
    )[0]

    detections = []
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)

    # Try to load a font, fall back to default
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
    except Exception:
        font = ImageFont.load_default()

    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        label_name = model.config.id2label[label.item()]
        confidence = round(score.item(), 3)

        detections.append({
            "label": label_name,
            "confidence": confidence,
            "class_id": label.item(),
        })

        # Draw bounding box
        box = [round(i, 2) for i in box.tolist()]
        x0, y0, x1, y1 = box
        draw.rectangle([x0, y0, x1, y1], outline="red", width=3)

        # Draw label background + text
        text = f"{label_name} {confidence:.0%}"
        text_bbox = draw.textbbox((x0, y0), text, font=font)
        draw.rectangle(text_bbox, fill="red")
        draw.text((x0, y0), text, fill="white", font=font)

    # Build summary
    if not detections:
        summary = "I couldn't detect any objects in this image with the current confidence threshold."
        counts = {}
    else:
        counts = {}
        for d in detections:
            counts[d["label"]] = counts.get(d["label"], 0) + 1

        parts = [f"{count} {label}{'s' if count > 1 else ''}" for label, count in counts.items()]
        summary = "I detected the following objects: " + ", ".join(parts) + "."

    return {
        "annotated_image": annotated,
        "detections": detections,
        "summary": summary,
        "counts": counts,
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
        min_value=0.3,
        max_value=0.9,
        value=0.7,
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
    st.caption("Powered by DETR (Facebook) + Streamlit")

# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------
st.title("👁️ Multi-Purpose Vision Chatbot")
st.caption("Upload an image → detect objects → ask questions about what I see")

# Process uploaded image
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Original Image", use_container_width=True)

    with st.spinner("Detecting objects with DETR model... (first run may take ~30 seconds)"):
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
