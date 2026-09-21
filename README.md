# 👁️ Multi-Purpose Vision Chatbot

An interactive **Streamlit** chatbot that can analyze both **text and images**.

- Upload any image → a pre-trained **DETR** model (Facebook) detects objects
- Ask natural-language questions about what was detected
- Also supports general conversation

---

## Features

| Feature | Description |
|---------|-------------|
| **Object Detection** | Facebook DETR (ResNet-50) – detects 80+ COCO classes |
| **Annotated Image** | Bounding boxes + labels drawn on the uploaded image |
| **Conversational AI** | Answers questions about the detections (powered by Groq / Llama 3.1) |
| **Fallback Mode** | Works without any API key (basic detection answers) |
| **Chat History** | Full conversation memory during the session |
| **Adjustable Confidence** | Slider to control detection sensitivity |

---

## Demo Flow

1. Upload an image (sidebar)
2. See original + annotated image with bounding boxes
3. Ask questions such as:
   - “What objects did you find?”
   - “How many people are in the image?”
   - “What is the most confident detection?”
   - “Describe the scene”
4. Continue chatting

---

## Quick Start (Local)

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/vision-chatbot.git
cd vision-chatbot

# 2. Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run app.py
```

The app will open at `http://localhost:8501`.

---

## Optional: Smarter Answers with Groq (Free)

1. Go to [https://console.groq.com/keys](https://console.groq.com/keys)
2. Create a free API key
3. Paste it in the sidebar of the app

Without a key the bot still works — it just gives simpler answers based on the detection results.

---

## Deploy on Streamlit Community Cloud (Free)

1. Push this repository to GitHub
2. Go to [https://share.streamlit.io](https://share.streamlit.io)
3. Click **New app**
4. Select your repository, branch `main`, main file `app.py`
5. Click **Deploy**

Your public URL will look like:  
`https://vision-chatbot-xxxxx.streamlit.app`

> **Note:** The first run may take 30–60 seconds while the DETR model is downloaded.

---

## Project Structure

```
vision-chatbot/
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
├── README.md           # This file
└── .gitignore
```

---

## How it Works

1. **Image Upload** → PIL image is processed by DETR
2. **DETR** returns bounding boxes, class labels and confidence scores
3. Annotated image is displayed and a summary is generated
4. When the user asks a question, the detection results are injected into a prompt
5. The prompt is sent to Groq (Llama 3.1) or a simple rule-based fallback
6. Response is shown in the chat interface

---

## Technologies Used

- **Streamlit** – Web UI & chat interface
- **Hugging Face Transformers + DETR** – Object detection
- **Pillow** – Image processing & drawing
- **Groq API** (optional) – Fast conversational LLM
- **PyTorch** – Backend for the detection model

---

## License

MIT License – feel free to use and modify.
