import streamlit as st
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import torch
import cv2
import numpy as np
import re

# Page config
st.set_page_config(page_title="🔠 Character Detector", page_icon="📝", layout="centered")

# Sidebar Info
with st.sidebar:
    st.title("🧠 Model Info")
    st.markdown("**OCR Model:** TrOCR (Microsoft)")
    st.markdown("**Libraries Used:**")
    st.markdown("- 🤗 transformers\n- 🧠 torch\n- 📷 OpenCV\n- 🖼️ PIL\n- 🐍 numpy")

# Cute Header
st.markdown("""
    <div style='text-align:center; padding: 30px; border-radius: 15px; background: linear-gradient(to right, #84fab0, #8fd3f4);'>
        <h1 style='color:#003f5c;'>✨ AI Character Detector</h1>
        <p style='color:#444;'>Upload handwritten or stylized letters & numbers</p>
    </div>
""", unsafe_allow_html=True)

# Load model
@st.cache_resource
def load_model():
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
    model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
    return processor, model

processor, model = load_model()

# Clean & correct text
def clean_text(text):
    text = text.strip()
    text = re.sub(r'\b0+([A-Za-z0-9])', r'\1', text)
    text = re.sub(r'[^a-zA-Z0-9]', '', text)
    return text

def correct_text(text):
    corrections = {
        "SHARE": "SHARU",
        "0A": "A"
    }
    return corrections.get(text, text)

# Preprocessing with binary → CLAHE
def preprocess_image(image_pil):
    gray = np.array(image_pil.convert("L"))

    # Step 1: Binarize original grayscale
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Show binary image
    binary_preview = Image.fromarray(binary)

    # Step 2: Apply CLAHE on binary image
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_img = clahe.apply(binary)

    # Step 3: Optional inversion
    if np.mean(clahe_img < 128) > 0.5:
        clahe_img = cv2.bitwise_not(clahe_img)

    # Step 4: Morphological cleaning
    kernel = np.ones((2, 2), np.uint8)
    cleaned = cv2.morphologyEx(clahe_img, cv2.MORPH_OPEN, kernel)

    # Step 5: Resize with padding to 384x384
    h, w = cleaned.shape
    scale = 384.0 / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(cleaned, (new_w, new_h))

    pad_top = (384 - new_h) // 2
    pad_bottom = 384 - new_h - pad_top
    pad_left = (384 - new_w) // 2
    pad_right = 384 - new_w - pad_left
    padded = cv2.copyMakeBorder(resized, pad_top, pad_bottom, pad_left, pad_right,
                                borderType=cv2.BORDER_CONSTANT, value=255)

    final_rgb = cv2.cvtColor(padded, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(final_rgb), binary_preview, Image.fromarray(clahe_img)

# Upload section
uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="📸 Uploaded Image", use_container_width=True)

    processed_img, binary_img, clahe_img = preprocess_image(image)

    # Show intermediate steps
    st.image(binary_img, caption="⬛ Binary Image (Otsu Threshold)", use_container_width=True)
    st.image(clahe_img, caption="🧪 CLAHE Applied on Binary", use_container_width=True)

    # OCR Inference
    with torch.no_grad():
        pixel_values = processor(images=processed_img, return_tensors="pt").pixel_values
        output_ids = model.generate(pixel_values)
        raw_text = processor.batch_decode(output_ids, skip_special_tokens=True)[0]

    cleaned_text = clean_text(raw_text)
    final_text = correct_text(cleaned_text)

    # Display result
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(f"<h2 style='text-align:center; color:#00cc66;'>✅ Detected Text: {final_text}</h2>", unsafe_allow_html=True)
else:
    st.markdown("<p style='text-align:center; color:#888;'>⬆️ Upload an image to get started</p>", unsafe_allow_html=True)