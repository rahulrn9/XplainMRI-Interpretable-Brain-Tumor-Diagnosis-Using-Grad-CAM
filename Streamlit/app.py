import streamlit as st
import numpy as np
import tensorflow as tf
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from datetime import datetime
import gspread
from oauth2client.service_account import ServiceAccountCredentials

MODEL_PATH = "models/mobilenetv2.h5"
CRED_PATH = "streamlit_app/creds.json"
SHEET_NAME = "BrainTumorPredictions"
CLASS_LABELS = ['glioma', 'meningioma', 'no_tumor', 'pituitary']
CONFIDENCE_THRESHOLD = 70

@st.cache_resource
def setup_sheet():
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    creds = ServiceAccountCredentials.from_json_keyfile_name(CRED_PATH, scope)
    client = gspread.authorize(creds)
    try:
        sheet = client.open(SHEET_NAME).sheet1
    except gspread.SpreadsheetNotFound:
        sheet = client.create(SHEET_NAME).sheet1
        sheet.append_row(["Timestamp", "Filename", "Prediction", "Confidence (%)", "Feedback"])
    return sheet

def get_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(last_conv_layer_name).output, model.output])
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_output = predictions[:, pred_index]
    grads = tape.gradient(class_output, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def overlay_heatmap(original_img, heatmap):
    heatmap = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    color_map = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(original_img, 0.6, color_map, 0.4, 0)
    return superimposed_img

def preprocess_image(uploaded_file):
    image = Image.open(uploaded_file).convert("RGB")
    image = image.resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array, np.array(image)

def log_to_google_sheets(sheet, filename, predicted_class, confidence, feedback=""):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    sheet.append_row([now, filename, predicted_class, f"{confidence:.2f}", feedback])

st.title("🧠 Brain Tumor MRI Classifier with Grad-CAM & Feedback 🔬")
uploaded_file = st.file_uploader("Upload a Brain MRI Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    with st.spinner("Analyzing image..."):
        model = tf.keras.models.load_model(MODEL_PATH)
        sheet = setup_sheet()

        img_array, original_pil = preprocess_image(uploaded_file)
        preds = model.predict(img_array)
        pred_index = np.argmax(preds[0])
        predicted_label = CLASS_LABELS[pred_index]
        confidence = preds[0][pred_index] * 100

        st.image(original_pil, caption="Uploaded MRI Image", use_column_width=True)
        if confidence < CONFIDENCE_THRESHOLD:
            st.warning(f"⚠️ Low confidence ({confidence:.2f}%) – Suggest human review.")
        else:
            st.success(f"🧠 Predicted: **{predicted_label.upper()}** ({confidence:.2f}%)")

        heatmap = get_gradcam_heatmap(img_array, model, last_conv_layer_name="Conv_1")
        original_bgr = cv2.cvtColor(np.array(original_pil), cv2.COLOR_RGB2BGR)
        cam_result = overlay_heatmap(original_bgr, heatmap)
        cam_rgb = cv2.cvtColor(cam_result, cv2.COLOR_BGR2RGB)

        st.markdown("### 🔥 Grad-CAM Heatmap")
        st.image(cam_rgb, use_column_width=True)

        filename = uploaded_file.name
        log_to_sheet = st.checkbox("✅ Log this prediction to Google Sheets", value=True)
        if log_to_sheet:
            log_to_google_sheets(sheet, filename, predicted_label, confidence)

        st.markdown("### 📝 Was this prediction correct?")
        feedback = st.radio("Your Feedback", ("Yes", "No"))
        if st.button("Submit Feedback"):
            log_to_google_sheets(sheet, filename, predicted_label, confidence, feedback)
            st.success("✅ Feedback submitted and logged!")
