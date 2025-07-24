# 🧠 NeuroVision – Brain Tumor MRI Classifier with Explainability and Feedback

This project is a full-stack AI-powered application for classifying brain tumor types from MRI images using deep learning. It includes explainable AI (Grad-CAM), confidence scoring, real-time feedback capture, and Google Sheets logging — all wrapped into an interactive Streamlit web app.

---

## ✅ Key Features

- 🎯 **Multi-class Brain Tumor Classification** using MobileNetV2
- 🔥 **Grad-CAM Heatmap Visualization** to explain what the model sees
- ⚠️ **Confidence Threshold Alert** to warn for uncertain predictions
- 📋 **Google Sheets Logging** of predictions and user feedback
- 📝 **Feedback Mechanism** to tag predictions as Correct/Incorrect
- 📊 Streamlit-based **interactive dashboard** for real-time inference

---

## 🗂️ Dataset Structure

Organize your MRI dataset like this:

data/BrainTumorDataset/
├── glioma/
├── meningioma/
├── no_tumor/
└── pituitary/

yaml
Copy
Edit

Each folder contains respective MRI images in `.jpg` or `.png`.

---

## 🏗️ Project Structure

BrainTumorClassifier/
├── models/ # Trained model (.h5)
│ └── mobilenetv2.h5
├── data/
│ └── BrainTumorDataset/ # Your image dataset (4 folders)
├── streamlit_app/
│ ├── app.py # Streamlit UI + Grad-CAM + feedback
│ └── creds.json # Google Sheets API credentials
├── train_sample_model.py # Script to train MobileNetV2
├── requirements.txt # Dependencies
├── .gitignore
└── README.md # You're reading this!

yaml
Copy
Edit

---

## 🧪 Model Training

To train a MobileNetV2 model on your dataset:
```bash
python train_sample_model.py
Model is saved to models/mobilenetv2.h5

🚀 Running the App
Install dependencies:

bash
Copy
Edit
pip install -r requirements.txt
Start the Streamlit app:

bash
Copy
Edit
streamlit run streamlit_app/app.py
🔐 Google Sheets Integration
Go to Google Cloud Console → Create a new project

Enable:

Google Sheets API

Google Drive API

Create a Service Account and download the JSON key

Rename to creds.json and place inside streamlit_app/

Share your Google Sheet with the service account email

🧠 Grad-CAM Explainability
Grad-CAM overlays a heatmap on the original MRI image to show which region influenced the prediction the most. It helps doctors trust the model’s output.

📈 Confidence Threshold
If model confidence is below 70%, the app shows a warning:

⚠️ Low confidence – Suggest human review

You can adjust this in app.py:

python
Copy
Edit
CONFIDENCE_THRESHOLD = 70
✍️ Feedback Mechanism
Users can submit:

✅ "Correct"

❌ "Incorrect"

This is logged to Google Sheets for future improvements or audit.

🧠 Model Architecture (MobileNetV2)
Input: 224x224x3 MRI image

Base: Pretrained MobileNetV2 (ImageNet)

Top:

GlobalAveragePooling

Dropout

Dense (Softmax)

Fine-tuned on 4 tumor classes using transfer learning.

📄 Dependencies
txt
Copy
Edit
tensorflow
streamlit
opencv-python
matplotlib
numpy
Pillow
gspread
oauth2client
Install with:

bash
Copy
Edit
pip install -r requirements.txt
📸 Sample Screenshot
(Insert a screenshot of Streamlit app here with Grad-CAM overlay)

📚 Acknowledgments
Dataset: Brain MRI Multi-class Tumor Dataset (Kaggle)

Model Base: MobileNetV2 – TensorFlow Keras

Visualization: Grad-CAM implementation based on tf.keras

🤖 Author
Rahul Naduvinamani
AI & Data Science | ML Systems | Backend & Infra | GitHub

