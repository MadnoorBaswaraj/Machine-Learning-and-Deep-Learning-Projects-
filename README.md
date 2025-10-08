Multi-Disease Prediction System using Deep Learning and Streamlit
📖 Overview

The Multi-Disease Prediction System is an AI-based healthcare web application that can predict multiple diseases using medical images and lab report values.
It integrates several trained deep learning (.h5) models into a unified Streamlit interface to assist patients and healthcare professionals in early diagnosis without direct hospital visits.

This system is especially useful during pandemics or remote healthcare scenarios, allowing users to upload reports or images and get disease predictions instantly.

⚙️ Features
🧠 Brain Tumor Detection

Classifies MRI brain images into glioma, meningioma, pituitary tumor, or no tumor.

🩺 Skin Cancer Detection

Detects whether a skin lesion is benign or malignant from uploaded images.

🎗️ Breast Cancer Prediction

Predicts the likelihood of breast cancer based on patient’s lab report values.

❤️ Heart Disease Prediction

Detects heart disease risk using patient health indicators such as cholesterol, blood pressure, etc.

💉 Diabetes Prediction

Predicts whether a patient is diabetic or non-diabetic based on clinical values.

🧍‍♂️ Parkinson’s Disease Detection

Identifies Parkinson’s disease using medical input parameters.

🦴 Bone Cancer Detection

Analyzes bone scan or X-ray images to classify tumors as benign or malignant.

🏥 Smart Recommendation System

If a patient’s result is positive, the app displays:

Precautionary measures

List of top hospitals in Hyderabad for further treatment

🧠 Tech Stack
Category	Technologies
Frontend / Deployment	Streamlit
Backend	Python
AI / ML Libraries	TensorFlow, Keras, Scikit-learn
Data Processing	NumPy, Pandas, OpenCV
Visualization	Matplotlib, Seaborn
IDE / Tools	Jupyter Notebook, VS Code, PyCharm


🚀 Installation and Setup
Clone the repository

git clone https://github.com/MadnoorBaswaraj/Machine-Learning-and-Deep-Learning-Projects-.git
cd Machine-Learning-and-Deep-Learning-Projects-

Create a virtual environment

python -m venv venv
source venv/bin/activate   # for macOS/Linux
venv\Scripts\activate      # for Windows
cd multi-disease-prediction



Install the dependencies

pip install -r requirements.txt


Run the Streamlit app

streamlit run app.py


🧩 How It Works

Upload Image or Enter Values:
The user either uploads an image (MRI, X-ray, or skin image) or enters numerical lab test values.

Model Prediction:
The respective deep learning model processes the input and predicts whether the disease is positive or negative.

Precautions & Hospital Suggestions:
If a disease is detected, the app provides safety tips and the nearest hospital recommendations in Hyderabad.

🩸 Use Case

During the pandemic, patients often faced challenges visiting hospitals for regular checkups.
This AI-driven solution allows remote diagnosis and early detection of diseases, saving time and enabling preventive healthcare.



📈 Future Enhancements

Integrate chatbot assistance for medical guidance.

Add real-time disease tracking and report generation.

Expand the hospital recommendation system to other cities in India.

Incorporate voice-based input for elderly patients.

👨‍💻 Author

Madnoor Baswaraj
B.Tech Student | AI & ML Enthusiast
💼 Passionate about Deep Learning, Computer Vision, and AI for Healthcare
