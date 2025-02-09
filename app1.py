import os
import pickle
import streamlit as st
from streamlit_option_menu import option_menu
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
#import torch
from PIL import Image
#from torchvision import models, transforms
from breast_cancer_web_app import main




# Set page configuration
st.set_page_config(page_title="Health Assistant",
                   layout="wide",
                   page_icon="🧑‍⚕️")

    
# getting the working directory of the main.py
working_dir = os.path.dirname(os.path.abspath(__file__))

# loading the saved models

diabetes_model = pickle.load(open(f'{working_dir}\\saved_models\\diabetes_model.sav', 'rb'))

heart_disease_model = pickle.load(open(f'{working_dir}\\saved_models\\heart_disease3_model.sav', 'rb'))

parkinsons_model = pickle.load(open(f'{working_dir}\\saved_models\\parkinsons_model.sav', 'rb'))

brain_tumor_model = load_model(f"{working_dir}\\saved_models\\brain_tumor_model.h5")

#breast_cancer_model = pickle.load(open(f'{working_dir}\\saved_models\\breast_cancer (1).sav', 'rb'))

bone_fracture_model = load_model(f'{working_dir}\\saved_models\\bone_fracture_model.h5')

skin_cancer_model = load_model(f'{working_dir}\\saved_models\\skin_model.h5')

#pneumonia_model = os.path.join(working_dir, "saved_models", "pretrained_vit_state_dict.pth")




# sidebar for navigation
with st.sidebar:
    selected = option_menu('Multiple Disease Prediction System',

                           ['Diabetes Prediction',
                            'Heart Disease Prediction',
                            'Parkinsons Prediction',
                            'Brain Tumor Prediction',
                            'Breast Cancer Prediction',
                            'Bone fracture Prediction',
                            'Skin Cancer Prediction',
                            'Chatbot'],
                           menu_icon='hospital-fill',
                           #icons=['activity', 'heart', 'person','','',''],
                           default_index=0)
    
    #selected=option_menu('chatbot',['chatbot'],menu_icon='hospital-fill',icons=['bot'])


# Diabetes Prediction Page
if selected == 'Diabetes Prediction':

    # page title
    st.title('Diabetes Prediction System')

    # getting the input data from the user
    col1, col2, col3 = st.columns(3)

    with col1:
        Pregnancies = st.text_input('Number of Pregnancies')

    with col2:
        Glucose = st.text_input('Glucose Level')

    with col3:
        BloodPressure = st.text_input('Blood Pressure value')

    with col1:
        SkinThickness = st.text_input('Skin Thickness value')

    with col2:
        Insulin = st.text_input('Insulin Level')

    with col3:
        BMI = st.text_input('BMI value')

    with col1:
        DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value')

    with col2:
        Age = st.text_input('Age of the Person')


    # code for Prediction
    diab_diagnosis = ''

    # creating a button for Prediction

    if st.button('Diabetes Test Result'):

        user_input = [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin,
                      BMI, DiabetesPedigreeFunction, Age]

        user_input = [float(x) for x in user_input]

        diab_prediction = diabetes_model.predict([user_input])

        if diab_prediction[0] == 1:
            diab_diagnosis = 'The person is diabetic'
        else:
            diab_diagnosis = 'The person is not diabetic'
        if diab_prediction[0] == 1:
            st.write("Precautions:")
            st.write("1. Avoid smoking.")
            st.write("2. Eat healthy food.")
            st.write("3. Exercise regularly.")
            st.write("4. Maintain a healthy weight.")
            st.write("5. Stay hydrated.")
            st.write("1.food:")
            st.write("2.Broccoli")
            st.write("3.Garlic")            
            st.write("4.Green tea")
            st.write("5.Lentils")
            st.write("6.lifestyle:")
            st.write("7.Maintain a healthy weight.")
            st.write("8.Regular physical activity.")
            st.write("9.Stress management.")
        else:
            st.write("Be Happy! You are healthy. Keep it up!")

    st.success(diab_diagnosis)

# Heart Disease Prediction Page
if selected == 'Heart Disease Prediction':

    # page title
    st.title('Heart Disease Prediction System')

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.text_input('Age')

    with col2:
        sex = st.text_input('Sex M-1 F-0')

    with col3:
        cp = st.text_input('Chest Pain types')

    with col1:
        trestbps = st.text_input('Resting Blood Pressure')

    with col2:
        chol = st.text_input('Serum Cholestoral in mg/dl')

    with col3:
        fbs = st.text_input('Fasting Blood Sugar > 120 mg/dl')

    with col1:
        restecg = st.text_input('Resting Electrocardiographic results')

    with col2:
        thalach = st.text_input('Maximum Heart Rate achieved')

    with col3:
        exang = st.text_input('Exercise Induced Angina')

    with col1:
        oldpeak = st.text_input('ST depression induced by exercise')

    with col2:
        slope = st.text_input('Slope of the peak exercise ST segment')

    with col3:
        ca = st.text_input('Major vessels colored by flourosopy')

    with col1:
        thal = st.text_input('thal: 0 = normal; 1 = fixed defect; 2 = reversable defect')

    # code for Prediction
    heart_diagnosis = ''

    # creating a button for Prediction

    if st.button('Heart Disease Test Result'):

        user_input = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]

        user_input = [float(x) for x in user_input]

        heart_prediction = heart_disease_model.predict([user_input])

        if heart_prediction[0] == 1:
            heart_diagnosis = 'The person is having heart disease'
        else:
            heart_diagnosis = 'The person does not have any heart disease'

        if heart_prediction[0] == 1:
            st.write("Precautions:")
            st.write("1. Avoid smoking.")
            st.write("2. Eat healthy food.")
            st.write("3. Exercise regularly.")
            st.write("4. Maintain a healthy weight.")
            st.write("5. Stay hydrated.")
            st.write("food:")
            st.write("Broccoli")
            st.write("Garlic")            
            st.write("Green tea")
            st.write("Lentils")
            st.write("lifestyle:")
            st.write("Maintain a healthy weight.")
            st.write("Regular physical activity.")
            st.write("Stress management.")

        else:
            st.write("Be Happy! You are healthy. Keep it up!")

    st.success(heart_diagnosis)

# Parkinson's Prediction Page
if selected == "Parkinsons Prediction":

    # page title
    st.title("Parkinson's Disease Prediction using ML")

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        fo = st.text_input('MDVP:Fo(Hz)')

    with col2:
        fhi = st.text_input('MDVP:Fhi(Hz)')

    with col3:
        flo = st.text_input('MDVP:Flo(Hz)')

    with col4:
        Jitter_percent = st.text_input('MDVP:Jitter(%)')

    with col5:
        Jitter_Abs = st.text_input('MDVP:Jitter(Abs)')

    with col1:
        RAP = st.text_input('MDVP:RAP')

    with col2:
        PPQ = st.text_input('MDVP:PPQ')

    with col3:
        DDP = st.text_input('Jitter:DDP')

    with col4:
        Shimmer = st.text_input('MDVP:Shimmer')

    with col5:
        Shimmer_dB = st.text_input('MDVP:Shimmer(dB)')

    with col1:
        APQ3 = st.text_input('Shimmer:APQ3')

    with col2:
        APQ5 = st.text_input('Shimmer:APQ5')

    with col3:
        APQ = st.text_input('MDVP:APQ')

    with col4:
        DDA = st.text_input('Shimmer:DDA')

    with col5:
        NHR = st.text_input('NHR')

    with col1:
        HNR = st.text_input('HNR')

    with col2:
        RPDE = st.text_input('RPDE')

    with col3:
        DFA = st.text_input('DFA')

    with col4:
        spread1 = st.text_input('spread1')

    with col5:
        spread2 = st.text_input('spread2')

    with col1:
        D2 = st.text_input('D2')

    with col2:
        PPE = st.text_input('PPE')

    # code for Prediction
    parkinsons_diagnosis = ''

    # creating a button for Prediction    
    if st.button("Parkinson's Test Result"):

        user_input = [fo, fhi, flo, Jitter_percent, Jitter_Abs,
                      RAP, PPQ, DDP,Shimmer, Shimmer_dB, APQ3, APQ5,
                      APQ, DDA, NHR, HNR, RPDE, DFA, spread1, spread2, D2, PPE]

        user_input = [float(x) for x in user_input]

        parkinsons_prediction = parkinsons_model.predict([user_input])

        if parkinsons_prediction[0] == 1:
            parkinsons_diagnosis = "The person has Parkinson's disease"
        else:
            parkinsons_diagnosis = "The person does not have Parkinson's disease"
        if parkinsons_prediction[0] == 1:
            st.write("Precautions:")
            st.write("1. Avoid alcohol and caffeine.")
            st.write("2. Avoid stress and anxiety.")
            st.write("3. Get regular physical activity.")        
            st.write("4. Eat a healthy diet.")
            st.write("5. Get enough sleep.")
            st.write("food:")
            st.write("Broccoli")
            st.write("Garlic")            
            st.write("Green tea")
            st.write("Lentils")
            st.write("Whole grains")
            st.write("Berries")
            st.write("lifestyle:")
            st.write("Maintain a healthy weight.")
            st.write("Regular physical activity.")
            st.write("Stress management.")
            st.write("Get regular medical checkups.")
            st.write("Visit a doctor if you have any concerns about your health.")
        else:
            st.write("Be Happy! You are healthy. Keep it up!")

    st.success(parkinsons_diagnosis)





if selected == "Brain Tumor Prediction":
    
    st.write("Please upload an Xray  for prediction.")

    uploaded_file = st.file_uploader("Upload a Brain MRI Image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        # Display the uploaded image
        image = load_img(uploaded_file, target_size=(256, 256))  # Adjust size as per your model input
        st.image(image, caption="Uploaded Image", use_column_width=True)
    
        # Preprocess the image
        img_array = img_to_array(image) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Predict using the model
        prediction = brain_tumor_model.predict(img_array)

    # Interpret the prediction
        class_labels = ['Glioma', 'Meningioma',  'No Tumor','Pituitary Tumor']
        predicted_class = np.argmax(prediction)
        confidence = np.max(prediction)

        st.write(f"Predicted class: {class_labels[predicted_class]}")
        st.write(f"Confidence: {confidence:.2f}")
        if class_labels[predicted_class] == 'Glioma':
            st.write("Precautions:")
            st.write("1. Avoid smoking.")      
            st.write("2. Eat healthy food.")
            st.write("3. Avoid alcohol consumption.")
            st.write("4. Regular physical activity.")
            st.write("5. Maintain a healthy weight.")
        elif class_labels[predicted_class] == 'Meningioma':
            st.write("Precautions:")            
            st.write("1. Avoid smoking.")            
            st.write("2. Eat healthy food.")            
            st.write("3. Avoid alcohol consumption.")            
            st.write("4. Regular physical activity.")            
            st.write("5. Maintain a healthy weight.")            
        elif class_labels[predicted_class] == 'No Tumor':
            st.write("Precautions:")            
            st.write("1. Avoid smoking.")            
            st.write("2. Eat healthy food.")            
            st.write("3. Avoid alcohol consumption.")            
            st.write("4. Regular physical activity.")            
            st.write("5. Maintain a healthy weight.")            
        elif class_labels[predicted_class] == 'Pituitary Tumor':            
            st.write("Precautions:")            
            st.write("1. Avoid smoking.")            
            st.write("2. Eat healthy food.")            
            st.write("3. Avoid alcohol consumption.")            
            st.write("4. Regular physical activity.")            
            st.write("5. Maintain a healthy weight.")   


        

if selected == "Breast Cancer Prediction":

    re=main()
    st.write(re)



if selected == "Bone fracture Prediction":
    
        
    st.title("Bone Fracture Prediction")

    def predict_bone_fracture(image, model):
    # Preprocess the image
        image = load_img(image, target_size=(256, 256))  # Resize to match model input
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)  # Add batch dimension
        image = image / 255.0  # Normalize the image

    # Make prediction
        prediction = model.predict(image)
        class_index = np.argmax(prediction, axis=1)[0]  # Get class index
        confidence = np.max(prediction)  # Get confidence

    # Class names (customize based on your dataset)
        class_names = ['Factured','Not Factured']
        result = class_names[class_index]

        return result, confidence

# Streamlit app

    st.write("Upload an X-ray image to predict whether there is a bone fracture.")

    # File uploader for X-ray images
    uploaded_file = st.file_uploader("Choose an X-ray image", type=["jpg", "jpeg", "png"])

    # Load the model
    

    if uploaded_file is not None:
        # Display the uploaded image
        st.image(uploaded_file, caption="Uploaded X-ray Image", use_column_width=True)

        # Make prediction
        if st.button("Predict"):
            result, confidence = predict_bone_fracture(uploaded_file, bone_fracture_model)
            
            st.write(f"Prediction: **{result}**")
            st.write(f"Confidence: **{confidence:.2f}**")
            if result == 'Factured':
                st.write("Precautions:")
                st.write("Avoid heavy lifting",
            "Use supports during recovery",
            "Follow physiotherapy advice"
        )
                st.write("food:")
                st.write("Dairy products (milk, cheese, yogurt)",
            "Leafy greens (spinach, kale)",
            "Nuts and seeds",
            "Calcium-rich foods",
            "Vitamin D supplements" )
                st.write("lifestyles:")
                st.write("Regular light exercises",
            "Maintain a healthy weight",
            "Stay active but cautious")
            else:
                st.write("Be Happy! You are healthy. Keep it up!")  









     
if selected == "Skin Cancer Prediction":

    st.title("Skin Cancer Prediction")
    # Import required libraries

# Preprocess the input image
    def preprocess_image(image):
    # Resize the image to the model's expected input shape
        image = image.resize((224, 224))  # Adjust size based on your model
        image_array = np.asarray(image) / 255.0  # Normalize to [0, 1]
        image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
        return image_array

# Make predictions
    def predict_skin_cancer(model, image_array):
        predictions = model.predict(image_array)
        class_index = np.argmax(predictions)  # Get the class with the highest probability
        confidence = predictions[0][class_index]  # Confidence score
        return class_index, confidence

    st.write("Upload a skin lesion image to predict its type.")

    # File uploader for image
    uploaded_file = st.file_uploader("Choose a skin lesion image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)



        # Preprocess the image
        image_array = preprocess_image(image)

        # Make prediction
        class_index, confidence = predict_skin_cancer(skin_cancer_model, image_array)

        # Map class index to skin cancer types
        classes = ["Malignant","Benign"]  # Update based on your model's output
        predicted_class = classes[class_index]

        # Display the prediction
        st.success(f"Prediction: {predicted_class}")
        st.info(f"Confidence: {confidence:.2f}")
        st.subheader("Precautions:")
        st.write("- Seek medical advice immediately")
        st.write("- Avoid prolonged sun exposure")
        st.write("- Use broad-spectrum sunscreen daily")
        st.write("- Wear protective clothing and hats")
        st.write("- Regularly examine your skin for changes")
    
    # Recommended Foods
        st.subheader("Recommended Foods:")
        st.write("- Dairy products (milk, cheese, yogurt)")
        st.write("- Leafy greens (spinach, kale)")
        st.write("- Nuts and seeds")
        st.write("- Foods rich in antioxidants (berries, dark chocolate)")
        st.write("- Fatty fish (salmon, mackerel) for Omega-3")
        st.write("- Calcium-rich foods and Vitamin D supplements")
    
    # Additional Advice
        st.subheader("Lifestyle Recommendations:")
        st.write("- Follow the treatment plan provided by your doctor")
        st.write("- Attend regular follow-up appointments")
        st.write("- Avoid tanning beds and other sources of UV light")
        st.write("- Stay hydrated and maintain a balanced diet")
        st.write("- Practice good skincare routines and moisturize regularly")

        
    
    




        

   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
  












if selected == "Chatbot":


# Define hospital data
    
    hospitals  = {
    "Noble Hospital": {"location": "Ameerpet", "specialization": "General surgery, Cardiology, Orthopedics"},
    "Yashoda Hospital": {"location": "Malakpet", "specialization": "Orthopedics, Cardiology, Neurology"},
    "Care Hospital": {"location": "Banjara Hills", "specialization": "Cardiology, General surgery, Orthopedics"},
    "Global Hospitals": {"location": "Lakdikapul", "specialization": "Organ transplant, Cardiology, Neurology, Oncology"},
    "KIMS Hospitals": {"location": "Secunderabad", "specialization": "Cardiology, Orthopedics, Neurology, Urology"},
    "Medicover Hospitals": {"location": "HITEC City", "specialization": "Cardiology, Orthopedics, Pediatrics"},
    
    # Additional hospitals
    "Star Hospitals": {"location": "Banjara Hills", "specialization": "Cardiology, Neurology, Orthopedics"},
    "AIG Hospitals": {"location": "Gachibowli", "specialization": "Neurology, Cardiology, Orthopedics"},
    "Shree Krishna Hospital": {"location": "Secunderabad", "specialization": "General surgery, Orthopedics, Pediatrics"},
    "Vasavi Hospital": {"location": "Kukatpally", "specialization": "General medicine, Pediatrics, General surgery"},
    "Omega Hospitals": {"location": "Kondapur", "specialization": "Cardiology, Nephrology, Urology"},
    "Kachiguda Hospital": {"location": "Kachiguda", "specialization": "Orthopedics, General surgery, Pediatrics"},
    "Pace Hospitals": {"location": "Ameerpet", "specialization": "Cardiothoracic surgery, Nephrology, Pediatrics"},
    "Rajendra Memorial Hospital": {"location": "Begumpet", "specialization": "Orthopedics, Pediatrics, General surgery"},
    "Sree Manju Hospital": {"location": "Malkajgiri", "specialization": "Orthopedics, General surgery, Gynecology"},
    "Chirayu Hospital": {"location": "Dilsukhnagar", "specialization": "Orthopedics, Pediatrics, General surgery"},
    "Surya Hospital": {"location": "Ameerpet", "specialization": "General surgery, Orthopedics, Gynecology"},
    "Sahaya Hospital": {"location": "Tolichowki", "specialization": "Orthopedics, Pediatrics, General surgery"},
    "Vijaya Hospital": {"location": "Malakpet", "specialization": "General surgery, Orthopedics, Gynecology"},
    "Rajiv Gandhi International Institute of Medical Sciences": {"location": "Bachupally", "specialization": "Neurosurgery, Cardiology, Orthopedics"},
    "Shree Sai Prathima Hospital": {"location": "Kachiguda", "specialization": "General surgery, Gynecology, Pediatrics"},
    "Sri Sai Hospital": {"location": "Miyapur", "specialization": "Orthopedics, General surgery, Dermatology"},
    "Sri Siva Hospital": {"location": "Dilsukhnagar", "specialization": "General surgery, Orthopedics, Pediatrics"},
    "Sree Krishna Institute of Medical Sciences": {"location": "Secunderabad", "specialization": "General surgery, Orthopedics, Pediatric care"},
    "Sree Sai Hospital": {"location": "Miyapur", "specialization": "Orthopedics, General surgery, Dermatology"},
    "Laxmi Hospital": {"location": "Dilsukhnagar", "specialization": "Pediatrics, General medicine, Gynecology"},
    
    # Adding more hospitals
    "Rainbow Children's Hospital": {"location": "Banjara Hills", "specialization": "Pediatrics, Neonatology, Pediatric surgery"},
    "Andhra Hospitals": {"location": "Saidabad", "specialization": "General surgery, Orthopedics, Gynecology"},
    "NIMS Hospital": {"location": "Punjagutta", "specialization": "Neurology, Oncology, General surgery"},
    "Mediciti Hospital": {"location": "Medchal", "specialization": "Cardiology, Nephrology, General surgery"},
    "Osmania General Hospital": {"location": "Old City", "specialization": "General medicine, General surgery, Orthopedics"},
    "Care Clinic": {"location": "Madhapur", "specialization": "Dermatology, General medicine, Orthopedics"},
    "Sakra Premium Hospital": {"location": "HITEC City", "specialization": "Cardiology, Pediatrics, Orthopedics"},
    "MaxCure Hospitals": {"location": "Kukatpally", "specialization": "Orthopedics, Cardiology, Pediatrics"},
    "Buddha Institute of Medical Sciences": {"location": "Warangal", "specialization": "Cardiology, Orthopedics, General surgery"},
    "Vasudeva Hospital": {"location": "Gachibowli", "specialization": "General surgery, Orthopedics, Neurology"},
    "Gandhi Medical College Hospital": {"location": "Musheerabad", "specialization": "General surgery, Pediatrics, Neurology"},
    "Krishna Institute of Medical Sciences": {"location": "Secunderabad", "specialization": "Cardiology, Neurology, Pediatrics"},
    "Medwin Hospitals": {"location": "Chikkadpally", "specialization": "General surgery, Orthopedics, Neurology"},
    "Shree Hospital": {"location": "Madhapur", "specialization": "Cardiology, Orthopedics, Pediatrics"},
    "Anand Hospital": {"location": "Dilsukhnagar", "specialization": "Orthopedics, General surgery, Neurology"},
    "Hegde Hospital": {"location": "Ameerpet", "specialization": "Orthopedics, Pediatrics, General surgery"},
    "Lakshmi Hospital": {"location": "Bachupally", "specialization": "General surgery, Orthopedics, Gynecology"},
    "Smile Hospital": {"location": "Tolichowki", "specialization": "Pediatrics, General surgery, Orthopedics"},
    
    # Adding more hospitals
    "Sunshine Hospital": {"location": "Secunderabad", "specialization": "Orthopedics, Neurosurgery, General surgery"},
    "Suraksha Hospital": {"location": "Gachibowli", "specialization": "Cardiology, Neurology, Pediatrics"},
    "Sree Bala Hospital": {"location": "Malkajgiri", "specialization": "Pediatrics, Orthopedics, General medicine"},
    "National Institute of Mental Health and Neurosciences (NIMHANS)": {"location": "Rajendranagar", "specialization": "Psychiatry, Neurology, Neurosurgery"},
    "Srinivas Hospital": {"location": "Banjara Hills", "specialization": "Cardiology, Orthopedics, Pediatrics"},
    "Vaidya Hospital": {"location": "Chandanagar", "specialization": "Ayurvedic medicine, General wellness, Orthopedics"},
    "Aditya Hospital": {"location": "Madhapur", "specialization": "General medicine, Orthopedics, Cardiology"},
    "Vibrant Health Care": {"location": "Gachibowli", "specialization": "Gynecology, Pediatrics, General surgery"},
    "Rainbow Hospitals for Women & Children": {"location": "Banjara Hills", "specialization": "Women’s health, Pediatric care, Neonatology"},
    "Medicity Hospital": {"location": "Malkajgiri", "specialization": "Orthopedics, General surgery, Neurology"},
}

    

# Define symptoms and diseases mapping
    symptoms_disease_mapping = {
    "fever, cough, sore throat": "Flu or COVID-19",
    "chest pain, shortness of breath": "Heart Attack or Angina",
    "headache, nausea, sensitivity to light": "Migraine",
    "joint pain, swelling, stiffness": "Arthritis",
    "abdominal pain, nausea, diarrhea": "Gastroenteritis",
    "skin rash, itching": "Allergic Reaction or Eczema",
    "fatigue, weight loss, frequent urination": "Diabetes or Thyroid Disorder",
    "dizziness, blurred vision, confusion": "Stroke or Hypoglycemia",
    "back pain, difficulty walking, leg weakness": "Sciatica or Spinal Stenosis",
    "fever, chills, night sweats": "Tuberculosis or Infection",
    "difficulty swallowing, persistent heartburn": "GERD or Esophageal Stricture",
    "swollen glands, sore throat, fever": "Mononucleosis or Tonsillitis",
    "rapid heartbeat, sweating, tremors": "Hyperthyroidism or Panic Attack",
    "pain during urination, blood in urine": "UTI or Kidney Stones",
    "persistent cough, weight loss, night sweats": "Tuberculosis or Lung Cancer",
    "sudden severe headache, neck stiffness": "Meningitis or Brain Hemorrhage",
    "bruising, excessive bleeding, fatigue": "Leukemia or Platelet Disorder",
    "tingling, numbness, weakness in limbs": "Multiple Sclerosis or Peripheral Neuropathy",
    "frequent infections, extreme tiredness": "Immunodeficiency or Chronic Fatigue Syndrome",
    "sharp abdominal pain, vomiting, fever": "Appendicitis or Gallstones",
    "persistent dry cough, shortness of breath": "Asthma or COPD",
    "severe sore throat, difficulty breathing": "Epiglottitis or Severe Tonsillitis",
    "yellowing of skin or eyes, dark urine": "Hepatitis or Jaundice",
    "painful blisters, tingling sensation": "Shingles or Herpes",
    "weight gain, sensitivity to cold, hair thinning": "Hypothyroidism or Cushing's Syndrome",
    "frequent nosebleeds, prolonged bleeding": "Hemophilia or Von Willebrand Disease",
    "double vision, slurred speech, one-sided weakness": "Stroke or Bell's Palsy",
    "restlessness, lack of focus, hyperactivity": "ADHD or Anxiety Disorder",
    "muscle cramps, fatigue, irritability": "Electrolyte Imbalance or Dehydration",
    "shortness of breath, swollen legs, fatigue": "Congestive Heart Failure or Pulmonary Hypertension",
    "pain in upper abdomen, bloating, nausea": "Pancreatitis or Gastric Ulcer",
    "loss of appetite, unintentional weight loss": "Cancer or Anorexia",
    "burning sensation in chest, acidic taste": "Heartburn or GERD",
    "persistent hoarseness, difficulty speaking": "Laryngitis or Vocal Cord Dysfunction",
    "difficulty concentrating, memory loss": "Dementia or Depression",
    "pale skin, fatigue, shortness of breath": "Anemia or Iron Deficiency",
    "difficulty urinating, weak urine stream": "Prostate Enlargement or Urethral Stricture",
    "irregular heartbeat, fatigue, fainting": "Arrhythmia or Atrial Fibrillation",
    "chronic diarrhea, blood in stool": "Crohn's Disease or Ulcerative Colitis",
    "severe leg pain, swelling, warmth": "Deep Vein Thrombosis (DVT) or Cellulitis",
}


# Define medicine information
    medicine_info = ({
    "Aspirin": "Used to relieve pain, reduce fever, and lower the risk of heart attack or stroke.",
    "Ciprofloxacin": "An antibiotic used to treat bacterial infections, including urinary tract infections.",
    "Loratadine": "An antihistamine used to treat allergies, hay fever, and skin rashes.",
    "Atorvastatin": "Used to lower cholesterol levels and reduce the risk of heart disease.",
    "Losartan": "Used to treat high blood pressure and protect the kidneys in diabetic patients.",
    "Ranitidine": "Used to reduce stomach acid and treat ulcers or gastroesophageal reflux disease (GERD).",
    "Albuterol": "Used to treat breathing problems such as asthma and chronic obstructive pulmonary disease (COPD).",
    "Hydrochlorothiazide": "A diuretic used to treat high blood pressure and fluid retention.",
    "Doxycycline": "An antibiotic used to treat bacterial infections, including acne and respiratory infections.",
    "Gabapentin": "Used to treat nerve pain and seizures.",
    "Prednisone": "A corticosteroid used to reduce inflammation and suppress the immune system.",
    "Clopidogrel": "Used to prevent blood clots in people with heart conditions or after a stroke.",
    "Warfarin": "An anticoagulant used to prevent and treat blood clots.",
    "Azithromycin": "An antibiotic used to treat bacterial infections, including respiratory and skin infections.",
    "Levothyroxine": "Used to treat hypothyroidism by replacing thyroid hormone.",
    "Diazepam": "Used to treat anxiety, muscle spasms, and seizures.",
    "Insulin": "Used to control blood sugar levels in people with diabetes.",
    "Lisinopril": "Used to treat high blood pressure and heart failure.",
    "Budesonide": "Used to treat asthma, allergic rhinitis, and inflammatory bowel disease.",
    "Metronidazole": "An antibiotic used to treat bacterial and parasitic infections.",
    "Fluoxetine": "An antidepressant used to treat depression, anxiety, and obsessive-compulsive disorder (OCD).",
    "Clarithromycin": "An antibiotic used to treat respiratory tract infections and skin infections.",
    "Furosemide": "A diuretic used to treat fluid retention and high blood pressure.",
    "Montelukast": "Used to prevent asthma symptoms and treat allergies.",
    "Salbutamol": "Used to relieve bronchospasm in conditions like asthma and COPD.",
    "Hydrocodone": "A pain reliever used to treat moderate to severe pain.",
    "Carbamazepine": "Used to treat seizures, nerve pain, and bipolar disorder.",
    "Propranolol": "Used to treat high blood pressure, anxiety, and migraines.",
    "Cetuximab": "A targeted therapy used to treat certain types of cancer.",
    "Esomeprazole": "Used to treat acid reflux, stomach ulcers, and conditions involving excessive stomach acid.",
    "Rosuvastatin": "Used to lower cholesterol levels and reduce the risk of heart disease and stroke.",
    "Tramadol": "A pain reliever used to treat moderate to severe pain.",
    "Zolpidem": "Used to treat insomnia by helping with sleep initiation.",
    "Methotrexate": "Used to treat autoimmune diseases such as rheumatoid arthritis and certain types of cancer.",
    "Amlodipine": "Used to treat high blood pressure and chest pain caused by angina.",
    "Bupropion": "An antidepressant used to treat depression and aid in smoking cessation.",
    "Ondansetron": "Used to prevent nausea and vomiting caused by chemotherapy, surgery, or radiation therapy.",
    "Spironolactone": "Used to treat fluid retention, high blood pressure, and hormonal acne.",
    "Allopurinol": "Used to reduce uric acid levels in the blood, often prescribed for gout.",
    "Tamsulosin": "Used to treat symptoms of an enlarged prostate (benign prostatic hyperplasia).",
    "Rivastigmine": "Used to treat mild to moderate dementia associated with Alzheimer's disease or Parkinson's disease.",
    "Duloxetine": "Used to treat depression, anxiety, and chronic pain conditions like fibromyalgia.",
    "Ivermectin": "Used to treat parasitic infections such as river blindness and scabies.",
    "Lamotrigine": "Used to treat epilepsy and bipolar disorder by stabilizing mood and preventing seizures.",
    "Phenytoin": "Used to control seizures in epilepsy and prevent seizures after neurosurgery.",
    "Erythromycin": "An antibiotic used to treat bacterial infections, including respiratory and skin infections.",
    "Lorazepam": "Used to treat anxiety disorders, insomnia, and seizures.",
    "Bisoprolol": "Used to treat high blood pressure and heart-related conditions such as heart failure.",
    "Mometasone": "A corticosteroid used to treat skin conditions, nasal allergies, and asthma.",
    "Topiramate": "Used to treat epilepsy and prevent migraines.",
    "Citalopram": "An antidepressant used to treat depression and anxiety disorders.",
    "Quetiapine": "Used to treat bipolar disorder, schizophrenia, and depression.",
    "Famotidine": "Used to treat stomach ulcers, acid reflux, and heartburn.",
    "Rivaroxaban": "An anticoagulant used to prevent and treat blood clots.",
    "Venlafaxine": "Used to treat depression, anxiety, and panic disorders.",
    "Ketorolac": "A pain reliever used for short-term management of moderate to severe pain.",
    "Hydroxychloroquine": "Used to treat malaria, lupus, and rheumatoid arthritis.",
    "Lidocaine": "A local anesthetic used to numb tissue in a specific area or to treat arrhythmias.",
    "Levetiracetam": "Used to treat seizures in epilepsy.",
    "Varenicline": "Used to help people quit smoking by reducing cravings and withdrawal symptoms.",
    "Sitagliptin": "Used to control blood sugar levels in people with type 2 diabetes.",
    "Dexamethasone": "A corticosteroid used to treat inflammation, allergies, and certain autoimmune diseases.",
    "Apixaban": "An anticoagulant used to reduce the risk of stroke and blood clots.",
    "Betamethasone": "A corticosteroid used to treat skin conditions and inflammatory diseases.",
    "Terbinafine": "An antifungal used to treat fungal infections of the skin and nails.",
    "Linezolid": "An antibiotic used to treat serious bacterial infections, including resistant strains.",
    "Cyclobenzaprine": "Used to relieve muscle spasms caused by acute musculoskeletal conditions.",
    "Gabapentin Enacarbil": "Used to treat nerve pain and restless legs syndrome.",

})


# Functions for chatbot operations
    def find_hospitals(location):
        location = location.lower()
        available_hospitals = [name for name, info in hospitals.items() if location in info['location'].lower()]
        if available_hospitals:
            result = "\n".join([f"{name} - {hospitals[name]['specialization']}" for name in available_hospitals])
            return result
        else:
            return "No hospitals found for the given location."

    def predict_disease(symptoms):
        symptoms = symptoms.lower()
        for symptom, disease in symptoms_disease_mapping.items():
            if all(word in symptoms for word in symptom.split(", ")):
                return f"Possible disease: {disease}"
        return "No matching disease found. Please consult a doctor."

    def get_medicine_info(medicine_name):
        medicine_name = medicine_name.capitalize()
        return medicine_info.get(medicine_name, "Medicine not found. Please consult a doctor or pharmacist.")

# Streamlit UI
   # st.set_page_config(page_title="Patient Assistant Chatbot", layout="wide")

# Chatbot interface in the bottom-right corner
    st.title("Patient Assistant ")
    st.markdown("<h3 style='text-align: center;'>💬 Patient Assistant Chatbot</h3>", unsafe_allow_html=True)
    st.write("Hi! How can I assist you today?")
    
    # Options for chatbot
    option = st.radio("Select an option:", ["Hyderabad Hospitals", "Predict Disease", "Medicine Info"])

    if option == "Hyderabad Hospitals":
        location = st.text_input("Enter your location:")
        if st.button("Find Hospitals"):
            if location:
                result = find_hospitals(location)
                st.success(result)
            else:
                st.error("Please enter a location.")

    elif option == "Predict Disease":
        symptoms = st.text_input("Enter your symptoms (e.g., fever, cough):")
        if st.button("Find Disease"):
            if symptoms:
                result = predict_disease(symptoms)
                st.success(result)
            else:
                st.error("Please enter your symptoms.")

    elif option == "Medicine Info":
        medicine_name = st.text_input("Enter the medicine name:")
        if st.button("Get Info"):
            if medicine_name:
                result = get_medicine_info(medicine_name)
                st.success(result)
            else:
                st.error("Please enter a medicine name.")

# Add chatbot location to bottom-right
   
