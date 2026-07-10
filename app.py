import streamlit as st
import tensorflow as tf
import os
import gdown
import numpy as np
import json
from PIL import Image
from tensorflow.keras.applications.resnet50 import preprocess_input

st.set_page_config(page_title="Plant Disease Classification", page_icon="🌿")

@st.cache_resource
def load_model():

    if not os.path.exists("model.h5"):
        file_id = "1VPCpaLqeOa9P9QjTOpGzYX14CBH8piEp"
        url = f"https://drive.google.com/uc?export=download&id={file_id}"
        gdown.download(url, "model.h5", quiet=False)

    return tf.keras.models.load_model("model.h5")

model = load_model()

with open("disease_info.json", "r") as file:
    disease_info = json.load(file)

class_names = {
0:'Apple___Apple_scab',
1:'Apple___Black_rot',
2:'Apple___Cedar_apple_rust',
3:'Apple___healthy',
4:'Blueberry___healthy',
5:'Cherry_(including_sour)___Powdery_mildew',
6:'Cherry_(including_sour)___healthy',
7:'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
8:'Corn_(maize)___Common_rust_',
9:'Corn_(maize)___Northern_Leaf_Blight',
10:'Corn_(maize)___healthy',
11:'Grape___Black_rot',
12:'Grape___Esca_(Black_Measles)',
13:'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
14:'Grape___healthy',
15:'Orange___Haunglongbing_(Citrus_greening)',
16:'Peach___Bacterial_spot',
17:'Peach___healthy',
18:'Pepper,_bell___Bacterial_spot',
19:'Pepper,_bell___healthy',
20:'Potato___Early_blight',
21:'Potato___Late_blight',
22:'Potato___healthy',
23:'Raspberry___healthy',
24:'Soybean___healthy',
25:'Squash___Powdery_mildew',
26:'Strawberry___Leaf_scorch',
27:'Strawberry___healthy',
28:'Tomato___Bacterial_spot',
29:'Tomato___Early_blight',
30:'Tomato___Late_blight',
31:'Tomato___Leaf_Mold',
32:'Tomato___Septoria_leaf_spot',
33:'Tomato___Spider_mites Two-spotted_spider_mite',
34:'Tomato___Target_Spot',
35:'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
36:'Tomato___Tomato_mosaic_virus',
37:'Tomato___healthy'
}

st.title("🌿 Plant Disease Classification")

uploaded_file = st.file_uploader(
    "Upload a leaf image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("Predict"):

        img = image.resize((224,224))
        img = np.array(img).astype(np.float32)
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)

        prediction = model.predict(img)

        predicted_index = np.argmax(prediction)

        confidence = np.max(prediction) * 100

        disease = class_names[predicted_index]

        st.success(f"Prediction : {disease}")

        st.info(f"Confidence : {confidence:.2f}%")

        if disease in disease_info:

            info = disease_info[disease]
        
            st.subheader("Description")
            st.write(info["description"])
        
            st.subheader("Symptoms")
            st.write(info["symptoms"])
        
            st.subheader("Causes")
            st.write(info["causes"])
        
            st.subheader("Treatment")
            st.write(info["treatment"])
        
            st.subheader("Prevention")
            st.write(info["prevention"])
