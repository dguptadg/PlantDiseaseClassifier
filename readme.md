
# Plant Disease Classification and Recommendation System

A deep learning project for identifying plant diseases from leaf images using multiple neural network architectures. The project compares a custom CNN with two popular transfer learning models—VGG16 and ResNet50—and also provides disease recommendations through a gemini API call.

---

## Live Link

https://plantdiseaseclassifier-vlcrnsby6utakphfwbanhm.streamlit.app/

---

## Project Overview

The main goal of this project is to classify plant leaf diseases from the PlantVillage dataset and compare the performance of different deep learning models under the same training setup.

The notebook covers the complete workflow, including:
- Dataset download using the Kaggle API
- Data preprocessing and visualization
- Model building and training
- Performance evaluation
- Model comparison
- Disease prediction on new images
- Recommendation generation using Gemini AI

---

## Dataset

- **Dataset:** PlantVillage Dataset
- **Number of Classes:** 38
- **Image Size:** 224 × 224
- **Validation Split:** 20%

The dataset is automatically downloaded from Kaggle and extracted before training begins.

---

## Models Implemented

### 1. Custom CNN
A Convolutional Neural Network built from scratch using TensorFlow/Keras.

### 2. VGG16 (Transfer Learning)
Uses the pretrained VGG16 backbone with custom classification layers.

### 3. ResNet50 (Transfer Learning)
Uses the pretrained ResNet50 architecture with fine-tuning for plant disease classification.

---

## Technologies Used

- Python
- TensorFlow / Keras
- NumPy
- Pandas
- Matplotlib
- Pillow
- Google Gemini API

---

## Evaluation Metrics

The models are evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC Score
- Classification Report
- Confusion Matrix

---

## Visualizations

The notebook generates several graphs to help analyze model performance.

### Training & Validation Accuracy

<img width="770" height="545" alt="image" src="https://github.com/user-attachments/assets/ec4a4c41-5c89-435f-a072-fa12ceb6b6ec" />


<img width="767" height="550" alt="Screenshot 2026-07-02 132725" src="https://github.com/user-attachments/assets/0a56f89a-7650-46a7-ae05-2235ed6029b1" />


<img width="742" height="557" alt="Screenshot 2026-07-02 132747" src="https://github.com/user-attachments/assets/ba63aeb7-a0bd-41c7-bcaf-61046223aa06" />


---

### Confusion Matrices

<img width="1226" height="983" alt="image" src="https://github.com/user-attachments/assets/09d103ff-d283-4e7b-8798-a76411940480" />


<img width="1052" height="862" alt="image" src="https://github.com/user-attachments/assets/5d0635da-96a7-4bae-aed8-92a07c6cf386" />


<img width="1117" height="1000" alt="image" src="https://github.com/user-attachments/assets/34629dd4-9a9c-43f0-be0e-374a982bf662" />


---

### ROC Curves

<img width="971" height="683" alt="image" src="https://github.com/user-attachments/assets/19d132df-8864-400b-b83f-c98ab412aee5" />


<img width="790" height="576" alt="image" src="https://github.com/user-attachments/assets/04b6c78b-bda2-4bd7-b31c-bec49953f5c0" />


<img width="782" height="575" alt="image" src="https://github.com/user-attachments/assets/f3e255f8-5142-4825-a5f5-143b74dd8f45" />


---

## Results

The notebook compares all three models using identical evaluation metrics, making it easy to understand how a custom CNN performs against transfer learning approaches. The final comparison includes accuracy, precision, recall, F1-score, ROC-AUC score, confusion matrices, and ROC curves for every model.

---

## Interface and Outputs
<img width="1235" height="772" alt="image" src="https://github.com/user-attachments/assets/53e425ec-8b36-48ec-b6c7-dff2f4806f88" />

<img width="1242" height="867" alt="image" src="https://github.com/user-attachments/assets/5391200a-2682-4271-ab09-9342afca35ce" />


## Future Improvements

- Fine-tune pretrained models further
- Add more real-world plant images
- Support additional crop diseases
- Optimize inference speed for mobile devices

---

## Author
Dhruv Gupta 
Roll number : 102303877
