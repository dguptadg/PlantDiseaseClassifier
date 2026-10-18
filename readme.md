
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

<img width="770" height="545" alt="ResNet Accuracy" src="./ResNet%20accuracy.png" />


---

### Confusion Matrices

<img width="770" height="545" alt="ResNet Confusion Matrix" src="./ResNet%20confusionMatrix.png" />


---

### ROC Curves

<img width="770" height="545" alt="ResNet ROC-AUC" src="./ResNet%20ROC-AUC.png" />


---

## Results

The notebook compares all three models using identical evaluation metrics, making it easy to understand how a custom CNN performs against transfer learning approaches. The final comparison includes accuracy, precision, recall, F1-score, ROC-AUC score, confusion matrices, and ROC curves for every model.

---

## Interface and Outputs
<img width="770" height="545" alt="Web Application Input" src="./Web%20input.png" />

<img width="770" height="545" alt="Web Application Output" src="./Web%20output%20final.png" /> 


## Future Improvements

- Fine-tune pretrained models further
- Add more real-world plant images
- Support additional crop diseases
- Optimize inference speed for mobile devices

---

## Author
Dhruv Gupta 
Roll number : 102303877
