Here's the updated **README** based on your provided table:

---

# Plant Disease Detection Classification

## Project Overview

This project focuses on building machine learning models for the classification of plant diseases based on images. Using a publicly available dataset, we will explore classical machine learning algorithms and deep learning models, optimizing them using different techniques like regularization, early stopping, and various optimizers. The goal is to determine the most effective combination of hyperparameters to optimize the models for the task of detecting plant diseases.

## Dataset

The dataset consists of images of plants, each associated with one of 38 classes representing different plant diseases. The dataset has been split into:

- **Training Set**: 70295 images in 38 classes.
- **Validation Set**: 17572 images in 38 classes.

### Data Preprocessing

- Images are resized to `128x128`.
- Data augmentation is applied to improve model generalization.

## Models Implemented

### 1. **Classical Machine Learning Models**
We implemented the following classical machine learning algorithms:
- **Logistic Regression**
- **SVM (Support Vector Machine)**

The models were trained and optimized based on the task and hyperparameter tuning.

### 2. **Simple Neural Network (No Optimization)**
A basic neural network was implemented for plant disease classification without any optimization techniques, using the default settings.

### 3. **Optimized Neural Network**
The neural network models were optimized using:
- **Optimizers** (Nadam, RMSprop, Adam, SGD)
- **Regularization** (L1, L2)
- **Early Stopping**
- **Dropout**
- **Learning Rate Adjustments**
- **Epochs and Number of Layers**

### 4. **Evaluation and Metrics**
Metrics such as **Accuracy**, **F1-Score**, **Recall**, and **Precision** were used to evaluate model performance.

### 5. **Final Model Selection**
The final models were selected based on accuracy and other performance metrics. Here are the results:

## Hyperparameters and Results

| Training Instance  | Optimizer | Regularizer    | Epochs | Early Stopping | Number of Layers       | Learning Rate | Accuracy | F1 Score | Recall | Precision |
|--------------------|-----------|----------------|--------|----------------|------------------------|---------------|----------|----------|--------|-----------|
| **Model 1**        | Nadam     | l2(0.001)      | 10     | No             | 6 (2 Conv2D, 2 Dense)  | 0.0001        | 0.8774   | 0.8785   | 0.8774 | 0.8876    |
| **Model 2**        | RMSprop   | l1(0.0005)     | 10     | No             | 6 (2 Conv2D, 2 Dense)  | 0.0001        | 0.9125   | 0.9119   | 0.9125 | 0.9145    |
| **Model 3**        | Adam      | None           | 10     | No             | 6 (2 Conv2D, 2 Dense)  | 0.001         | 0.8808   | 0.8856   | 0.8808 | 0.9020    |
| **Model 4**        | SGD       | l2(0.001)      | 10     | No             | 6 (2 Conv2D, 2 Dense)  | 0.001         | 0.9118   | 0.9112   | 0.9118 | 0.9183    |
| **Model 5 (Simple NN)** | None  | None           | 10     | No             | 7 (3 Conv2D, 2 Dense)  | Default       | 0.9176   | 0.9177   | 0.9178 | 0.9213    |



## Saved Models

Due to the large size of the trained models, they were not stored directly in this repository. Instead, I have uploaded the saved models to a cloud storage platform (e.g., Google Drive) to ensure they are accessible.

You can download the saved models from the link below:

[**Download Saved Models**](https://drive.google.com/drive/folders/1vmLu4bWquu_Ikh1446U86rvJ36QFQZd4?usp=sharing)

These models include the best-performing configurations after hyperparameter tuning and optimization.


## Results Summary

- **Best Performing Model**: **Model 5** (Simple NN with 7 layers and default settings) achieved the highest accuracy of **91.76%** and outperformed the other models on **F1 Score**, **Precision**, and **Recall**.
- **Optimized Neural Networks**: Models using **RMSprop** and **SGD** achieved strong results, with **RMSprop** showing a good balance between **Accuracy** and **F1 Score**.
- **ML Algorithm Performance**: Classical ML algorithms, while effective, did not perform as well as the neural network models.

## Evaluation

- **Confusion Matrix**: The confusion matrix is generated for each model to visualize performance.
- **F1-Score, Precision, Recall**: These metrics will be used to assess the classification performance of all models.

## Predictions

The best model will be used to make predictions on the test dataset, which has not been used in training. The performance will be evaluated using **Accuracy**, **F1 Score**, **Precision**, and **Recall**.

## Requirements

- Python 3.x
- TensorFlow/Keras
- Scikit-learn
- XGBoost (optional)
- Matplotlib
- Numpy


## Video Presentation

A **5-minute video** will be uploaded, where I will:
- Discuss the dataset and problem statement.
- Explain the models used (classical ML and neural networks).
- Detail the hyperparameter choices and their impact on model performance.
- Demonstrate the final model selection and performance.

