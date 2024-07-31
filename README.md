# Breast Cancer Wisconsin (Diagnostic) Dataset Analysis

## Overview

This project involves analyzing the Breast Cancer Wisconsin (Diagnostic) dataset, which includes features computed from digitized images of fine needle aspirates (FNA) of breast masses. The dataset is available on Kaggle.

**Dataset Source:**
- Kaggle: [Breast Cancer Wisconsin (Diagnostic) Dataset](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data)

## Dataset Description

The dataset contains 569 instances with 32 features:
1. ID number
2. Diagnosis (M = malignant, B = benign)
3-32. Ten real-valued features computed for each cell nucleus (mean, standard error, and worst values).

**Feature Details:**
- `radius`, `texture`, `perimeter`, `area`, `smoothness`, `compactness`, `concavity`, `concave points`, `symmetry`, `fractal dimension`
- For each feature, statistics are provided for mean, standard error, and worst value.

**Class Distribution:**
- 357 benign
- 212 malignant

## Project Steps

1. **Data Loading:** 
   - Loaded the dataset from Kaggle.
   
2. **Exploratory Data Analysis (EDA):**
   - Created boxplots, correlation matrices, and pairplots.

![image](https://github.com/user-attachments/assets/8efdcb47-e852-4901-b869-0fe75a223869)
![image](https://github.com/user-attachments/assets/e6a03983-6313-4cc4-96b6-f975178b39ac)

3. **Feature Engineering:**
   - Created additional features including:
     - Mean of worst features.
     - Mean of standard error features.
     - Ratios and differences between mean and worst features.

4. **Feature Scaling:**
   - Scaled features to normalize the data.

5. **Train-Test Split:**
   - Split data into training and test sets.

6. **Model Evaluation:**
   - Evaluated models including Logistic Regression, SVM, XGBoost, CatBoost, Neural Networks, LightGBM, and Random Forest.
   - Achieved a highest accuracy of 98.2% with Random Forest.

7. **Hyperparameter Tuning:**
   - Used grid search for hyperparameter tuning.

8. **Performance Metrics:**
   - ROC curve
   - Precision-Recall curve
   - Feature importance
   - Model comparison
   - Confusion matrix
   - Learning curves
   - SHAP (Shapley Additive Explanations)
   - Cross-validation scores
   - Residuals plot

## Results

- Highest Accuracy: 98.2% with Random Forest
- Visualizations: ROC curve, precision-recall curve, feature importance plots, and more.

  ![image](https://github.com/user-attachments/assets/16fbea11-99d7-4b3d-b697-93217b685fd7)
  ![image](https://github.com/user-attachments/assets/8e5e515f-5824-424e-89ec-86975cd3dfdc)
  ![image](https://github.com/user-attachments/assets/a470911b-3f7a-4b77-b55f-8bd00ad57399)




## Files

- `Breast-Cancer-Diagnostic-Analysis.ipynb`: Script for data loading, cleaning,feature engineering,model training, evaluation, hyperparameter tuning and visualization plots.
- `data.csv`: Dataset.
- `README.md`: This file.

## Requirements

- Python 3.x
- Libraries: pandas, numpy, scikit-learn, xgboost, catboost, lightgbm, keras, matplotlib, seaborn, shap
