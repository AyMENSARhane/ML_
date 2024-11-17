# Overview
**Intro_ML** is a machine learning project that focuses on data preprocessing, hyperparameter optimization, and model training. It supports two datasets: **Chronic Kidney Disease** and **Banknote Authentication**. This project includes robust pipelines for preprocessing, model evaluation, and saving the best-performing models for future use.

---

## Features
- **Dataset Management**: Fetch and preprocess datasets.
- **Preprocessing Pipeline**: Includes data cleaning, handling missing values, encoding categorical variables, and normalization.
- **Model Training**: Supports Logistic Regression, Decision Trees, Random Forests, SVM, and KNN classifiers.
- **Hyperparameter Optimization**: Uses `GridSearchCV` to find the best parameters.
- **Model Evaluation**: Evaluates models using metrics such as recall, precision, and accuracy.
- **Saved Results**: Best-performing models and hyperparameter search results are stored for further analysis.

---

## Repository Structure
The repository contains the following structure:
- `tools.py`: Core functions and classes for preprocessing and model evaluation.
- `workflow.ipynb`: Jupyter Notebook for running the full workflow.
- `best-models/`: Directory containing saved best-performing models (subfolders for each dataset).
- `grid-cv-results/`: Directory with CSV files of hyperparameter tuning results.
- `tests/`: Unit tests for validating the project.

---

## Setup Instructions
To set up the project on your machine, follow these steps:

1. Clone the repository.
2. Install the required Python libraries.

---

## Usage
Run the `workflow.ipynb` notebook, which demonstrates how to:
1. Load and preprocess datasets.
2. Train models and optimize hyperparameters.
3. Save results and best-performing models.

---

## Access Results
- **Grid search results**: CSV files stored in `grid-cv-results/`.
- **Best models**: Saved models in `best-models/`.

---

## Running Tests
Unit tests are provided to validate preprocessing functions and dataset handling.

---

## Results and Evaluation
- **Hyperparameter tuning results** are available in `grid-cv-results/`. Example files include:
  - `DT_result-KDN Dataset.csv`: Results for Decision Tree on the Kidney dataset.
  - `LR_result-BKN Dataset.csv`: Results for Logistic Regression on the Banknote dataset.
- **Best-trained models** are saved in `best-models/`:
  - **Kidney Dataset**: Models like `best_model_rf.pkl`, `best_model_knn.pkl`.
  - **Banknote Dataset**: Models like `best_model_lr.pkl`, `best_model_svm.pkl`.

---


