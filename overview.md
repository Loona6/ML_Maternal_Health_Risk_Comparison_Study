# Maternal Health Risk Prediction Under Noisy and Missing Data

## Project Overview

## Title

**Maternal Health Risk Prediction Under Noisy and Missing Data: A Comparative Study of Classical ML Classifiers**

---

# Project Context

Maternal health complications, especially preeclampsia and pregnancy-related risks, remain a major healthcare challenge in low-resource environments.

In rural healthcare settings, clinical measurements are often collected using:

- Low-cost medical devices
- Incomplete monitoring equipment
- Manual measurements
- Limited access to specialists

As a result, real-world healthcare data is rarely perfect.

Common problems include:

- Missing measurements because equipment is unavailable
- Sensor errors due to device calibration issues
- Incorrect vital sign readings
- Data quality variation between locations

However, many machine learning studies evaluate models only on clean datasets.

This creates a gap:

> A model that performs well on perfect laboratory data may fail when deployed in real clinical environments.

---

# Research Objective

This project aims to evaluate how classical machine learning classifiers behave when maternal health data quality gradually decreases.

Instead of only finding the model with the highest accuracy, we study:

> Which ML classifier remains the most reliable when clinical data becomes noisy and incomplete?

---

# Research Question

**Under simulated real-world conditions of missing and noisy antenatal measurements, which classical ML classifier produces the most reliable predictions and probability estimates, and at what degradation level does performance become unsafe?**

---

# Dataset

## Maternal Health Risk Dataset

Source:

Kaggle Maternal Health Risk Dataset

Dataset size:

```
1014 patient records
```

Features:

| Feature | Description |
|---|---|
| Age | Patient age |
| SystolicBP | Upper blood pressure |
| DiastolicBP | Lower blood pressure |
| BS | Blood glucose level |
| BodyTemp | Body temperature |
| HeartRate | Heart rate |

Target:

```
RiskLevel
```

Classes:

```
Low Risk
Mid Risk
High Risk
```

---

# Models Evaluated

The project focuses on lightweight classical ML models suitable for low-resource deployment.

Models:

1. Logistic Regression

2. Random Forest

3. Support Vector Machine (SVM)

4. XGBoost (optional comparison)

---

# Experimental Plan

The project is divided into multiple stages.

---

# Stage 1: Exploratory Data Analysis

Goal:

Understand the clean dataset before modelling.

Tasks:

- Dataset statistics
- Feature distributions
- Class distribution
- Feature relationships
- Correlation analysis

Output:

Understanding of the original data characteristics.

Notebook:

```
01_eda.ipynb
```

---

# Stage 2: Clean Baseline Evaluation

Goal:

Measure model performance on the original clean dataset.

No artificial corruption is applied.

Evaluation metrics:

- Accuracy
- Macro F1 Score
- High-risk Recall
- Brier Score
- Confusion Matrix

Purpose:

Establish the starting performance before introducing real-world problems.

Notebook:

```
02_baseline_experiment.ipynb
```

---

# Stage 3: Data Degradation Simulation

Goal:

Simulate realistic clinical data problems.

Types of degradation:

## Missing Data

Examples:

- Missing glucose measurement
- Missing blood pressure reading

Techniques:

- MCAR (random missing values)
- MNAR simulation (equipment stock-out)

---

## Sensor Noise

Examples:

Blood pressure device error:

```
±5 mmHg
±10 mmHg
±15 mmHg
```

Temperature calibration error:

```
biased under-reporting
```

---

# Stage 4: Robustness Evaluation

Goal:

Measure how quickly each model performance decreases under degradation.

Experiments:

Clean Data

↓

Mild Degradation

↓

Moderate Degradation

↓

Severe Degradation


Compare:

- Accuracy drop
- Macro F1 drop
- High-risk recall drop
- Probability reliability change

Notebook:

```
03_degradation_experiment.ipynb
```

---

# Stage 5: Model Interpretability

Goal:

Understand why models make predictions.

Technique:

SHAP analysis

Questions:

- Which features influence predictions?
- Do important features remain stable under noise?

Notebook:

```
04_interpretability_shap.ipynb
```

---

# Project Structure

```
maternal-health-risk-prediction/

│
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   │
│   ├── raw/
│   │   └── Maternal Health Risk Data Set.csv
│   │
│   └── processed/
│
│
├── src/
│   │
│   ├── __init__.py
│   │
│   ├── preprocessing.py
│   │   # Dataset loading
│   │   # Label encoding
│   │   # Basic preparation
│   │
│   ├── degradation.py
│   │   # Missing data simulation
│   │   # Sensor noise injection
│   │
│   ├── models.py
│   │   # Model definitions
│   │   # ML pipelines
│   │
│   ├── metrics.py
│   │   # Accuracy
│   │   # F1
│   │   # Brier score
│   │   # Other evaluation functions
│   │
│   └── config.py
│       # Paths
│       # Random seeds
│       # Experiment settings
│
│
├── notebooks/
│   │
│   ├── 01_eda.ipynb
│   │
│   ├── 02_baseline_experiment.ipynb
│   │
│   ├── 03_degradation_experiment.ipynb
│   │
│   └── 04_interpretability_shap.ipynb
│
│
├── results/
│   │
│   ├── baseline_results.csv
│   ├── degradation_results.csv
│   │
│   └── figures/
│       ├── accuracy_comparison.png
│       ├── f1_comparison.png
│       ├── brier_comparison.png
│       └── shap_plots/
│
└── report/
```

---

# File Responsibilities

## preprocessing.py

Contains:

- Dataset loading
- Target separation
- Label encoding

Does NOT contain:

- Noise generation
- Model training
- Evaluation

---

## models.py

Contains:

- Logistic Regression pipeline
- Random Forest
- SVM
- XGBoost

Handles:

- Scaling where required
- Model configuration

---

## degradation.py

Contains:

All artificial data corruption methods:

- MCAR missing values
- MNAR missing values
- Blood pressure noise
- Temperature bias

---

## metrics.py

Contains:

Evaluation functions:

- Accuracy
- Macro F1
- Recall
- Brier Score
- Confusion matrix utilities

---

## config.py

Contains:

Shared settings:

- Dataset path
- Random seed
- Cross-validation settings
- Noise severity levels

---

# Expected Final Contribution

The project does not aim to create the most accurate maternal risk classifier.

Instead, it aims to answer:

> Which model is the safest and most reliable when real-world clinical data quality decreases?

Expected outcome:

A robustness comparison showing:

- Which models degrade slowly
- Which models maintain reliable probabilities
- Which models are suitable for low-resource healthcare deployment

---

# Development Philosophy

Keep the project:

- Reproducible
- Modular
- Easy to evaluate
- Easy to extend

Avoid:

- Overcomplicated deep learning models
- Excessive feature engineering
- Unnecessary optimization

The focus is **robustness evaluation of classical ML models under realistic clinical data degradation**.