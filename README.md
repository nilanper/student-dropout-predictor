---
title: Student Dropout Predictor with SHAP Explainer
emoji: 🎓
colorFrom: blue
colorTo: purple
sdk: gradio
app_file: app.py
pinned: false
---

# Student Dropout Predictor with SHAP Explainer

This Gradio app allows institutions to:
- Train a custom dropout prediction model
- Generate dropout probabilities
- Explain predictions using SHAP (Explainable AI)

## App Overview

The app has two main tabs:

### 1) Train Institution Model

Upload a labeled dataset to train an institution-specific model.

Expected training file:
- Target column (binary: Dropout / No Dropout)
- Student ID column
- Student Name column
- Remaining columns are used as features

What happens:
- Data preprocessing for numeric and categorical columns
- Model training using XGBoost
- Validation metrics are displayed

### 2) Predict + Explain

Use this tab after training is complete.

You can:
- Upload a new dataset
- Generate dropout probability and prediction
- Download prediction results
- Select a student and view a SHAP waterfall plot
- Download the SHAP plot

## Important Notes

- You must train the model first in Tab 1
- The prediction file must contain the same feature columns used during training
- The target column in the training file must be binary

## Sample Data

Included in `sample_data/`:
- `training_sample.csv`
- `prediction_sample.csv`

## Project Structure

```text
your-space/
├── app.py
├── requirements.txt
├── README.md
└── sample_data/
    ├── training_sample.csv
    └── prediction_sample.csv