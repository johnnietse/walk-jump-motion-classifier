# Walking vs Jumping Classification Project

## Overview

This project aims to build a machine learning model that classifies movement data as either "walking" or "jumping." The model is trained using accelerometer data collected from smartphone sensors, and it is implemented as a Tkinter desktop application that allows users to visualize the results.

## Table of Contents

- [Features](#features)
- [Technologies Used](#technologies-used)
- [Data Collection](#data-collection)
- [Feature Extraction and Normalization](#feature-extraction-and-normalization)
- [Training the Classifier](#training-the-classifier)
- [Model Evaluation](#model-evaluation)
- [Model Deployment](#model-deployment)
- [Usage](#usage)
- [License](#license)

## Features

- Classifies accelerometer data into two categories: walking and jumping.
- User-friendly GUI to load CSV files and visualize results.
- Detailed evaluation metrics including precision, recall, F1-score, and confusion matrix.
- ROC curve plotting and AUC score calculation.

## Technologies Used

- Python
- Scikit-learn
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Tkinter

## Data Collection

Data is collected using a smartphone's accelerometer, segmented into 5-second intervals for analysis. The dataset consists of labeled segments indicating whether the activity during that time was walking or jumping.

## Feature Extraction and Normalization

Features are extracted from the accelerometer data, focusing on metrics like median acceleration. The data is then normalized to ensure consistent input for the logistic regression model.

## Training the Classifier

To train the model, the dataset is split into features (`X`) and labels (`Y`). A logistic regression classifier is created and trained using the following code snippet:

```python
# Train the model
xTrain = df_train_normalized.iloc[:, 0:11].values
yTrain = df_train_normalized['label'].values
model = LogisticRegression()
clf = make_pipeline(preprocessing.StandardScaler(), model)
clf.fit(xTrain, yTrain)
