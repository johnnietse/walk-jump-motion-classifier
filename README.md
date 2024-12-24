# Walking vs Jumping Classification Project

## Overview of Project Goal

The goal of this project is to develop a desktop application that distinguishes between 'walking' and 'jumping' using accelerometer data collected from a smartphone. The application is implemented and deployed as an executable Tkinter desktop app that allows users to visualize the results.

## Project Description
This project involves creating a simple desktop application that accepts accelerometer data (x, y, and z axes) in CSV format and outputs a separate CSV file containing labels (‘walking’ or ‘jumping’) for the corresponding input data for further analysis. The classification will be performed using a logistic regression model.

## Table of Contents

- [Features](#features)
- [Project Steps](#project-steps)
- [Libraries Used](#libraries-used)
- [Code Structure](#code-structure)
- [Data Collection](#data-collection)
- [Data Storing](#data-storing)
- [Visualization](#visualization)
- [Pre-processing](#pre-processing)
- [Feature Extraction and Normalization](#feature-extraction-and-normalization)
- [Training the Model](#training-the-model)
- [Creating a Simple Desktop Application](#creating-a-simple-desktop-application)
- [Usage Instructions](#usage-instructions)

## Features
- Accepts accelerometer data in CSV format (x, y, z axes).
- Processes and classifies the data into 'walking' or 'jumping' labels.
- Outputs results into a separate CSV file.
- Visualizes the data for better understanding and analysis.
- User-friendly desktop application with a simple UI built using Tkinter.

## Project Steps
The project is organized into seven main steps:
1. **Data Collection**: Gather accelerometer data from smartphones.
2. **Data Storing**: Store collected data in an HDF5 file for efficient access.
3. **Visualization**: Use Matplotlib to display relevant data segments.
4. **Pre-processing**: Clean and filter the data to remove noise and outliers.
5. **Feature Extraction & Normalization**: Extract meaningful features from the data and normalize them for model training.
6. **Training the Model**: Train the logistic regression model using the prepared dataset.
7. **Creating a Simple Desktop Application**: Build a user interface for loading files and displaying results.



## Libraries Used
To get started, ensure you have the required libraries installed. You can install them using pip:
- Tkinter
- Pandas
- Matplotlib
- Scikit-learn
- NumPy
- SciPy
- Seaborn
- h5py


## Code Structure
The project includes the following main modules:

- Main Application: Uses Tkinter for the UI and handles data loading, processing, and visualization.
- Feature Extraction: Contains functions to extract statistical features from the accelerometer data.
- Data Ingestion: Manages the loading and segmentation of the data into training and test sets.


## Data Collection
Data is collected from accelerometers and stored in CSV format. For demonstration purposes, two datasets are used: one for walking and one for jumping.

```python
import pandas as pd

# Load the files
FileWalking_Johnnie = pd.read_csv('./csv/johnnie/Walking Raw Data.csv')
FileJumping_Johnnie = pd.read_csv('./csv/johnnie/Jumping Raw Data.csv')
```


## Data Storing
The collected data is split into segments and saved into an HDF5 file for efficient access during training and testing.

```python
from sklearn.model_selection import train_test_split

def addFile(fp, action):
    df = fp
    ingest.label(df, action)
    splt = ingest.split(df)
    print(f"Data Split... {len(splt)} segments.")
    return splt

# Split the data into 5 second clips
walkingData_Johnnie = addFile(FileWalking_Johnnie, ingest.ISWALKING)
jumpingData_Johnnie = addFile(FileJumping_Johnnie, ingest.ISJUMPING)

# Combine and split the data into training and test sets
data = walkingData_Johnnie + jumpingData_Johnnie
trainData, testData = train_test_split(data, test_size=0.1, shuffle=True)

# Save the data to an HDF5 file
with pd.HDFStore('data.hdf5', 'w') as fp:
    for i, frame in enumerate(trainData):
        fp[f'dataset/training/data_{i}'] = frame
    for i, frame in enumerate(testData):
        fp[f'dataset/test/data_{i}'] = frame
```


## Visualization
Visualization of the raw training data is performed to understand the distributions of accelerometer readings.

```python
import matplotlib.pyplot as plt

# Example of plotting
def visualize_data(data):
    plt.figure()
    plt.scatter(data.index, data['acceleration'], label='Acceleration')
    plt.title('Acceleration Data Visualization')
    plt.xlabel('Time')
    plt.ylabel('Acceleration')
    plt.legend()
    plt.show()
```

## Pre-processing
The data is pre-processed by applying a moving average filter and removing NaN values.


```python
# Preprocess the data
def preprocess_data(data):
    data.dropna(inplace=True)
    data = data.rolling(window=90).mean()  # Apply moving average filter
    return data

processed_trainData = [preprocess_data(frame) for frame in trainData]
processed_testData = [preprocess_data(frame) for frame in testData]
```


## Feature Extraction and Normalization
Relevant features are extracted from the accelerometer data, and normalization is applied to prepare the data for model training.

```python
from sklearn import preprocessing

def extract_features(dataset):
    features_df = pd.DataFrame(columns=['max', 'min', 'mean', 'median', 'label'])
    features_df.loc[0, 'max'] = dataset['acceleration'].max()
    features_df.loc[0, 'min'] = dataset['acceleration'].min()
    features_df.loc[0, 'mean'] = dataset['acceleration'].mean()
    features_df.loc[0, 'median'] = dataset['acceleration'].median()
    features_df.loc[0, 'label'] = dataset['label'].iloc[0]
    return features_df

# Extract features for training and test data
features_trainData = [extract_features(frame) for frame in processed_trainData]
features_testData = [extract_features(frame) for frame in processed_testData]

# Normalize the data
scaler = preprocessing.StandardScaler()
df_train_normalized = pd.DataFrame(scaler.fit_transform(features_trainData))
df_test_normalized = pd.DataFrame(scaler.transform(features_testData))
```

## Training the Model
A logistic regression model is trained using the extracted features.

```python
from sklearn.linear_model import LogisticRegression

# Prepare the training data
xTrain = df_train_normalized.iloc[:, :-1].values
yTrain = df_train_normalized['label'].values

# Train the model
model = LogisticRegression()
model.fit(xTrain, yTrain)
```


## Creating a Simple Desktop Application
A simple user interface (UI) is created using Tkinter to load the CSV file and display results.

```python
from tkinter import *
from tkinter import filedialog, ttk

def openFile():
    file = filedialog.askopenfilename()
    # Load and process the selected file
    ...

root = Tk()
frm = ttk.Frame(root, padding=10)
frm.grid()
root.title("Accelerometer Data Classification")
root.geometry("400x200")
ttk.Label(frm, text="Accelerometer Classification App").grid(column=0, row=0)
ttk.Button(frm, text="Quit", command=root.destroy).grid(column=1, row=1)
ttk.Button(frm, text="Load File", command=openFile).grid(column=0, row=1)
root.mainloop()
```



## Usage Instructions
1. Data Preparation: Ensure that you have the accelerometer data files in CSV format.
2. Run the Application: Execute the script containing the Tkinter interface (app.py) to launch the GUI, where you can load your CSV file and classify the data.
3. Load the data file: Click on the "Load File" button to select your CSV file containing accelerometer data.
4. View Results: The application will process the data and output the classification results into a new separate CSV file named 'output.csv'.
