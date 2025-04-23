![Picture18](https://github.com/user-attachments/assets/a0a344ad-080e-4bd8-af27-7a82409ed230)# Walking vs Jumping Classification Project

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
- [Screenshots](#screenshots)

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
root.title("Walk-or-jump-motion-classifer")
root.geometry("400x200")
ttk.Label(frm, text="Walk-or-jump-motion-classifer").grid(column=0, row=0)
ttk.Button(frm, text="Quit", command=root.destroy).grid(column=1, row=1)
ttk.Button(frm, text="Load File", command=openFile).grid(column=0, row=1)
root.mainloop()
```



## Usage Instructions
1. Data Preparation: Ensure that you have the accelerometer data files in CSV format.
2. Run the Application: Execute the script containing the Tkinter interface (app.py) to launch the GUI, where you can load your CSV file and classify the data.
3. Load the data file: Click on the "Load File" button to select your CSV file containing accelerometer data.
4. View Results: The application will process the data and output the classification results into a new separate CSV file named 'output.csv'.

---

## 📸 Screenshots of other test results (with a different set of data)
![Walkorjump-motion-classifer](https://github.com/user-attachments/assets/5e2c432b-d48b-44d8-8362-469f83619416)
![raw-acceleration-data](https://github.com/user-attachments/assets/efaf1ec3-3440-49e2-9626-e95ad49205e4)
![filtered-training-data-pt1](https://github.com/user-attachments/assets/9182e231-407d-4bf4-90cc-7f6dcadbcace)
![filtered-training-data-pt2](https://github.com/user-attachments/assets/671c42b3-32ca-47d0-9e41-7b7cf2dac2d9)
![filtered-training-data-pt3](https://github.com/user-attachments/assets/2c71128d-8e84-48f2-bc06-5e1af9c927dd)
![unfiltered-acceleration-data](https://github.com/user-attachments/assets/9d7877b5-1fa0-421f-8ce9-d9196f7ca9e5)

![Rolling-average-filtered-data-40pts](https://github.com/user-attachments/assets/b88313db-892f-4665-92b1-97404dc34a19)
![Rolling-average-filtered-data-60pts](https://github.com/user-attachments/assets/8948d6ef-6c9c-4cd4-ac6e-d80a9ff362cb)
![Rolling-average-filtered-data-90pts png ](https://github.com/user-attachments/assets/92cfc013-ab84-4525-8cf7-40cdc6907b14)
![Rolling-average-filtered-data-120pts png ](https://github.com/user-attachments/assets/62332158-9f9b-424a-af68-3988d16a90e3)

![Scatter-Plot-of-Features-after-filtering-and-feature-extraction](https://github.com/user-attachments/assets/c512629f-9a35-460c-9eaf-785bded6f64c)
![Scatter-Plot-of-Features-(Outliers-Removed)](https://github.com/user-attachments/assets/507b5e75-0956-4df3-811b-024bc93caec6)
![Model-Confusion-Matrix](https://github.com/user-attachments/assets/c0639a8a-0084-464c-8d69-1aaba329b1c7)
![Model-ROC-Curve](https://github.com/user-attachments/assets/6e66f8f0-158f-4d6b-bd1c-e74e657b147a)
![Application generated plot of jumping and walking (Test Data)](https://github.com/user-attachments/assets/ca716fc5-7ca4-4e57-92e6-767b6d4b612f)


![Uploading Picture18.svg…](<svg width="597" height="321" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" xml:space="preserve" overflow="hidden"><defs><clipPath id="clip0"><rect x="87" y="594" width="597" height="321"/></clipPath><image width="765" height="412" xlink:href="data:image/jpeg;base64,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z3Oa56kaspwdOVkt9Nz2cFWy+nh8RDF0nOpJL2bUrKLvq2ut1/Wt1t0UUV0HjBRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQBTk82C8kkWFpkdFHyMAQQW65I9RXK+J/hp4X8aeItK13XPCNvqer6WQbO7uFQvFhtwH3uQG5AOQCSRXbUVEoRmrSV0dFDEVsLP2lCbjLVXTadno1dd1uVftU3/AD5Tf99J/wDFVhWvhHR7W/uL4eHlmvbi8a/e5umWeQTlY13qzsSnywxABcACNQAMCunoqznOa1bwrpGuaPNpd54djksJVCPCgjjGAoQYKsCPlAXg9BjpVq30bT7SGGKDw9BDFCgjjSOGFVRQHAVQDwAJJBgdnb1NbdFAHJaV4H0bRfFV/wCJLPQpYtZvohBNcm4Lfuxt+VVaQqg+RM7QM7FznAr5w/blha3g8BRvjesd4Djpn/R6+vK+Sv28v9Z4I+l7/wC0KAPrWiiigDkNS/5K54e/7Aep/wDpRYV19chqX/JXPD3/AGA9T/8ASiwrr6ACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACvkr9vL/WeCPpe/8AtCvrWvkr9vL/AFngj6Xv/tCgD61ooooA5DUv+SueHv8AsB6n/wClFhXX1yGpf8lc8Pf9gPU//Siwrr6ACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACivnX4laX8c5/wBpDwxc+F7pU+HCG3+2KJYhEIw3+kCVGO9nK52lQeq4wc19FVz0qzqynHla5XbXr5ryPZx+XLA0sPVVeFT2sea0Xdw1+GfaXkFFFFdB4wUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUV5p8WP2ifA/wV1TSNP8ValJZ3Wp5aJYoGl2Rg4Mj4+6uePXg8cGvSlYOoZSGUjII6Gso1YTlKEXdrfyO6tgcVh6NPE1qbjCpflbTSlZ2dn1s9xaKKK1OEKKKKACiiigAooooAKKKKACiiigAr5K/by/1ngj6Xv/tCvrWvkr9vL/WeCPpe/wDtCgD1/V/iZ4gtrfxDHaxaU2paZ4jt9Dgtyskgu/PitZIzkMChUXJLnDALE7Y4xW3r3j2/0X4iaLpb2sP/AAjuo7bZdQjKSObw+fmEr5odAPJX5hG4JLglNua6TT/B+g6ReT3dhomnWV1cT/aZp7e0jjeSbDDzGYDJbDuNx5+dvU1LH4Z0eLVI9STSrFNRjV1S8W2QTKHYu4D4yAzMzHnksSetAHJya1BqHxt02yiiu0mstE1ASPPZyxRNunsSPLkdQkvTnYWx0OM131chqX/JXPD3/YD1P/0osK6+gAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKAOB+JHwJ8DfFzUNKvvFmgxatdaYSbeRpZE+UkEowVhvXIB2tkdfU570AKAAMAdBS0VnGnCMnKKs3v5nXVxmIr0qdCrUlKFO/Km21G+r5Vsrve24UUUVocgUUUUAFFFFABRRRQAUUUUAFFFFABXyV+3l/rPBH0vf8A2hX1rXyV+3l/rPBH0vf/AGhQB9a0UUUAchqX/JXPD3/YD1P/ANKLCuvrkNS/5K54e/7Aep/+lFhXX0AFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUVQ17XLHwzol/q+p3C2mnWED3NzOwJEcaKWZsDk4APSk2krsuEJVJKEFdvRIv0VwXwh+N3hP44aPeal4VvZLmGzm8ieOeFopI2IyCVPYjofY9wa72op1IVYqcHdM6MVhMRga8sNioOE47pqzXqmFFFFaHIFFFfOmi+IvjfJ+1Ze6Zfadt+FwEmyX7PGIBCIiY3WXG8ymTaCuT1bgAAjnrVlRcU4t8ztotvN+R7OXZZLMo15Rqwh7KDn78uXmt9mPeT6LqfRdFFFdB4wUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABXyV+3l/rPBH0vf/AGhX1rXyV+3l/rPBH0vf/aFAH1rRWfN4h0q3VWl1OzjVpjbgvOgBlHVOv3vbrVqS8t4rqK2eeNLiUExws4DuB1IHU4oA5bUv+SueHv8AsB6n/wClFhXX1yGpf8lc8Pf9gPU//Siwrr6ACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKp6xpFn4g0m90zUbdLuwvIXt7i3kGVkjdSrKfYgkVcopNJqzKjKUJKUXZo4v4X/B7wl8GtJutN8JaSul211N582ZXleRsYGWck4A4Azgc+prtKKKmEI04qEFZLsb4nFV8ZWliMTNznLdybbfq3qFFFFWcwUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFfJX7eX+s8EfS9/wDaFfWtfJX7eX+s8EfS9/8AaFAHoeq/B3xLqmpeI54rbRtJnvNbbU9L1Kx1S6SXT82kNsZTEkSLK/7gOYnJRi5DEgZbotc+GOr698S7HxHLJpsVvGbUSMu9riBbWe7ePyiVxmZLoLJyNoVgN4bI9QooA4W402O0+NGk3Ky3Dvc6JqJZJZ3dF2z2I+RScJ152gZ713VchqX/ACVzw9/2A9T/APSiwrr6ACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAoorg/jn4O8QePvhXr+g+FtX/sLXLyJFt7zzGjAw6syFlG5QyhlyP71Z1JOEHKKu107nXhKMMRiKdGrUUIyaTk9opuzbt0W53lFeY/s4+APFHwz+FWnaD4v1oa5rMMkjmZZXlWKNmysQdwGYAeo4zgcAV6dSpSlUgpSjZvp2Lx2Hp4XFVKFGoqkYtpSW0knur9wooorU4QooooAKKKKACvkr9vL/WeCPpe/wDtCvrWvkr9vL/WeCPpe/8AtCgD61ooooA5DUv+SueHv+wHqf8A6UWFdfXIal/yVzw9/wBgPU//AEosK6+gAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAr5K/by/1ngj6Xv/ALQr61r5K/by/wBZ4I+l7/7QoA+taK8P1T426jZXmvafBrvhmfV9P8Sro9rY+S269ja2t5xGT9o/dOnmyh5TuUeUT5eflPS678SL/Tfip4e0q3vtFuPDeon7I0UGJtQN2PtIfgTLsjQwKpIjkIYSA7QpZQDc1L/krnh7/sB6n/6UWFdfXASalc3fxu022l0q7soLbRNQEV5O8Jiu909iSYwkjONuMHeqdeM9a7+gAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAr5K/by/1ngj6Xv/ALQr61r5K/by/wBZ4I+l7/7Q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" preserveAspectRatio="none" id="img1"></image><clipPath id="clip2"><rect x="0" y="0" width="5676492" height="3057143"/></clipPath></defs><g clip-path="url(#clip0)" transform="translate(-87 -594)"><g transform="matrix(0.000105 0 0 0.000105 87 594)"><g clip-path="url(#clip2)" transform="matrix(1.00162 0 0 1 -0.00334822 -0.115234)"><use width="100%" height="100%" xlink:href="#img1" transform="scale(7420.25 7420.25)"></use></g></g></g></svg>)
