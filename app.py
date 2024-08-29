from tkinter import *
from tkinter import filedialog
from tkinter import ttk
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
import features 
import numpy as np
import ingest
from sklearn import preprocessing
from scipy.signal import savgol_filter

# data, labels = model.runModel(pd.read_csv('./csv/johnnie/Jumping Raw Data.csv'))
# print(labels)
def addFile(fp, action):
    df = fp
    ingest.label(df, action)
    splt = ingest.split(df)
    print(f"Data Split... {len(splt)} segments.")
    return splt

def openFile():
    file = filedialog.askopenfilename()
    
    trainData = []
    testData = []

    # Read the test and training data from the HDF5 file
    with pd.HDFStore('data.hdf5', 'r') as fp:
        for key in fp.keys():
            if key.startswith('/dataset/training/data_'):
                trainData.append(fp[key])
            elif key.startswith('/dataset/test/data_'):
                testData.append(fp[key])

    # Visualization => Display the first 3 walking and jumping frames from the test data
    # plots.plot3(trainData, title='Raw Training Data')
                
    # trainData = []
    # trainData = hdf5.addFile(pd.read_csv('./csv/johnnie/Walking Raw Data.csv'), ingest.ISWALKING)
    # trainData += hdf5.addFile(pd.read_csv('./csv/johnnie/Jumping Raw Data.csv'), ingest.ISJUMPING)

    testData = []
    testData = addFile(pd.read_csv(file), 0)

    processed_trainData = []
    processed_testData = []

    # Preprocess the data
    for frame in trainData:
        frame.dropna()
        # Apply a moving average filter
        processed_trainData.append(frame.rolling(window=90).mean()) # low pass used to preserve large spikes in jumping data

    for frame in testData:
        frame.dropna()
        # Apply a moving average filter
        processed_testData.append(frame.rolling(window=90).mean()) # low pass used to preserve large spikes in jumping data



    # Visualization => Display the first 3 walking and jumping frames from the test data (after filtering)
    # plots.plot3(trainData, title='Filtered Training Data')


    # Extract features from the model
    features_trainData = []
    features_testData = []
    for frame in trainData:
        features_trainData.append(features.extract(frame))

    for frame in testData:
        features_testData.append(features.extract(frame))

    # Combine the data into a single DataFrame
    df_train = pd.concat(features_trainData)
    df_train.index = range(len(df_train)) # Restore data indexes

    df_test = pd.concat(features_testData)
    df_test.index = range(len(df_test)) # Restore data indexes


    # Outlier removal
    print('Removing outliers...')
    print(f'Length before outlier removal: {len(df_test)} (test)')
    print(f'Length before outlier removal: {len(df_train)} (train)')
    # for col in df_train.columns:
    #     Q1 = df_train[col].quantile(0.25)
    #     Q3 = df_train[col].quantile(0.75)
    #     IQR = Q3 - Q1
    #     lower_bound = Q1 - 1.5 * IQR
    #     upper_bound = Q3 + 1.5 * IQR
    #     df_train = df_train[(df_train[col] >= lower_bound) & (df_train[col] <= upper_bound)]
    #     df_test = df_test[(df_test[col] >= lower_bound) & (df_test[col] <= upper_bound)]


    df_train.index = range(len(df_train)) # Restore data indexes
    df_test.index = range(len(df_test)) # Restore data indexes
    print(f'Length after outlier removal: {len(df_test)} (test)')
    print(f'Length after outlier removal: {len(df_train)} (train)')

    # Normalize the data (Leave the label column untouched)
    sc = preprocessing.StandardScaler()
    sc.fit(df_train.iloc[:, 0:11])  # Fit the scaler only on the training data

    df_train_normalized = pd.concat([pd.DataFrame(sc.transform(df_train.iloc[:, 0:11])), df_train.iloc[:, 12]], axis=1)
    df_test_normalized = pd.concat([pd.DataFrame(sc.transform(df_test.iloc[:, 0:11])), df_test.iloc[:, 12]], axis=1)

    # print(f'mean before normalizing {df_test}')
    # print(f'mean after normalizing: {df_test_normalized}')


    # Train the model
    xTrain = df_train_normalized.iloc[:, 0:11].values
    yTrain = df_train_normalized['label'].values


    print(f'Training model with {len(trainData)} samples...')


    yTrain = np.array(yTrain).astype(int)
    xTrain = np.array(xTrain)

    model = LogisticRegression()
    model.fit(xTrain, yTrain)

    print(f'Predicting {len(testData)} samples...')

    # Predict the test data
    yTest = np.array(df_test_normalized["label"].values).astype(int)
    xTest = np.array(df_test_normalized.iloc[:, 0:11].values)
    predict = model.predict(xTest)

    accuracy = accuracy_score(yTest, predict)

    print(f"Accuracy: {accuracy}")
    print(yTest)
    print(predict)

    df_test_normalized['label'] = predict

    output = sc.inverse_transform(df_test_normalized.iloc[:, 0:11])
    output = pd.DataFrame(output)
    output['label'] = predict
    output['label'].replace({1: 'Jumping', 0: 'Walking'}, inplace=True)
    output.to_csv('output.csv')
    output.index = range(len(output))

    walkingOutput =  output[output['label'] == 'Walking']
    jumpingOutput =  output[output['label'] == 'Jumping']

    plt.figure()
    plt.scatter(walkingOutput.index.values*5, walkingOutput.iloc[:, 2], label='Walking', color='blue')
    plt.scatter(jumpingOutput.index.values*5, jumpingOutput.iloc[:, 2], label='Jumping', color='red')
    plt.title('Median Acceleration vs Time')
    plt.legend()
    plt.show()

# openFile()
root = Tk()
frm = ttk.Frame(root, padding=10)
frm.grid()
root.title("ELEC 292 Project")
root.geometry("400x200")
ttk.Label(frm, text="ELEC 292 Project").grid(column=0, row=0)
ttk.Button(frm, text="Quit", command=root.destroy).grid(column=1, row=1)
ttk.Button(frm, text="Load File", command=openFile).grid(column=0, row=1)
root.mainloop()