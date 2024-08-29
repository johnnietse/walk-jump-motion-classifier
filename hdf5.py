import pandas as pd
from sklearn.model_selection import train_test_split
import ingest
import numpy as np

def addFile(fp, action):
    df = fp
    ingest.label(df, action)
    splt = ingest.split(df)
    print(f"Data Split... {len(splt)} segments.")
    return splt

def toDataset(array):
    df = pd.DataFrame()
    for i in range(len(array)):
        df = pd.concat([df, array[i]])
    return df
# Load the files
# Other members of the group did not contribute to the project,
#   therefore the same data was used 3 times over to simulate a complete group

FileWalking_Johnnie = pd.read_csv('./csv/johnnie/Walking Raw Data.csv')
FileJumping_Johnnie = pd.read_csv('./csv/johnnie/Jumping Raw Data.csv')

# Split the data into 5 second clips
walkingData_Johnnie = addFile(FileWalking_Johnnie, ingest.ISWALKING)
jumpingData_Johnnie = addFile(FileJumping_Johnnie, ingest.ISJUMPING)

# Create the training and test data

# Combine the data (3 times to simulate a group)
data_Johnnie =  walkingData_Johnnie + jumpingData_Johnnie
data_Johnnie1 = walkingData_Johnnie + jumpingData_Johnnie
data_Johnnie2 = walkingData_Johnnie + jumpingData_Johnnie

# Combine the data into a single dataset
data = data_Johnnie + data_Johnnie1 + data_Johnnie2

# Split the data into training and test data
trainData, testData = train_test_split(data, test_size=0.1, shuffle=True)


# Save the data to an HDF5 file
with pd.HDFStore('data.hdf5', 'w') as fp:
    fp['Johnnie']  = toDataset(data_Johnnie)
    fp['Johnnie1'] = toDataset(data_Johnnie)
    fp['Johnnie2'] = toDataset(data_Johnnie)
    for i, frame in enumerate(trainData):
        fp[f'dataset/training/data_{i}'] = frame
    for i, frame in enumerate(testData):
        fp[f'dataset/test/data_{i}'] = frame