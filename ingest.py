import pandas as pd
import h5py as hd
from sklearn.model_selection import train_test_split
import features

ISJUMPING = 1
ISWALKING = 0


# Split the data into 5 second clips
def split(df, segmentLen=5):
    # get the start and end time of the data
    starttime = df.iloc[0:, 0].min()
    endtime = df.iloc[0:, 0].max()
    numSegments = int((endtime - starttime) / segmentLen) + 1
    datasets = []
    for i in range(numSegments):
        start = starttime + i * segmentLen
        end = starttime + (i + 1) * segmentLen
        segment = df[(df.iloc[:, 0] >= start) & (df.iloc[:, 0] < end)]
        datasets.append(segment)
    # Return the data split into segments
    return datasets

def label(df, action):
    df['label'] = action

def loadFile(fp, action):
    df = pd.DataFrame()
    splt = split(fp)
    print(f"Data Split... {len(splt)} segments.")
    for frame in splt:
        transform = features.extract(frame)
        label(transform, action)
        df = pd.concat([df, transform])
    return df