import pandas as pd

# Extract features from a dataset
def extract(dataset):
    # Drop NaN values (resulting from the moving average filter)
    # Extract features from the dataset
    chars = pd.DataFrame(columns=['max', 'min', 'mean', 'median', 'X-Y mean', 'X-Y median', 
                               'X-Y skewness', 'X-Y variance', 'Sum-Z mean', 'Sum-Z median',
                               'Sum-Z skewness', 'Sum-Z variance', 'label'])
    chars.loc[0, 'max'] = dataset.iloc[:, 4].max()
    chars.loc[0, 'min'] = dataset.iloc[:, 4].min()
    chars.loc[0, 'mean'] = dataset.iloc[:, 4].mean()
    chars.loc[0, 'median'] = dataset.iloc[:, 4].median()

    # Calculate statistics for X-Y
    chars.loc[0, 'X-Y mean'] = (dataset.iloc[:, 1] - dataset.iloc[:, 2]).mean()
    chars.loc[0, 'X-Y median'] = (dataset.iloc[:, 1] - dataset.iloc[:, 2]).median()
    chars.loc[0, 'X-Y skewness'] = (dataset.iloc[:, 1] - dataset.iloc[:, 2]).skew()
    chars.loc[0, 'X-Y variance'] = (dataset.iloc[:, 1] - dataset.iloc[:, 2]).var()

    # Calculate statistics for Sum-Z
    chars.loc[0, 'Sum-Z mean'] = (dataset.iloc[:, 4] - dataset.iloc[:, 3]).mean()
    chars.loc[0, 'Sum-Z median'] = (dataset.iloc[:, 4] - dataset.iloc[:, 3]).median()
    chars.loc[0, 'Sum-Z skewness'] = (dataset.iloc[:, 4] - dataset.iloc[:, 3]).skew()
    chars.loc[0, 'Sum-Z variance'] = (dataset.iloc[:, 4] - dataset.iloc[:, 3]).var()
    chars.loc[0, 'label'] = dataset.iloc[0, 5]
    return chars
