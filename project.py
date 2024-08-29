import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import RocCurveDisplay, confusion_matrix, f1_score, precision_score, recall_score, accuracy_score, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
import features 
import numpy as np
from sklearn import preprocessing
import seaborn as sns
import plots

trainData = []
testData = []

# Read the test and training data from the HDF5 file
with pd.HDFStore('data.hdf5', 'r') as fp:
    data_Johnnie = fp['Johnnie']
    data_Johnnie1 = fp['Johnnie1']
    data_Johnnie2 = fp['Johnnie2']
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
# testData = []
# testData = hdf5.addFile(pd.read_csv('./csv/johnnie/Walking Raw Data.csv'), ingest.ISWALKING)

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
plots.plotContrast(data_Johnnie, data_Johnnie1, title='Raw Data')
plots.plot3(trainData, title='Filtered Training Data')


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
for col in df_train.columns:
    Q1 = df_train[col].quantile(0.25)
    Q3 = df_train[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df_train = df_train[(df_train[col] >= lower_bound) & (df_train[col] <= upper_bound)]
    df_test = df_test[(df_test[col] >= lower_bound) & (df_test[col] <= upper_bound)]


df_train.index = range(len(df_train)) # Restore data indexes
df_test.index = range(len(df_test)) # Restore data indexes
print(f'Length after outlier removal: {len(df_test)} (test)')
print(f'Length after outlier removal: {len(df_train)} (train)')

print(f'test before normalizing {df_test}')
print(f'train before normalizing {df_train}')

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

clf = make_pipeline(preprocessing.StandardScaler(), model)
clf.fit(xTrain, yTrain)

print(f'Predicting {len(testData)} samples...')

# Predict the test data
yTest = np.array(df_test_normalized["label"].values).astype(int)
xTest = np.array(df_test_normalized.iloc[:, 0:11].values)
predict = clf.predict(xTest)
prob = clf.predict_proba(xTest)

# Calculate precision, recall, and F1-score
precision = precision_score(yTest, predict)
recall = recall_score(yTest, predict)
f1 = f1_score(yTest, predict)
accuracy = accuracy_score(yTest, predict)

print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1-score: {f1}")
print(f"Accuracy: {accuracy}")

print(predict)
print(yTest)

# Plot ROC curve
fpr, tpr, _ = roc_curve(yTest, prob[:, 1])
roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr).plot()
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()

# Calculate AUC
auc = roc_auc_score(yTest, prob[:, 1])
print("AUC:", auc)

# Calculate F1 score
f1 = f1_score(yTest, predict)
print("F1 Score:", f1)


# Compute confusion matrix
cm = confusion_matrix(yTest, predict)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.show()