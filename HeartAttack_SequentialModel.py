import pickle
import csv
import pandas as pd
import numpy as np
from matplotlib import rcParams
from keras.models import Sequential
from keras.callbacks import ReduceLROnPlateau
from keras.layers import Dense, BatchNormalization, GaussianNoise
from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, LabelEncoder
from sklearn.neural_network import MLPClassifier
########################################################################################################################
'''class Encoder (object):

    def __init__(self):
        self.oe = OrdinalEncoder()
        self.le = LabelEncoder()
        self.encoding = True
    # prepare input data
    def prepare_inputs(self, X_oe):
        self.oe.fit(X_oe)
        X_oe = self.oe.transform(X_oe)
        return X_oe

    # prepare target data
    def prepare_targets(self, y_le):
        if self.encoding:
            self.le.fit(y_le)
            y_le = self.le.transform(y_le)
            self.encoding = False
        else:
            y_le = self.le.inverse_transform(y_le)
            self.encoding = True
        return y_le'''
########################################################################################################################

#dataframe = pd.read_csv("Binary_HeartFailure.csv", names=['Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholestoral', 'FastingBloodSugar', 'RestingCardioResults', 'MaxHeartRate',
#                                                  'ExerciseInducedPain', 'ST_Dep', 'SlopeOfST','NumOfColoredVessels', 'Thal', 'DiagNum'], dtype= float)

dataframe = pd.read_csv('First HD Dataset w Phynet Patients.csv', names = ['Patient_Account_Number', 'Patient_Age', \
                            'Patient_Race', 'Gender', 'Cholesterol', 'BP_Sys', 'BP_Dia', 'BMI', 'Glucose', 'HR', \
                            'RedBloodCellCount', 'RespiratoryRate', 'Sodium', 'WhiteBloodCellCount', 'HD_Prediction'],\
                        dtype=object)

print(dataframe.shape)
dataframe.drop('Patient_Account_Number', axis=1, inplace = True)
dataframe.drop('Patient_Race', axis=1, inplace=True)
print(dataframe.shape)

dataframe = dataframe.dropna()
dataframe = dataframe.apply(pd.to_numeric, errors='coerce')
dataframe = dataframe.astype(float)

#target = 'DiagNum'
target = 'HD_Prediction'

dataframe.HD_Prediction.value_counts().plot(kind='bar', color=['green', 'red'])
plt.title("Value Count of Heart Disease")
plt.show()

rcParams['figure.figsize'] = 20, 14
plt.matshow(dataframe.corr())
plt.yticks(np.arange(dataframe.shape[1]), dataframe.columns)
plt.xticks(np.arange(dataframe.shape[1]), dataframe.columns, rotation=90)
plt.colorbar()
plt.show()

X = np.array(dataframe.drop([target], 1))
y = np.array(dataframe[target])
print(X.shape)
'''
myEncoder = Encoder()
myEncoder.prepare_inputs(X)
myEncoder.prepare_targets(y)
'''

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.3, random_state=1)
print(X_train.shape)
model = Sequential()
model.add(BatchNormalization()) #Incorporate BatchNormalization (model becomes more accurate earlier on with BN)
model.add(Dense(200, input_dim=13, activation='relu'))
model.add(Dense(150, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(
    optimizer='adam',
    loss='mse',
    metrics=['accuracy'])

# fit the model using the training data set
history = model.fit(X_train, y_train, epochs=500, validation_data=(X_val, y_val))

#######################################
# for key in history.history.keys():  #
#     print(key)                      #
#######################################
#results = myEncoder.prepare_targets(y_test)

predictions = model.predict(X_test)

#print(predictions)

#data_result = pd.DataFrame(predictions, columns=['DiagNum'])
#data_result.to_csv('DiagNum_results.csv', index=False)

data_result = pd.DataFrame(predictions, columns=['HD_Prediction'])
data_result.to_csv('HDPrediction_results.csv', index=False)

r = csv.reader(open('HDPrediction_results.csv'))
values = data_result.to_numpy()

#print(values)

for i in range(len(values)):
    if values[i] < 0.1:
        values[i] = 0
    elif 0.1 < values[i] < 0.25:
        values[i] = 1
    elif 0.25 < values[i] < 0.50:
        values[i] = 2
    elif 0.50 < values[i] < 0.75:
        values[i] = 3
    elif 0.75 < values[i]:
        values[i] = 4
    else:
        print('\n')

data_result = pd.DataFrame(values, columns=['RiskNum'])
data_result.to_csv('RiskNum_results.csv', index=False)

# evaluate model
_, train_acc = model.evaluate(X_train, y_train, verbose=0)
_, val_acc = model.evaluate(X_test, y_test, verbose=0)
print('Train: %.3f, Validation: %3f' % (train_acc, val_acc))

'''
loss_train = history.history['loss']
loss_val = history.history['val_loss']
epochs = range(1,501)
plt.plot(epochs, loss_train, 'g', label='Training Loss')
plt.plot(epochs, loss_val, 'b', label = 'Validation loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

acc_train = history.history['accuracy']
acc_val = history.history['val_accuracy']
epochs = range(1,501)
plt.plot(epochs, loss_train, 'g', label='Training Accuracy')
plt.plot(epochs, loss_val, 'b', label = 'Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
'''