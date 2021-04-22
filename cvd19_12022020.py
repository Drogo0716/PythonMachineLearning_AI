import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, LabelEncoder
from sklearn.neural_network import MLPClassifier



def load_dataset(filename):
    data = pd.read_csv(filename, skiprows=1, sep=",", names = ['personId','gender','age','claimId','admitDate','dischargeDate','erVisit','inpatient','dx1','dx2',
                                                   'dx3','dx4','dx5','dx6','dx7','dx8','dx9','dx10','dx11','dx12','dx13'])

    target = "dx1"
    # split into X(input) and y(output)
    X = data.drop(['personId','age','claimId', target], axis=1)
    y = data[target]

    # format all fields as a String
    X = X.astype(object)
    # reshape target as 1 column, or 2D array
    y = y.values.reshape((len(y),1))
    return X, y


# prepare input data
def prepare_inputs(X_train, X_test):
    oe = OrdinalEncoder()
    oe.fit(X_train)
    X_train_enc = oe.transform(X_train)
    X_test_enc = oe.transform(X_test)
    return X_train_enc, X_test_enc

# prepare target data
def prepare_targets(y_train, y_test):
    le = LabelEncoder()
    le.fit(y_train)
    y_train_enc = le.transform(y_train)
    y_test_enc = le.transform(y_test)
    return y_train, y_test


X,y = load_dataset("claims2.csv")
# print(X)
# print(y[2])
# print(X.shape)
# print(y.shape)
# print(X.dtypes)
# print(y.dtype)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)

X_train_enc, X_test_enc = prepare_inputs(X_train, X_test)
y_train_enc, y_test_enc = prepare_targets(y_train, y_test)

# define the  model
model = Sequential()
model.add(Dense(10, input_dim=X_train_enc.shape[1], activation='relu', kernel_initializer='he_normal'))
model.add(Dense(1, activation='sigmoid'))
# compile the keras model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit the keras model on the dataset
model.fit(X_train_enc, y_train_enc, epochs=100, batch_size=16, verbose=2)
# evaluate the keras model
_, accuracy = model.evaluate(X_test_enc, y_test_enc, verbose=0)
print('Accuracy: %.2f' % (accuracy*100))
