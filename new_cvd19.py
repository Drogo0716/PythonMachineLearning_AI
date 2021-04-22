import pickle
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

dataframe = pd.read_csv('claims2.csv', sep=",")
print(dataframe.dtypes)
le = preprocessing.LabelEncoder()
personId = le.fit_transform(list(dataframe["personId"]))
claimId = le.fit_transform(list(dataframe["claimId"]))
erVisit = le.fit_transform(list(dataframe["erVisit"]))
inpatient = le.fit_transform(list(dataframe["inpatient"]))
admitDate = le.fit_transform(list(dataframe["admitDate"]))
dischargeDate = le.fit_transform(list(dataframe["dischargeDate"]))
dx1 = le.fit_transform(list(dataframe["dx1"]))
dx2 = le.fit_transform(list(dataframe["dx2"]))
dx3 = le.fit_transform(list(dataframe["dx3"]))
dx4 = le.fit_transform(list(dataframe["dx4"]))
dx5 = le.fit_transform(list(dataframe["dx5"]))
dx6 = le.fit_transform(list(dataframe["dx6"]))
dx7 = le.fit_transform(list(dataframe["dx7"]))
dx8 = le.fit_transform(list(dataframe["dx8"]))
dx9 = le.fit_transform(list(dataframe["dx9"]))
dx10 = le.fit_transform(list(dataframe["dx10"]))
dx11 = le.fit_transform(list(dataframe["dx11"]))
dx12 = le.fit_transform(list(dataframe["dx12"]))
dx13 = le.fit_transform(list(dataframe["dx13"]))
gender = le.fit_transform(list(dataframe["gender"]))

X = list(zip(erVisit, inpatient, admitDate, dischargeDate, dx2, dx3, dx4, dx5, dx6, dx7, dx8, dx9, dx10,
             dx11, dx12, dx13))
y = list(zip(dx1))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123)

#print(X)
#print(y)
#print(X_train)
# print(dataframe["dx1"])
# print(y)

scaler = StandardScaler()
# Fit the scaler by passing in the training data
train_scaled = scaler.fit_transform(X_train)

# Transform the test data the same way
test_scaled = scaler.fit_transform(X_test)

model = MLPClassifier(random_state=1, max_iter=500)


# Train model with scaled data and target values
model.fit(train_scaled, y_train)

#scores_train.append(model.score(train_scaled,y_train))

results = model.predict(X_test)
decoded_results = le.inverse_transform(results)
#scores_test.append(model.score(X_test, y_test))
'''
print(results)
data_result = pd.DataFrame(results, columns=['results'])

data_result.to_csv('test_results_cvd19.csv')

# This is one way to output accuracy scores
print(accuracy_score(y_train, model.predict(train_scaled)))
print(accuracy_score(y_test, model.predict(test_scaled)))


# Can also output accuracy score like this
train_acc = accuracy_score(y_train, model.predict(train_scaled))
test_acc = accuracy_score(y_test, model.predict(test_scaled))
print('Train: %.3f, Test: %3f' % (train_acc, test_acc))
'''