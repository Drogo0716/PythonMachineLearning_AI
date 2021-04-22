import pandas as pd
import sys
import pprint
from pandas.plotting import scatter_matrix
import csv
import numpy as np
import pickle
from matplotlib import pyplot as plt
from matplotlib import rcParams
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, plot_confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OrdinalEncoder, MinMaxScaler


#######################################################################################################################


class Encoder (object):

    def __init__(self):  # Constructor for Encoder class
        self.oe = OrdinalEncoder()
        self.le = LabelEncoder()
        self.encoding = True

    def prepare_inputs(self, X_oe):  # prepare input data
        self.oe.fit(X_oe)
        X_oe = self.oe.transform(X_oe)
        return X_oe

    def prepare_targets(self, y_le):  # prepare target data
        if self.encoding:
            self.le.fit(y_le)
            y_le = self.le.transform(y_le)
            self.encoding = False
        else:
            y_le = self.le.inverse_transform(y_le)
            self.encoding = True
        return y_le

########################################################################################################################
########################################################################################################################
# classification_report, accuracy_score, and confusion_matrix are imported function from the sklearn.metrics module
########################################################################################################################


def print_score(model, X_train, y_train, X_test, y_test, train=True):
    if train:  # When training model...
        pred = model.predict(X_train)  # Pass the scaled input data to the model's predict function
        clf_report = pd.DataFrame(classification_report(y_train, pred, output_dict=True))  # generate class. report
        print("Train Result:\n====================")
        print(f"Accuracy Score: {accuracy_score(y_train, pred) * 100:.2f}%")  # Use accuracy score function
        print("--------------------")
        print(f"Classification Report:\n{clf_report}") # display classification report
        print("--------------------")
        print(f"Confusion Matrix:\n {confusion_matrix(y_train,pred)}\n")  # display confusion matrix

    elif not train:  # When not training, pass the test data to sklearn.metrics functions instead.
        pred = model.predict(X_test)
        clf_report = pd.DataFrame(classification_report(y_test, pred, output_dict=True))
        print("Test Results:\n====================")
        print(f"Accuracy Score: {accuracy_score(y_test, pred) * 100:.2f}%")
        print("--------------------")
        print(f"Classification Report:\n{clf_report}")
        print("--------------------")
        print(f"Confusion Matrix:\n {confusion_matrix(y_train,pred)}\n")

########################################################################################################################
########################################################################################################################


df = pd.read_csv("Dataset2_Distinct_TrueTest.csv", dtype=object)
race = np.array(df['Patient_Race'].astype('category'))  # Race will have to be encoded separately
#  print(len(df['Patient_Race']))

df.drop('Patient_Account_Number', axis=1, inplace=True)
df.drop('Risk_Score', axis=1, inplace=True)
df.drop('Encounter_Date', axis=1, inplace=True)
df.drop('Diagnosis_Name', axis=1, inplace=True)
df.drop('Patient_Race', axis=1, inplace=True)

myEncoder = Encoder()
race_enc = myEncoder.prepare_inputs(race.reshape(-1, 1))  # encoded patient_race will return as type numpy.ndarray
race_series = pd.Series(race_enc.flatten())  # convert numpy object to a series object to replace old column in df
df.insert(2, 'Patient_Race_Encoded', race_series)

df = df.astype(float)
df = df.apply(pd.to_numeric, errors='coerce')
df.dropna(inplace=True)
df.to_csv("Dataset2_Distinct_TrueTest_PostDrop_na.csv", index=False)

X = np.array(df)


scaler = StandardScaler()
#  scaler = MinMaxScaler()
vars_Scaled = scaler.fit_transform(X)

print(vars_Scaled)
#sys.exit()

with open("HeartDisease_Risk_SVM_Model_04212021.pkl", 'rb') as file:
    model = pickle.load(file)

results_list = []

results = model.predict(vars_Scaled)

results_list.append(results)
print(type(results))

df2 = pd.DataFrame(results, columns=['Risk_Score'])
df2.to_csv('AI_Risk_Score_04212021.csv', index=False)

print(df2)
print(type(df2))

'''
with open(r'HD_Model_Results.txt', 'a') as f:
    for x in range(0,len(results_list)):
        f.write(str(results_list[x]))

for i in df2.iterrows():
    print(i)
    print(type(i))
'''



