import pandas as pd
from pandas.plotting import scatter_matrix
import csv
import numpy as np
import pickle
from matplotlib import pyplot as plt
from matplotlib import rcParams
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OrdinalEncoder
#from PhynetData_MLP import Encoder
from sklearn.neural_network import MLPClassifier

class Encoder (object):

    def __init__(self):
        self.oe = OrdinalEncoder()
        self.le = LabelEncoder()
        self.encoding = True
    # prepare input data
    def prepare_inputs(self, X_oe):
        self.oe.fit(X_oe)
        X_oe = self.oe.transform(X_oe)
        return X_oe


dataframe = pd.read_csv("ct4.csv", names=['PRVDR_NPI','BENE_HIC_NUM','NumEDVisits','BENE_CMSRiskScoreType',
         'BENE_CMSRiskScore','BENE_DOB','BENE_SEX','BENE_RACE'], dtype = str)

prediction = "NumEDVisits"

X = dataframe.drop([prediction], axis=1)
X = dataframe.drop(['PRVDR_NPI', 'BENE_HIC_NUM'], axis=1)
y = dataframe[prediction]

'''
print(dataframe.keys())
attributes = pd.DataFrame(columns=["BENE_CMSRiskScore","BENE_CMSRiskScoreType", "BENE_DOB", "NumEDVisits"])
scatter_matrix(dataframe[attributes], figsize=(12,8))
'''

myEncoder = Encoder()
X_enc = myEncoder.prepare_inputs(X)
#y_enc = myEncoder.prepare_targets(y)
#print(X.head(5))

#print(X_enc)

sc = StandardScaler()
X_scaled = sc.fit_transform(X_enc)

#print(X_scaled)

with open("finalized_phynet_model.pkl", 'rb') as file:
    MLP_model = pickle.load(file)

#model.fit(X_scaled, y_enc)

score = MLP_model.score(X_scaled,y)
print("Test Score: {0:.2f} %".format(100*score))

result_list = []

results = MLP_model.predict(X_scaled)
accuracy = accuracy_score(y,results)
print('Accuracy Score: %3f' % accuracy)
result_list.append(results)
#result_list = np.array(result_list)
#results = myEncoder.prepare_targets(results)
print(results)

with open(r"ReAdmit_Model_Results.txt", "a") as f:
    for x in range(0,len(result_list)):
        f.write(str(result_list[x]))
