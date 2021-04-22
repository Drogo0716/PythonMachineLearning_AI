import pickle
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, LabelEncoder
from sklearn.neural_network import MLPClassifier
#######################################################################################################################

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

    # prepare target data
    def prepare_targets(self, y_le):
        #self.le.fit(y_le)
        if self.encoding:
            self.le.fit(y_le)
            y_le = self.le.transform(y_le)
            self.encoding = False
        else:
            y_le = self.le.inverse_transform(y_le)
            self.encoding = True
        return y_le

#######################################################################################################################

dataframe = pd.read_csv("ct3.csv", names=['PRVDR_NPI','BENE_HIC_NUM','NumEDVisits','BENE_CMSRiskScoreType',
         'BENE_CMSRiskScore','BENE_DOB','BENE_SEX','BENE_RACE'], dtype = str)


prediction = "NumEDVisits"
# X includes all features except "PRVDR_NPI" and "BENE_HIC_NUM" for the training set
X = dataframe.drop([prediction], axis=1)
#print(X[:8])
#print(X[8:])
X = dataframe.drop(['PRVDR_NPI', 'BENE_HIC_NUM'], axis=1)
# X = X.astype(str)
print(X.dtypes)

y = dataframe[prediction]

# If using this encoding change train_scaled/test_scaled to X_train_enc/X_test_enc
myEncoder = Encoder()

X_enc = myEncoder.prepare_inputs(X)
#y_enc = myEncoder.prepare_targets(y)
#print(y_enc)

X_train, X_test, y_train, y_test = train_test_split(X_enc, y, test_size=0.3, random_state=123)

# print(X_train.shape)
# print(X_test.shape)


scaler = StandardScaler()
# Fit the scaler by passing in the training data
train_scaled = scaler.fit_transform(X_train)
# print(train_scaled)
# Transform the test data the same way
test_scaled = scaler.fit_transform(X_test)
# print(test_scaled)


# Create model
# Default Constructor for MLP Classifier set the params to the following values:
# hidden_layer_sizes = 100, length(# of hidden layers)=1, activation='relu', solver='adam', alpha = 0.0001,
# batch_size='auto'(min(200, n_samples)),
# learning_rate='constant', learning_rate_init=0.001, power_t=0.5,max_iter=200, shuffle=True, random_state = None,
# tol = 1e-4, verbose = false, warm_start = false, momentum=0.9(only used when solver='sgd'),
# nesterovs_momentum=True('sgd' solver only)
# early_stopping =False(Only usable when solver='sgd' OR 'adam'),
# validation_fraction=0.1 (only used when early_stopping=True),
# beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e_8, n_iter_no_change=10, max_fun=15000

# scores_train = []
# scores_test = []

model = MLPClassifier(random_state=1, max_iter=500)
# Adding random_state=1 to the constructor increased test accuracy, increased max_iter to 500 to obtain convergence

# Train model with scaled data and target values
model.fit(train_scaled, y_train)

# scores_train.append(model.score(train_scaled,y_train))

results = model.predict(test_scaled)
#results = myEncoder.prepare_targets(results)
# scores_test.append(model.score(X_test, y_test))

print(results)
data_result = pd.DataFrame(results, columns=['results'])

data_result.to_csv('test_results.csv', index = False)

# Output accuracy scores
train_acc = accuracy_score(y_train, model.predict(train_scaled))
test_acc = accuracy_score(y_test, model.predict(test_scaled))
print('Train: %.3f, Test: %3f' % (train_acc, test_acc))

# Save model to disk
filename = 'finalized_phynet_model.pkl'
with open(filename, 'wb') as file:
    pickle.dump(model, file)

# Some time later...

'''
loaded_model = pickle.load(open(filename, 'rb'))
result = loaded_model.score(test_scaled,y_test)
print(result)
'''


