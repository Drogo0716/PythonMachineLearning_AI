import pickle
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, LabelEncoder
from sklearn.neural_network import MLPClassifier
########################################################################################################################
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
        if self.encoding:
            self.le.fit(y_le)
            y_le = self.le.transform(y_le)
            self.encoding = False
        else:
            y_le = self.le.inverse_transform(y_le)
            self.encoding = True
        return y_le
########################################################################################################################

#dataframe = pd.read_csv("HeartFailure.csv", names=['Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholestoral', 'FastingBloodSugar', 'RestingCardioResults', 'MaxHeartRate',
#                                                   'ExerciseInducedPain', 'ST_Dep', 'SlopeOfST','NumOfColoredVessels', 'Thal', 'DiagNum'], dtype= str)

dataframe = pd.read_csv('First HD Dataset w Phynet Patients.csv', names = ['Patient_Account_Number', 'Patient_Age', \
                            'Patient_Race', 'Gender', 'Cholesterol', 'BP_Sys', 'BP_Dia', 'BMI', 'Glucose', 'HR', \
                            'RedBloodCellCount', 'RespiratoryRate', 'Sodium', 'WhiteBloodCellCount', 'HD_Prediction'],\
                        dtype=object)

dataframe.drop('Patient_Account_Number', axis=1, inplace = True)
dataframe.drop('Patient_Race', axis=1, inplace=True)

'''
print(dataframe.describe())
print(dataframe.shape)
print(dataframe.dtypes)
print(dataframe.head(5))
print(dataframe.dtypes)
print("The current dataframe has been displayed. :]")
'''

#target = 'DiagNum'
target = 'HD_Prediction'

dataframe = dataframe.dropna()
dataframe = dataframe.apply(pd.to_numeric, errors='coerce')
dataframe = dataframe.astype(float)

X = dataframe.drop([target], axis=1)
y = dataframe[target]

myEncoder = Encoder()
X_enc = myEncoder.prepare_inputs(X)
y_enc = myEncoder.prepare_targets(y)

X_train, X_test, y_train, y_test = train_test_split(X_enc, y_enc, test_size=0.35, random_state=123)

scaler = StandardScaler()
# Fit the scaler by passing in the training data
train_scaled = scaler.fit_transform(X_train)
# Transform the test data the same way
test_scaled = scaler.fit_transform(X_test)


model = MLPClassifier(hidden_layer_sizes=(100, 100, 100), random_state=1, max_iter=1000, verbose=1)

model.fit(train_scaled, y_train)

# Output accuracy scores
train_acc = accuracy_score(y_train, model.predict(train_scaled))
test_acc = accuracy_score(y_test, model.predict(test_scaled))
print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))

results = model.predict(test_scaled)
results = myEncoder.prepare_targets(results)

print(results)

print(confusion_matrix(y_test.astype(int), results.astype(int)))
print(classification_report(y_test.astype(int), results.astype(int)))

plt.plot(model.loss_curve_)
plt.show()

'''
# Here is the code to implement the Decision TreeClassifier as the model. This performed worse than MLPClassifier

train_scores, test_scores = list(), list()
values = [i for i in range(1,30)]

# evaluate a decision tree for each depth
for i in values:

    # configure the model
    model = DecisionTreeClassifier(max_depth=i)
    # fit model on the training dataset
    model.fit(train_scaled, y_train)

    # evaluate on the train dataset
    train_yhat = model.predict(train_scaled)
    train_acc = accuracy_score(y_train, train_yhat)
    train_scores.append(train_acc)

    # evaluate on the test dataset
    test_yhat = model.predict(test_scaled)
    test_acc = accuracy_score(y_test, test_yhat)
    test_scores.append(test_acc)

    # summarize progress
    print('>%d, train: %.3f, test: %.3f' % (i, train_acc, test_acc))

# plot of train and test scores vs tree depth
plt.plot(values, train_scores, '-o', label='Train')
plt.plot(values, test_scores, '-o', label = 'Test')
plt.legend()
plt.show()
'''