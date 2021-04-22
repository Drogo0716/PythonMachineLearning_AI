import pickle
import csv
import pandas as pd
import numpy as np
import seaborn as sns
from keras.models import Sequential
from keras.callbacks import ReduceLROnPlateau
from keras.layers import Dense, BatchNormalization, GaussianNoise
from matplotlib import rcParams
from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, plot_confusion_matrix
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
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

def print_score(model, X_train, y_train, X_test, y_test, train=True):
    if train:
        pred = model.predict(X_train)
        clf_report = pd.DataFrame(classification_report(y_train, pred, output_dict=True))
        print("Train Result:\n====================")
        print(f"Accuracy Score: {accuracy_score(y_train, pred) * 100:.2f}%")
        print("--------------------")
        print(f"Classification Report:\n{clf_report}")
        print("--------------------")
        print(f"Confusion Matrix:\n {confusion_matrix(y_train,pred)}\n")

    elif not train:
        pred = model.predict(X_test)
        clf_report = pd.DataFrame(classification_report(y_test, pred, output_dict=True))
        print("Test Results:\n====================")
        print(f"Accuracy Score: {accuracy_score(y_test, pred) * 100:.2f}%")
        print("--------------------")
        print(f"Classification Report:\n{clf_report}")
        print("--------------------")
        print(f"Confusion Matrix:\n {confusion_matrix(y_train,pred)}\n")

########################################################################################################################

#dataframe = pd.read_csv("HeartFailure.csv", names=['Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholestoral', 'FastingBloodSugar', 'RestingCardioResults', 'MaxHeartRate',
#                                                  'ExerciseInducedPain', 'ST_Dep', 'SlopeOfST','NumOfColoredVessels', 'Thal', 'DiagNum'], dtype= float)

dataframe = pd.read_csv('First HD Dataset w Phynet Patients.csv', names = ['Patient_Account_Number', 'Patient_Age', \
                            'Patient_Race', 'Gender', 'Cholesterol', 'BP_Sys', 'BP_Dia', 'BMI', 'Glucose', 'HR', \
                            'RedBloodCellCount', 'RespiratoryRate', 'Sodium', 'WhiteBloodCellCount', 'HD_Prediction'],\
                        dtype=object)

dataframe.drop('Patient_Account_Number', axis=1, inplace = True)
dataframe.drop('Patient_Race', axis=1, inplace=True)

dataframe = dataframe.dropna()
dataframe = dataframe.apply(pd.to_numeric, errors='coerce')
dataframe = dataframe.astype(float)
dataframe = dataframe.reset_index()

dataframe.to_csv('csv.csv', index = False)

pd.DataFrame(dataframe)
#print(dataframe.shape)
# print(X.shape)
#print(dataframe.DiagNum.value_counts())

'''Create a bar graph to show count of each value of DiagNum'''
dataframe.HD_Prediction.value_counts().plot(kind="bar", color=["green", "blue", "yellow", "brown", "red"])
plt.title("Count of each DiagNum")
plt.show()

'''Split data into categorical and continuous values'''
categorical_val = []
continuous_val = []
for column in dataframe.columns:
    print('=========================')
    print(f"{column} : {dataframe[column].unique()}")
    if len(dataframe[column].unique()) <= 10:
        categorical_val.append(column)
    else:
        continuous_val.append(column)

#print(categorical_val)
rcParams['figure.figsize'] = 20, 14
plt.matshow(dataframe.corr())
plt.yticks(np.arange(dataframe.shape[1]), dataframe.columns)
plt.xticks(np.arange(dataframe.shape[1]), dataframe.columns)
plt.colorbar()
plt.show()

dataframe.drop('HD_Prediction', axis=1).corrwith(dataframe.HD_Prediction).plot(kind='bar', grid=True, figsize=(12, 8),
                                                   title="Correlation with target")
plt.show()

categorical_val.remove('HD_Prediction')
dataset = pd.get_dummies(dataframe, columns=categorical_val)

#print(dataset.head())
dataset.head().to_csv('dummies.csv', index=True)

#sc = StandardScaler()
#cols_scaled = ['Age','RestingBP','Cholestoral','MaxHeartRate','ST_Dep']
#dataset[cols_scaled] = sc.fit_transform(dataset[cols_scaled])

X = np.array(dataframe.drop(['HD_Prediction'], 1))

y = np.array(dataframe['HD_Prediction'])

print(X.shape)
print(y.shape)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)

scaler = StandardScaler()
# Fit the scaler by passing in the training data
train_scaled = scaler.fit_transform(X_train)
# Transform the test data the same way
test_scaled = scaler.fit_transform(X_test)


########################################################################################################################
# Logistic Regression Model (Currently Throws an Error)
lr_model = LogisticRegression(solver='liblinear')
lr_model.fit(train_scaled, y_train)
try:
    print_score(lr_model, train_scaled, y_train, test_scaled, y_test, train=True)
    print_score(lr_model, train_scaled, y_train, test_scaled, y_test, train=False)
except ValueError:
    pass

test_score=accuracy_score(y_test, lr_model.predict(X_test)) * 100
train_score=accuracy_score(y_train, lr_model.predict(X_train)) * 100

results_data=pd.DataFrame(data=[["Logistic Regression", train_score, test_score]],
                          columns=['Model', 'Training Accuracy %', 'testing Accuracy %'])
results_data.to_csv('Model_Results.csv', index=False)
########################################################################################################################
# KNN Model
knn_model = KNeighborsClassifier()
knn_model.fit(train_scaled, y_train)
try:
    print_score(knn_model, train_scaled, y_train, test_scaled, y_test, train=True)
    print_score(knn_model, train_scaled, y_train, test_scaled, y_test, train=False)
except ValueError:
    pass

test_score=accuracy_score(y_test, knn_model.predict(test_scaled)) * 100
train_score=accuracy_score(y_train, knn_model.predict(train_scaled)) * 100

fields=["KNeighborsClassifier",train_score,test_score]
with open(r'Model_Results.csv', 'a') as f:
    writer = csv.writer(f)
    writer.writerow(fields)
f.close()
########################################################################################################################
# Support Vector Model
svm_model = SVC(kernel='rbf', gamma=0.1, C=1.0)
svm_model.fit(train_scaled, y_train)

try:
    print_score(svm_model, train_scaled, y_train, test_scaled, y_test, train=True)
    print_score(svm_model, train_scaled, y_train, test_scaled, y_test, train=False)
except ValueError:
    pass

test_score = accuracy_score(y_test, svm_model.predict(test_scaled)) * 100
train_score = accuracy_score(y_train, svm_model.predict(train_scaled)) * 100
fields = ["SupportVectorMachine", train_score, test_score]
with open(r'Model_Results.csv', 'a') as f:
    writer = csv.writer(f)
    writer.writerow(fields)
f.close()
########################################################################################################################
# Decision Tree Classifier
tree_model=DecisionTreeClassifier(random_state=42)
tree_model.fit(train_scaled, y_train)

try:
    print_score(tree_model, train_scaled, y_train, test_scaled, y_test, train=True)
    print_score(tree_model, train_scaled, y_train, test_scaled, y_test, train=False)
except ValueError:
    pass

test_score = accuracy_score(y_test, tree_model.predict(test_scaled)) * 100
train_score = accuracy_score(y_train, tree_model.predict(train_scaled)) * 100
fields = ["Decision Tree Classifier", train_score, test_score]

with open(r'Model_Results.csv', 'a') as f:
    writer = csv.writer(f)
    writer.writerow(fields)
f.close()
########################################################################################################################
# Random Forest
rf_model = RandomForestClassifier(n_estimators=1000, random_state=42)
rf_model.fit(train_scaled, y_train)

try:
    print_score(rf_model, train_scaled, y_train, test_scaled, y_test, train = True)
    print_score(rf_model, train_scaled, y_train, test_scaled, y_test, train = False)
except ValueError:
    pass

test_score = accuracy_score(y_test, rf_model.predict(test_scaled)) * 100
train_score = accuracy_score(y_train, rf_model.predict(train_scaled)) * 100
fields = ["RandomForestClassifier", train_score, test_score]

with open(r'Model_Results.csv', 'a') as f:
    writer = csv.writer(f)
    writer.writerow(fields)
f.close()
########################################################################################################################
#
'''
corr = dataframe.corr()
ax = sns.heatmap(corr,
                 vmin=-1, vmax=1, center=0,
                 cmap=sns.diverging_palette(20,220, n=200),
                 square=True)
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    horizontalalignment='right'
);'''


'''
for i, column in enumerate(categorical_val, 1):
    plt.subplot(3, 3, i)
    dataframe[dataframe["DiagNum"] == 0][column].hist(bins=35, color='green', label='Have Heart Disease = None', alpha=0.6)
    dataframe[dataframe["DiagNum"] == 1][column].hist(bins=35, color='yellow', label='Have Heart Disease = Low Risk', alpha=0.6)
    dataframe[dataframe["DiagNum"] == 2][column].hist(bins=35, color='blue', label='Have Heart Disease = Med Risk', alpha=0.6)
    dataframe[dataframe["DiagNum"] == 3][column].hist(bins=35, color='brown', label='Have Heart Disease = High Risk',alpha=0.6)
    dataframe[dataframe["DiagNum"] == 4][column].hist(bins=35, color='red', label='Have Heart Disease = Has Heart Disease', alpha=0.6)

    plt.legend()
    plt.xlabel(column)
plt.show()
'''
