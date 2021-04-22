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
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, plot_confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
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

# dataframe = pd.read_csv("Binary_HeartFailure.csv", names=['Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholestoral', 'FastingBloodSugar', 'RestingCardioResults', 'MaxHeartRate',
#'ExerciseInducedPain', 'ST_Dep', 'SlopeOfST','NumOfColoredVessels', 'Thal', 'DiagNum'], dtype= float)

'''
dataframe = pd.read_csv('First HD Dataset w Phynet Patients.csv', names = ['Patient_Account_Number', 'Patient_Age', \
                            'Patient_Race', 'Gender', 'Cholesterol', 'BP_Sys', 'BP_Dia', 'BMI', 'Glucose', 'HR', \
                            'RedBloodCellCount', 'RespiratoryRate', 'Sodium', 'WhiteBloodCellCount', 'HD_Prediction'],\
                        dtype=object)

dataframe.drop('Patient_Account_Number', axis=1, inplace = True)
dataframe.drop('Patient_Race', axis=1, inplace=True)
'''

dataframe = pd.read_csv('HeartDiseaseAI_2017-2021_AllDupesRemoved.csv', dtype=object)
dataframe.drop('Patient_Account_Number', axis=1, inplace=True)
race = np.array(dataframe['Patient_Race'].astype('category'))  # Race will have to be encoded separately
dataframe.drop('Patient_Race', axis=1, inplace=True)
#df.drop('Respiratory_Rate', axis=1, inplace=True)
dataframe = dataframe.astype(float)

pd.DataFrame(dataframe)
print(dataframe.describe())
print(dataframe.dtypes)
#print(dataframe.shape)
# print(X.shape)
#print(dataframe.DiagNum.value_counts())
#dataframe = dataframe.dropna()
#dataframe = dataframe.apply(pd.to_numeric, errors='coerce')
#dataframe = dataframe.astype(float)
#dataframe = dataframe.reset_index()

#dataframe.to_csv('csv.csv', index = False)

myEncoder = Encoder()  # Instantiate Encoder()
race_enc = myEncoder.prepare_inputs(race.reshape(-1, 1))  # encoded patient_race will return as type numpy.ndarray
race_series = pd.Series(race_enc.flatten())  # convert numpy object to a series object to replace old column in df
dataframe.insert(2, 'Patient_Race_Encoded', race_series)
# '''
y = np.array(dataframe['CumulativeRiskScore'])
X = np.array(dataframe.drop(['CumulativeRiskScore'], axis=1))

'''Create a bar graph to show count of each value of HD_Prediction'''

dataframe.CumulativeRiskScore.value_counts().plot(kind="bar", color=["green", "blue", "yellow", "brown", "red"])
plt.title("Count of each HD_Prediction")
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
print(continuous_val)
#print(categorical_val)
#rcParams['figure.figsize'] = 20, 14
#plt.matshow(dataframe.corr())
#plt.yticks(np.arange(dataframe.shape[1]), dataframe.columns)
#plt.xticks(np.arange(dataframe.shape[1]), dataframe.columns, rotation=90)
#plt.colorbar()
#plt.show()

dataframe.drop('CumulativeRiskScore', axis=1).corrwith(dataframe.CumulativeRiskScore).plot(kind='bar', grid=True, figsize=(12, 8),
                                                   title="Correlation with target")
plt.show()

categorical_val.remove('CumulativeRiskScore')
dataframe = pd.get_dummies(dataframe, columns=categorical_val)

#print(dataset.head())
dataframe.head().to_csv('dummies.csv', index=False)

X = np.array(dataframe.drop(['CumulativeRiskScore'], 1))

y = np.array(dataframe['CumulativeRiskScore'])

sc = StandardScaler()
X_scaled = sc.fit_transform(X)
# cols_scaled = ['Patient_Age', 'Cholesterol', 'BP_Sys', 'BP_Dia', 'BMI', 'Glucose', 'HR', 'RedBloodCellCount', 'RespiratoryRate', 'Sodium', 'WhiteBloodCellCount']
#dataframe[cols_scaled] = sc.fit_transform(dataframe[cols_scaled])

# X = np.array(dataframe.drop(['HD_Prediction'], 1))

# y = np.array(dataframe['HD_Prediction'])

#norm_X = preprocessing.normalize(X)

print(X.shape)
print(y.shape)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.33, random_state=1)
print(np.any(np.isnan(dataframe))) #and gets False
########################################################################################################################
# Logistic Regression Model

params = {"C": np.logspace(-4, 4, 20),
          "solver": ["liblinear"]}

lr_model = LogisticRegression()

gs_cv = GridSearchCV(lr_model, params, scoring = "accuracy", n_jobs= -1, verbose = 1, cv=5, iid= True)
gs_cv.fit(X_train,y_train)
best_params = gs_cv.best_params_
print(f"Best Parameters: {best_params}")
lr_model = LogisticRegression(**best_params)

lr_model.fit(X_train, y_train)
try:
    print_score(lr_model, X_train, y_train, X_test, y_test, train=True)
    print_score(lr_model, X_train, y_train, X_test, y_test, train=False)
except ValueError:
    pass

test_score=accuracy_score(y_test, lr_model.predict(X_test)) * 100
train_score=accuracy_score(y_train, lr_model.predict(X_train)) * 100

results_data=pd.DataFrame(data=[["Logistic Regression", train_score, test_score]],
                          columns=['Model', 'Training Accuracy %', 'testing Accuracy %'])
results_data.to_csv('Hyper-Tuned_Model_Results.csv', index=False)
########################################################################################################################
# KNN Model
train_score = []
test_score = []
neighbors = range(1, 30)

#norm_train = preprocessing.normalize(X_train)

for k in neighbors:
    knn_model = KNeighborsClassifier(n_neighbors=k)
    knn_model.fit(X_train, y_train)
    train_score.append(accuracy_score(y_train, knn_model.predict(X_train)))
    test_score.append(accuracy_score(y_train, knn_model.predict(X_train)))

plt.figure(figsize=(12, 8))

plt.plot(neighbors, train_score, label="Train Score")
plt.plot(neighbors, test_score, label="Test Score")
plt.xticks(np.arange(1, 21, 1))
plt.xlabel("Number of Neighbors")
plt.ylabel("Model Score")
plt.legend()
plt.show()

print(f"Maximum KNN score on the test data: {max(train_score) * 100:.2f}%")

with open(r'knn_model_stats.csv', 'a') as f:
    writer = csv.writer(f)
    writer.writerow(train_score)
    writer.writerow(test_score)
f.close()

knn_model = KNeighborsClassifier(n_neighbors=27)
knn_model.fit(X_train, y_train)

try:
    print_score(knn_model, X_train, y_train, X_test, y_test, train=True)
    print_score(knn_model, X_train, y_train, X_test, y_test, train=False)
except ValueError:
    pass

test_score = accuracy_score(y_test, knn_model.predict(X_test)) * 100
train_score = accuracy_score(y_train, knn_model.predict(X_train)) * 100

fields = ["KNeighborsClassifier", train_score, test_score]
with open(r'Hyper-Tuned_Model_Results.csv', 'a') as f:
    writer = csv.writer(f)
    writer.writerow(fields)
f.close()
########################################################################################################################
# Support Vector Model
svm_model = SVC(kernel='rbf', gamma=0.1, C=1.0)
'''
params = {"C":(0.1, 0.5, 1, 2, 5, 10, 20),
          "gamma":(0.001, 0.01, 0.1, 0.25, 0.5, 0.75, 1),
          "kernel":(['rbf'])} #poly and linear can be inserted here as well

gs_cv = GridSearchCV(svm_model, params, n_jobs=-1, cv=5, verbose=2, scoring="accuracy")
gs_cv.fit(X_train,y_train)
best_params=gs_cv.best_params_
print(f"Best Parameters For SVM: {best_params}")
'''
svm_model = SVC(kernel ='rbf', gamma=0.001, C=0.1)#**best_params)
svm_model.fit(X_train, y_train)
print(svm_model.predict(X_train))

try:
    print_score(svm_model, X_train, y_train, X_test, y_test, train=True)
    print_score(svm_model, X_train, y_train, X_test, y_test, train=False)
except ValueError:
    pass

test_score = accuracy_score(y_test, svm_model.predict(X_test)) * 100
train_score = accuracy_score(y_train, svm_model.predict(X_train)) * 100
fields = ["SupportVectorMachine", train_score, test_score]
with open(r'Hyper-Tuned_Model_Results.csv', 'a') as f:
    writer = csv.writer(f)
    writer.writerow(fields)
f.close()
########################################################################################################################
# Decision Tree Classifier

params = {"criterion" : ('gini', "entropy"),
          "splitter" : ('best', 'random'),
          "max_depth" : (list(range(1,20))),
          "min_samples_split": [2,3,4],
          "min_samples_leaf" : (list(range(1,20)))}

tree_model=DecisionTreeClassifier(random_state=42)
gs_cv = GridSearchCV(tree_model, params, scoring = "accuracy", n_jobs=-1, verbose=1, cv=3, iid=True)
gs_cv.fit(X_train, y_train)
best_params = gs_cv.best_params_
print(f'Best Parameters for Decision Tree Classifier: {best_params}')

tree_model = DecisionTreeClassifier(**best_params)
tree_model.fit(X_train, y_train)

try:
    print_score(tree_model, X_train, y_train, X_test, y_test, train=True)
    print_score(tree_model, X_train, y_train, X_test, y_test, train=False)
except ValueError:
    pass

test_score = accuracy_score(y_test, tree_model.predict(X_test)) * 100
train_score = accuracy_score(y_train, tree_model.predict(X_train)) * 100
fields = ["Decision Tree Classifier", train_score, test_score]

with open(r'Hyper-Tuned_Model_Results.csv', 'a') as f:
    writer = csv.writer(f)
    writer.writerow(fields)
f.close()
########################################################################################################################
# Random Forest
'''
n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=10)]
max_features = ['auto', 'sqrt']
max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
max_depth.append(None)
min_samples_split = [2,5,10]
min_samples_leaf = [1,2,4]
bootstrap = [True, False]

params_grid = {'n_estimators': n_estimators, 'max_features': max_features,
               'max_depth': max_depth, 'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf, 'bootstrap': bootstrap}

rf_model = RandomForestClassifier(random_state=42)
gs_cv = GridSearchCV(rf_model, params_grid, scoring = "accuracy", cv=3, verbose = 2, n_jobs=-1)
gs_cv.fit(X_train,y_train)
best_params = gs_cv.best_params_
print(f'Best parameters for Random Forest Classifier: {best_params}')

rf_model = RandomForestClassifier(**best_params)
rf_model.fit(X_train, y_train)

try:
    print_score(rf_model, X_train, y_train, X_test, y_test, train = True)
    print_score(rf_model, X_train, y_train, X_test, y_test, train = False)
except ValueError:
    pass

test_score = accuracy_score(y_test, rf_model.predict(X_test)) * 100
train_scaled = accuracy_score(y_train, rf_model.predict(X_train)) * 100
fields = ["RandomForestClassifier", train_score, test_score]

with open(r'Hyper-Tuned_Model_Results.csv', 'a') as f:
    writer = csv.writer(f)
    writer.writerow(fields)
f.close()'''
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
