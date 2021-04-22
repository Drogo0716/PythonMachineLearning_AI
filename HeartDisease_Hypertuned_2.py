import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

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
        print(f"Confusion Matrix:\n {confusion_matrix(y_test,pred)}\n")

########################################################################################################################


dataframe = pd.read_csv("Binary_HeartFailure.csv", names=['Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholestoral', 'FastingBloodSugar', 'RestingCardioResults', 'MaxHeartRate',
                                                  'ExerciseInducedPain', 'ST_Dep', 'SlopeOfST','NumOfColoredVessels', 'Thal', 'DiagNum'], dtype= float)
dataframe.info()
print(dataframe.describe())

rcParams['figure.figsize'] = 20, 14
plt.matshow(dataframe.corr())
plt.yticks(np.arange(dataframe.shape[1]), dataframe.columns)
plt.xticks(np.arange(dataframe.shape[1]), dataframe.columns, rotation=90)
plt.colorbar()
#plt.title('Correlation Martix of Heart Disease Data')
plt.show()

rcParams['figure.figsize'] = 20,14
plt.bar(dataframe['DiagNum'].unique(), dataframe['DiagNum'].value_counts(), color = ['green','red'])
plt.xticks([0,1])
plt.xlabel('Target Classes')
plt.ylabel('Count')
plt.title('Count of each Target Class')
plt.show()

dataframe.hist()
plt.show()

dataframe.plot(kind='scatter', x='Age', y='RestingBP', alpha=0.1)
plt.show()

dataframe.plot(kind='scatter', x='Age', y='DiagNum')
plt.show()

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


dataframe = pd.get_dummies(dataframe, columns = ['Sex', 'ChestPainType', 'FastingBloodSugar', 'RestingCardioResults',
                                                 'ExerciseInducedPain', 'SlopeOfST', 'NumOfColoredVessels', 'Thal'])

sc = StandardScaler()
cols_scaled = ['Age','RestingBP','Cholestoral','MaxHeartRate','ST_Dep']
dataframe[cols_scaled] = sc.fit_transform(dataframe[cols_scaled])

#print(dataframe)

X = dataframe.drop(['DiagNum'], axis=1)
y = dataframe['DiagNum']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)

#######################################################################################################################
knn_scores = []
# Best Results When K is 3(currently)
for k in range(1,21):
    knn_classifier = KNeighborsClassifier(n_neighbors = k)
    knn_classifier.fit(X_train,y_train)
    knn_scores.append(knn_classifier.score(X_test, y_test))

plt.plot([k for k in range(1,21)], knn_scores, color='red')
for i in range(1,21):
    plt.text(i,knn_scores[i-1], (i, knn_scores[i-1]))
plt.xticks([i for i in range(1,21)])
plt.xlabel('Number of Neighbors (K)')
plt.ylabel('Scores')
plt.title('K Neighbors Classifier scores for Different K Values')
plt.show()
#######################################################################################################################
svc_scores = []
kernels = ['linear','poly','rbf','sigmoid']
# gamma = [0.001, 0.01, 0.1, 0.25, 0.5, 0.75, 1]
# C = [0.1, 0.5, 1, 2, 5, 10, 20]
for i in range(len(kernels)):
    #for j in range(len(gamma)):
        #for k in range(len(C)):
    svc_model = SVC(kernel=kernels[i], gamma=0.001, C=0.1)
    svc_model.fit(X_train, y_train)
    svc_scores.append(svc_model.score(X_test,y_test))

plt.bar(kernels, svc_scores, color = ['red','green','blue','yellow'])
for i in range(len(kernels)):
    plt.text(i, svc_scores[i],svc_scores[i])
plt.xlabel('Kernels')
plt.ylabel('Scores')
plt.title('Support Vector Classifier Scores for Different Kernels')
plt.show()

# for i in svc_scores:
    # print(i)

#######################################################################################################################

dt_scores = []
for i in range(1, len(X.columns) +1):
    dt_model = DecisionTreeClassifier(max_features = i, random_state = 0)
    dt_model.fit(X_train, y_train)
    dt_scores.append(dt_model.score(X_test, y_test))

plt.plot([i for i in range(1, len(X.columns) + 1)], dt_scores, color = 'blue')
for i in range(1, len(X.columns) + 1):
    plt.text(i, dt_scores[i-1], (i, dt_scores[i-1]))
plt.xticks([i for i in range(1, len(X.columns) + 1)])
plt.xlabel('Max Features')
plt.ylabel('Scores')
plt.title('Decision Tree Classifier Scores for Different number of Maximum Features')
plt.show()

#######################################################################################################################
'''
rf_scores = []
estimators = [10, 100, 200, 500, 1000]
for i in estimators:
    rf_model = RandomForestClassifier(n_estimators = i, random_state=0)
    rf_model.fit(X_train,y_train)
    rf_scores.append(rf_model.score(X_test,y_test))
plt.bar([i for i in range(len(estimators))], rf_scores, color=['purple','green','blue','yellow','red'])
for i in range(len(estimators)):
    plt.text(i, rf_scores[i], rf_scores[i])
plt.xticks(ticks = [i for i in range(len(estimators))], labels=[str(estimator) for estimator in estimators])
plt.xlabel('Number of Estimators')
plt.ylabel('Scores')
plt.title('Random Forest Classifier Scores for Different Number of Estimators')
plt.show()
'''
#######################################################################################################################

