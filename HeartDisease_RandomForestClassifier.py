import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

dataframe = pd.read_csv("HeartFailure.csv", names=['Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholestoral', 'FastingBloodSugar', 'RestingCardioResults', 'MaxHeartRate',
                                                   'ExerciseInducedPain', 'ST_Dep', 'SlopeOfST','NumOfColoredVessels', 'Thal', 'DiagNum'], dtype= float)

target = 'DiagNum'

X = dataframe.drop([target], axis=1)
Y = dataframe[target]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.35, random_state=1)

'''
sns.countplot(x="DiagNum", data=dataframe)

plt.show()'''

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

forest = RandomForestClassifier(n_estimators = 100, criterion = 'entropy', random_state = 1)
forest.fit(X_train, Y_train)

model = forest
print(model.score(X_train, Y_train))

cm = confusion_matrix(Y_test, model.predict(X_test))
print(cm)

print(classification_report(Y_test.astype(int), model.predict(X_test).astype(int)))
results = model.predict(X_test)
print(results)

train_acc = accuracy_score(Y_train, model.predict(X_train))
test_acc = accuracy_score(Y_test, model.predict(X_test))

print('Train: %.3f Test: %.3f' % (train_acc, test_acc))
