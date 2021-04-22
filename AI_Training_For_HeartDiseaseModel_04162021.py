import pickle
import sys, logging
import csv
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import rcParams
from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, LabelEncoder, RobustScaler, MinMaxScaler
from sklearn.svm import SVC

########################################################################################################################
logging.basicConfig(filename='debugLog_HeartFailureAI_04012020.txt',
                    level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

training = True
splitting = False
saving = False

pd.set_option('display.max_columns', None)  # set this number to >= your number of cols
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)


########################################################################################################################

class Encoder(object):

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
        # print(f"Classification Report:\n{clf_report}")  # display classification report
        # print("--------------------")
        # print(f"Confusion Matrix:\n {confusion_matrix(y_train, pred)}\n")  # display confusion matrix

        logging.debug(f'Result of training prediction: {pd.DataFrame(pred)}\n\n')
        logging.debug(f'\n{type(pred)}')
        logging.debug(f'Result of training accuracy score: {accuracy_score(y_train, pred) * 100:.2f}%\n\n')
        # logging.debug(f'Classification Report output for training data: {clf_report}\n\n')
        # logging.debug(f'Confusion Matrix output for training data: {confusion_matrix(y_train,pred)}\n\n')

    elif not train:  # When not training, pass the test data to sklearn.metrics functions instead.
        pred = model.predict(X_test)
        clf_report = pd.DataFrame(classification_report(y_test, pred, output_dict=True))
        print("Test Results:\n====================")
        print(f"Accuracy Score: {accuracy_score(y_test, pred) * 100:.2f}%")
        print("--------------------")
        # print(f"Classification Report:\n{clf_report}")
        # print("--------------------")
        # print(f"Confusion Matrix:\n {confusion_matrix(y_test, pred)}\n")

        logging.debug(f'Result of testing prediction: {pd.DataFrame(pred)}\n\n')
        logging.debug(f'Result of testing accuracy score: {accuracy_score(y_train, pred) * 100:.2f}%\n\n')
        # logging.debug(f'Classification Report output for testing data: {clf_report}\n\n')
        # logging.debug(f'Confusion Matrix output for testing data: {confusion_matrix(y_train, pred)}\n\n')


########################################################################################################################


if splitting:
    '''DATASET IS INSTANTIATED BELOW'''

    df = pd.read_csv("Dataset2_Distinct.csv", dtype=object)

    df1 = df.iloc[:8000, :]
    df2 = df.iloc[8001:, :]

    df1.to_csv('Dataset2_Distinct_Model.csv', index=False)
    df2.to_csv('Dataset2_Distinct_TrueTest.csv', index=False)
    sys.exit()

else:
    df = pd.read_csv("Dataset2_Distinct_Model.csv", dtype=object)

    df.drop('Patient_Account_Number', axis=1, inplace=True)
    df.drop('Encounter_Date', axis=1, inplace=True)
    df.drop('Diagnosis_Name', axis=1, inplace=True)

    race = np.array(df['Patient_Race'].astype('category'))  # Race will have to be encoded separately
    df.drop('Patient_Race', axis=1, inplace=True)

    df = df.astype(float)

    # df.reset_index()
    df = df.dropna()
    df = df.apply(pd.to_numeric, errors='coerce')

    print(np.any(np.isnan(df)))
    print(np.all(np.isfinite(df)))
    # sys.exit()

    myEncoder = Encoder()  # Instantiate Encoder()
    logging.debug(f'\n\n{race}\n\n')
    race_enc = myEncoder.prepare_inputs(race.reshape(-1, 1))  # encoded patient_race will return as type numpy.ndarray
    logging.debug(f'\n\n{race_enc}\n\n')
    race_series = pd.Series(race_enc.flatten())  # convert numpy object to a series object to replace old column in df
    logging.debug(f'\n\n{race_series}\n\n')
    df.insert(2, 'Patient_Race_Encoded', race_series)

    y = np.array(df['Risk_Score'])
    X = np.array(df.drop(['Risk_Score'], axis=1))

    # For KNN Model use MinMaxScaler, for SVM Model use Standard Scaler

    scaler = StandardScaler()
    # scaler = MinMaxScaler()
    # scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    # All scalers have the inverse_transform() method to scale the data back to original values if needed

    # logging.debug(f'Number of columns in dataset: {len(df.columns)}')
    # logging.debug(f'Number of rows in dataset: {len(df.index)}')
    # logging.debug(f'\n\nDescription of DataFrame:\n\n {df.describe()}')
    # logging.debug(f'Here is the dataframe column after encoding/scaling:\n\n{pd.DataFrame(X_scaled)}\n\n')

########################################################################################################################
    # VISUALIZATION PRIOR TO SCALING INPUT DATA
    df = df.astype(float)
########################################################################################################################

    df.Risk_Score.value_counts().plot(kind='bar', color=['green', 'red', 'purple', 'blue', 'black'])
    plt.title("Value Count of Each Risk Level")
    plt.show()

########################################################################################################################
# CORRELATION MATRIX
    rcParams['figure.figsize'] = 40, 40
    plt.matshow(df.corr())
    plt.yticks(np.arange(df.shape[1]), df.columns)
    plt.xticks(np.arange(df.shape[1]), df.columns, rotation=90)
    plt.colorbar()
    plt.show()

    # CORRELATION TO TARGET BAR GRAPH
    df.drop('Risk_Score', axis=1).corrwith(df.Risk_Score).plot(kind='bar', grid=True, figsize=(12, 8),
                                                               title="Correlation with Risk")
    plt.show()

########################################################################################################################
# TRAIN TEST SPLIT##
####################
    print(X.shape)
    print(y.shape)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.25, random_state=1)

    print(type(X_train))
    print(type(X_test))

    # logging.debug(f'Here is the training data after scaling:\n\n{pd.DataFrame(X_train)}\n\n')
    # logging.debug(f'Here is the testing data after scaling:\n\n{pd.DataFrame(X_test)}\n\n')

    # sys.exit()

########################################################################################################################
# KNN Model
    print("\nTHE FOLLOWING RESULTS ARE FOR THE KNN MODEL:\n")
    if training:
        knn_model = KNeighborsClassifier(n_neighbors=3, leaf_size=1, p=1, weights='uniform')
        knn_model.fit(X_train, y_train)  # replaced train_scaled with X_train
        try:
            print_score(knn_model, X_train, y_train, X_test, y_test, train=True)  # replaced train_scaled with X_train
            print_score(knn_model, X_train, y_train, X_test, y_test, train=False)
        except ValueError:
            pass

        test_score = accuracy_score(y_test, knn_model.predict(X_test)) * 100
        train_score = accuracy_score(y_train, knn_model.predict(X_train)) * 100  # replaced train_scaled with X_train

        # print(KNeighborsClassifier.kneighbors(knn_model))

        fields = ["KNeighborsClassifier", train_score, test_score]
        with open(r'eBO_HD_RiskScore_Results_04152021.csv', 'a') as f:
            writer = csv.writer(f)
            writer.writerow(fields)
        f.close()
########################################################################################################################
# SVM Model
        svm_model = SVC(kernel='rbf', gamma=0.25, C=2)  # **best_params)
        svm_model.fit(X_train, y_train)
        # print(svm_model.predict(X_train))
        print("\nTHE FOLLOWING RESULTS ARE FOR THE SVM MODEL:\n")
        try:
            print_score(svm_model, X_train, y_train, X_test, y_test, train=True)
            print_score(svm_model, X_train, y_train, X_test, y_test, train=False)
        except ValueError:
            pass

        test_score = accuracy_score(y_test, svm_model.predict(X_test)) * 100
        train_score = accuracy_score(y_train, svm_model.predict(X_train)) * 100
        fields = ["SupportVectorMachine", train_score, test_score]
        with open(r'HeartDisease_SVM_Model_Results_04152021.csv', 'a') as f:
            writer = csv.writer(f)
            writer.writerow(fields)
        f.close()

########################################################################################################################
    else:
        knn_grid_params = {'leaf_size': list(range(1, 10)),
                           'weights': ['uniform', 'distance'],
                           'n_neighbors': list(range(1, 30)),
                           'p': [1, 2]}

        knn = KNeighborsClassifier()
        clf = GridSearchCV(knn, knn_grid_params, cv=3, verbose=1, n_jobs=-1)
        clf_results = clf.fit(X_train, y_train)

        logging.debug(f"Here is the GridSearchCV results for KNN:\n\n{clf}")

        print(f"Best Score: {clf_results.best_score_}")
        print(f"Best p: {clf_results.best_estimator_}")
        print(f"Best n_neighbors: {clf_results.best_params_}")

        # Implement a graph the visualize the optimal K value of KNN
        k_range = range(1, 31)
        k_scores = []

        for k in k_range:
            # run model for each K
            knn = KNeighborsClassifier(n_neighbors=k, p=1, leaf_size=1)
            # Obtain the cross-validation scores and append them to the list
            scores = cross_val_score(knn, X_train, y_train, cv=3, scoring='accuracy')
            k_scores.append(scores.mean())

        logging.info(k_scores)

        plt.plot(k_range, k_scores)
        plt.xlabel('Value of K for KNN')
        plt.ylabel('Cross-Validation Accuracy')
        plt.show()
########################################################################################################################
# Support Vector Model

        svm_model = SVC(kernel='rbf', gamma=0.25, C=1.0)
        params = {"C": (0.1, 0.5, 1, 2, 5, 10, 20),
                  "gamma": (0.001, 0.01, 0.1, 0.25, 0.5, 0.75, 1),
                  "kernel": (['rbf'])}  # poly and linear can be inserted here as well

        gs_cv = GridSearchCV(svm_model, params, n_jobs=-1, cv=5, verbose=1, scoring="accuracy")
        gs_cv.fit(X_train, y_train)
        best_params = gs_cv.best_params_
        print(f"Best Parameters For SVM: {best_params}")

########################################################################################################################
if saving and training:
    models = [knn_model, svm_model]
    # Save model to disk
    filename = 'xxx.pkl'
    with open(filename, 'wb') as file:
        pickle.dump(models[0], file)

    '''Split data into categorical and continuous values'''
    '''
    categorical_val = []
    continuous_val = []
    for column in dataframe.columns:
        print('=========================')
        print(f"{column} : {dataframe[column].unique()}")
        if len(dataframe[column].unique()) <= 10:
            categorical_val.append(column)
        else:
            continuous_val.append(column)
    
    categorical_val.remove('HD_Prediction')
    dataset = pd.get_dummies(dataframe, columns=categorical_val)
    
    dataset.to_csv('dummiesFile_02282021.csv', index = True)
    
    sc = StandardScaler()
    cols_scaled = ['Age','RestingBP','Cholestoral','MaxHeartRate','ST_Dep']
    dataset[cols_scaled] = sc.fit_transform(dataset[cols_scaled])
    '''
