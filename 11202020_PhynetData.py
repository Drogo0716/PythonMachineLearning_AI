import tensorflow as tf
from keras.models import Sequential
from keras.layers import Activation, Dense, BatchNormalization
import numpy as np
import pandas as pd
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.optimizers import SGD
from keras.regularizers import l2
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression, Ridge, LinearRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score, KFold
import pickle
import matplotlib.pyplot as plt

# input data
names = ['PRVDR_NPI','BENE_HIC_NUM','NumEDVisits','BENE_CMSRiskScoreType',
         'BENE_CMSRiskScore','BENE_DOB','BENE_SEX','BENE_RACE']

data = pd.read_csv("CT3.csv", sep=',', names=names)
array = data.values
# print(array)


X = array[:,3:8]
#X=X.astype('int')
Y = array[:,2]
# The following line was needed to correct an error with the datatype. Y was of datatype:object and
# needed to be of datatype:int
Y=Y.astype('int')
normalized_X = preprocessing.normalize(X)

#print(X)
print(Y)
#print(X.shape)
print(Y.shape)
#print(Y.dtypes)


# Split into Train and Test
n_train = 350
trainX, testX = normalized_X[:n_train, :], X[n_train:, :]
trainY, testY = Y[:n_train], Y[n_train:]
print(testY)
model = Sequential()
model.add(Dense(1988, input_dim=5, activation='relu', kernel_regularizer=l2(0.01)))
model.add(BatchNormalization()) # Incorporate BatchNormalization (model becomes more accurate earlier on with BN)
model.add(Dense(1988, activation='relu', kernel_regularizer=l2(0.01)))
model.add(Dense(994, activation='relu', kernel_regularizer=l2(0.01)))
model.add(Dense(497, activation='relu', kernel_regularizer=l2(0.01)))
model.add(Dense(20, activation='relu', kernel_regularizer=l2(0.01)))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss = 'mean_squared_error',
              optimizer = 'adam',
              metrics = ['accuracy'])
# define learning rate schedule
rlrp = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_delta=1E-7, batch_size=350, verbose=1)


print(model.get_weights())

history = model.fit(trainX, trainY, epochs = 500,
                    validation_data = (testX,testY), callbacks=[rlrp])

# evaluate the model
_, train_acc = model.evaluate(trainX, trainY, verbose=0)
_, test_acc = model.evaluate(testX, testY, verbose=0)
print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))

results = model.predict(testX, verbose =1)
print(results)
data_result = pd.DataFrame(results, columns=['results'])

data_result.to_csv('test_results.csv')

# plot loss learning curves
plt.subplot(211)
plt.title('Cross-Entropy Loss', pad=-40)
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
# plot accuracy learning curves
plt.subplot(212)
plt.title('Accuracy', pad=-40)
plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='test')
plt.legend()
plt.show()

'''
model.summary()

#printing out to file
loss_history = history.history["loss"]
numpy_loss_history = np.array(loss_history)
np.savetxt("loss_history.txt", numpy_loss_history,
           delimiter="\n")

binary_accuracy_history = history.history["binary_accuracy"]
numpy_binary_accuracy = np.array(binary_accuracy_history)
np.savetxt("binary_accuracy.txt", numpy_binary_accuracy, delimiter="\n")

print(np.mean(history.history["binary_accuracy"]))

result = model.predict(X).round()
print(result)
'''

'''
# complie model
opt = SGD(lr=0.01, momentum=0.9)
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
# early stopping after 200 epochs
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=200)
# fit model - Batch Gradient Descent: len(trainX), Stochastic Gradient Descent: 1, Minibatch Gradient Descent: more than 1 but less than the number or training examples
history = model.fit(trainX, trainY, validation_data=(testX, testY), epochs=1000, batch_size=len(trainX), verbose=0)

# evaluate model
_, train_acc = model.evaluate(trainX, trainY, verbose=0)
_, test_acc = model.evaluate(testX, testY, verbose=0)
print('Train: %.3f, Test: %3f' % (train_acc, test_acc))
'''