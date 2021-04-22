# example of batch normalization, WeightRegularization, and GaussianNoise being used to train model
# for a binary classification problem
from sklearn.datasets import make_circles
from keras.layers import Dense, BatchNormalization, GaussianNoise
from keras.models import Sequential
from keras.optimizers import SGD
from keras.regularizers import l2
from keras.callbacks import EarlyStopping
from matplotlib import pyplot as plt

# generate the dataset
X, y = make_circles(n_samples=1000, noise=0.1, random_state=1)

# Split into Train and Test
n_train = 500
trainX, testX = X[:n_train, :], X[n_train:, :]
trainy, testy = y[:n_train], y[n_train:]

print(X)
print(y)
print(trainX.shape)
print(trainy.shape)

#define model
model = Sequential()
#model.add(Dense(50, input_dim=2, activation='relu'))
# Comment out below line and uncomment above line if not using WeightReg.
model.add(Dense(500, input_dim=2, activation='relu', kernel_regularizer=l2(0.01)))
model.add(BatchNormalization()) # Incorporate BatchNormalization (model becomes more accurate earlier on with BN)
#model.add(GaussianNoise(0.1))
model.add(Dense(1, activation='sigmoid'))
'''
print('Here is the X axis of the dataset : ', X)
print('Here is the y axis of the dataset : ', y)
print(trainX)
print(testX)
print(trainy)
print(testy)'''

# complie model
opt = SGD(lr=0.01, momentum=0.9)
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
# early stopping after 200 epochs
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=200)
# fit model - Batch Gradient Descent: len(trainX), Stochastic Gradient Descent: 1, Minibatch Gradient Descent: more than 1 but less than the number or training examples
history = model.fit(trainX, trainy, validation_data=(testX, testy), epochs=1000, batch_size=len(trainX), verbose=0)

# evaluate model
_, train_acc = model.evaluate(trainX, trainy, verbose=0)
_, test_acc = model.evaluate(testX, testy, verbose=0)
print('Train: %.3f, Test: %3f' % (train_acc, test_acc))

# plot loss learning curves
plt.subplot(211)
plt.title('Cross-Entropy Loss', pad=40)
plt.plot(history.history['loss'], label = 'train')
plt.plot(history.history['val_loss'], label = 'test')

# plot accuracy learning curves
plt.subplot(212)
plt.title('Accuracy', pad=40)
plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='test')
plt.legend()
plt.show()
