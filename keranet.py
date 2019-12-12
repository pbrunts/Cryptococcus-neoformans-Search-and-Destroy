#!/usr/bin/env python
# coding: utf-8

#first neural network with keras make predictions
from pandas import read_csv
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt

# load the dataset
dataframe = read_csv('training_stemv2.csv')
dataset = dataframe.values

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

# split into input (X) and output (y) variables
X = dataset[:,0:4].astype(float)
y = dataset[:,4]


#testing data
test_dataframe = read_csv('testing_stemv2.csv')
test_dataset = test_dataframe.values

test_X = test_dataset[:,0:4].astype(float)
test_y = test_dataset[:,4]


# define the keras model
opt = SGD(lr=0.001, momentum=0.9)
model = Sequential()
model.add(Dense(10, input_dim=4, activation='sigmoid'))
#model.add(Dense(3, activation='softmax'))
model.add(Dense(2, activation='softmax'))
model.add(Dense(1, activation='sigmoid'))
# compile the keras model
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
# fit the keras model on the dataset
history = model.fit(X, y, epochs=15, batch_size=10, validation_data=(test_X, test_y), shuffle=True, verbose=1)

tr_predictions = model.predict_classes(X)
print(tr_predictions)
t_predictions = model.predict_classes(test_X)
#print(t_predictions)
print(model.summary())
scores = model.evaluate(test_X, test_y, verbose=0)

print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

print(classification_report(test_y, t_predictions))
print(confusion_matrix(test_y, t_predictions))


print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
plt.savefig('model_accuracy.png')
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
plt.savefig('model_loss.png')



