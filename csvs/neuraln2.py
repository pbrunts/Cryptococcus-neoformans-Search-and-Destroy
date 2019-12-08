#first neural network with keras make predictions
from pandas import read_csv
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
# load the dataset
dataframe = read_csv('training_stemv2.csv')
dataset = dataframe.values
# split into input (X) and output (y) variables
X = dataset[:,0:4].astype(float)
y = dataset[:,4]
# define the keras model
model = Sequential()
model.add(Dense(10, input_dim=4, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
# compile the keras model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit the keras model on the dataset
model.fit(X, y, epochs=15, batch_size=10, verbose=0)

# make class predictions with the model
predictions = model.predict_classes(X)

#trying to get stats
pred_bool = np.argmax(predictions, axis=1)
print(classification_report(y, pred_bool))

# summarize the first 5 cases
for i in range(25):
        print('%s => %d (expected %d)' % (X[i].tolist(), predictions[i], y[i]))

_, accuracy = model.evaluate(X, y)
print('Accuracy (when predictin on itself): %.2f' % (accuracy*100))

#WHERE MY CODE BEGINS

#testing data
test_dataframe = read_csv('testing_stemv2.csv')
test_dataset = test_dataframe.values

test_X = test_dataset[:,0:4].astype(float)
test_y = test_dataset[:,4]

# make class predictions with the model
t_predictions = model.predict_classes(test_X)

# summarize the first 5 cases
for i in range(25):
        print('%s => %d (expected %d)' % (test_X[i].tolist(), t_predictions[i], test_y[i]))

scores = model.evaluate(test_X, test_y, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

#trying to get stats
t_pred_bool = np.argmax(t_predictions, axis=1)
print(classification_report(test_y, t_pred_bool))
print(confusion_matrix(test_y, t_pred_bool))
