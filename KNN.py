#!/usr/bin/env python
import sys
import os
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
import graphviz
from sklearn import metrics

kNeighbors=3

if len(sys.argv) < 3:
  print('''
    usage: python ./KNN.py TRAINING TESTING
    TRAINING: file path to csv file of all training objects
    TESTING: path to csv file of all test objects

    both csvs should be output from ldaIsh.py to ensure proper formatting
  ''')
  exit(1)


train = sys.argv[1]
test = sys.argv[2]

tr = pd.read_csv(train)
te = pd.read_csv(test)

names = tr.columns.values.tolist()[1:-1]

Attributes=tr[names]
Labels=tr['label']

#put in KNN model
knn = KNeighborsClassifier(n_neighbors=kNeighbors)
knn.fit(Attributes, Labels)
predLabels=knn.predict(te[names])
features=names
#labels=['1','0']

#labels = clf.predict(te[names].copy())
truth = te['label'].values.tolist()


tp = 0
tn = 0
fp = 0
fn = 0


for i in range(len(predLabels)):
  if predLabels[i] == truth[i]:
    if predLabels[i] == 1:
      tp = tp + 1
    else:
      tn = tn + 1
  else:
    if predLabels[i] == 1:
      fp = fp + 1
    else:
      fn = fn + 1
print(tp)
print(tn)
print(fp)
print(fn)

precision = tp/(tp+fp) 
accuracy = (tp+tn)/(tp+tn+fp+fn) 
recall = tp/(tp+fn) 
f1 = 2 * ((precision * recall) / (precision + recall))


print("Note: tree creation is not always deterministic.")
print("Number of topics: ", len(names))
print("% of docs that are positive: ", (tp+fn)/(tp+fn+fp+tn))
print("% of docs that are negative: ", (fp+tn)/(tp+fn+fp+tn))

print()
print("tp: ", tp)
print("fp: ", fp)
print("tn: ", tn)
print("fn: ", fn)
print()
print("precision: ", precision)
print("accuracy:  ", accuracy)
print("recall:    ", recall)
print("F1:        ", f1)

