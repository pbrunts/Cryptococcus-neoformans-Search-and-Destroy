#!/usr/bin/env python
import sys
import os
import pandas as pd
from sklearn import svm
from random import randint
from random import seed
import time
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
import graphviz

if len(sys.argv) < 3:
  print('''
    usage: python ./tree.py TRAINING TESTING
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

clf=svm.SVC(kernel='linear', degree=4)
clf=clf.fit(Attributes, Labels)

features=names
labels=['1','0']

#plot_tree(clf)
#dot_data=tree.export_graphviz(clf, out_file=None, feature_names=features, 
#        class_names=labels)
#graph=graphviz.Source(dot_data)
#graph.render("tree")


labels = clf.predict(te[names])
truth = te['label'].values.tolist()


#X = te[['topic_0', 'topic_1']].values
#y = truth
#plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='bwr')
#red_patch = mpatches.Patch(color='red', label='Positive')
#blue_patch = mpatches.Patch(color='blue', label='Negative')
#plt.legend(handles=[red_patch, blue_patch])
#plt.title('2-Topic SVM Weight Plot')
#plt.xlabel('Weight for Topic 1')
#plt.ylabel('Weight for Topic 2')
#plt.show()


seed(time.time())
#labels = [randint(0,1) for i in range(len(truth))]

tp = 0
tn = 0
fp = 0
fn = 0


for i in range(len(labels)):
  if labels[i] == truth[i]:
    if labels[i] == 1:
      tp = tp + 1
    else:
      tn = tn + 1
  else:
    if labels[i] == 1:
      fp = fp + 1
    else:
      fn = fn + 1

try:
  precision = tp/(tp+fp) 
  accuracy = (tp+tn)/(tp+tn+fp+fn) 
  recall = tp/(tp+fn) 
  f1 = 2 * ((precision * recall) / (precision + recall))
except ZeroDivisionError as e:
  precision = 0
  accuracy = 0
  recall = 0
  f1 = 0

print(labels)
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
