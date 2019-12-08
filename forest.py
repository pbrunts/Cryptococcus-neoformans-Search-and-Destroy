#!/usr/bin/env python
import sys
import os
import pandas as pd
from sklearn import ensemble
from sklearn.tree import plot_tree
import graphviz
from random import randint
from random import seed
import time

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

clf=ensemble.RandomForestClassifier(n_estimators = 25, max_depth=10)
clf=clf.fit(Attributes, Labels)

features=names
labels=['1','0']

#plot_tree(clf)
#dot_data=tree.export_graphviz(clf, out_file=None, feature_names=features, 
#        class_names=labels)
#graph=graphviz.Source(dot_data)
#graph.render("tree")


labels = clf.predict(te[names].copy())
truth = te['label'].values.tolist()

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


precision = tp/(tp+fp) 
accuracy = (tp+tn)/(tp+tn+fp+fn) 
recall = tp/(tp+fn) 
f1 = 2 * ((precision * recall) / (precision + recall))

print(labels)
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
