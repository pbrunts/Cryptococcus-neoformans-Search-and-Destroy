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

nestimators=7
features=names
labels=['1','0']

recordedRecall=0
recordedPrecision=0
recordedAccuracy=0
bestF1=0
maxdepth=50
realTP=0
realTN=0
realFP=0
realFN=0

for i in range(5):
    clf=ensemble.RandomForestClassifier(n_estimators = nestimators, max_depth=maxdepth)
    clf=clf.fit(Attributes, Labels)
    
    labels = clf.predict(te[names].copy())
    truth = te['label'].values.tolist()
    seed(time.time())
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
    
    if(f1>bestF1):
        bestF1=f1
        recordedAccuracy=accuracy
        recordedPrecision=precision
        recordedRecall=recall
        realTP=tp
        realTN=tn
        realFP=fp
        realFN=fn
    




#plot_tree(clf)
#dot_data=tree.export_graphviz(clf, out_file=None, feature_names=features, 
#        class_names=labels)
#graph=graphviz.Source(dot_data)
#graph.render("tree")


#labels = [randint(0,1) for i in range(len(truth))]

print(labels)
print("Note: tree creation is not always deterministic.")
print("Number of topics: ", len(names))
print("% of docs that are positive: ", (tp+fn)/(tp+fn+fp+tn))
print("% of docs that are negative: ", (fp+tn)/(tp+fn+fp+tn))

print()
print("tp: ", realTP)
print("fp: ", realFP)
print("tn: ", realTN)
print("fn: ", realFN)
print()
print("precision: ", recordedPrecision)
print("accuracy:  ", recordedAccuracy)
print("recall:    ", recordedRecall)
print("F1:        ", bestF1)
