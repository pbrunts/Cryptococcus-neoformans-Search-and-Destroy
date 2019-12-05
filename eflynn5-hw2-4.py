#!/usr/bin/env python

import pandas as pd
from sklearn import tree
import graphviz
D=pd.read_csv("data-film.csv")

Attributes=D[['AVGRATING_WEBSITE_1','AVGRATING_WEBSITE_2', 'AVGRATING_WEBSITE_3', 'AVGRATING_WEBSITE_4' ]]
Labels=D['GENRE']

clf=tree.DecisionTreeClassifier()
clf=clf.fit(Attributes, Labels)

features=['AVGRATING_WEBSITE_1','AVGRATING_WEBSITE_2', 'AVGRATING_WEBSITE_3', 'AVGRATING_WEBSITE_4']
labels=['ACTION', 'ROMANCE','COMEDY']

tree.plot_tree(clf)
dot_data=tree.export_graphviz(clf, out_file=None, feature_names=features, 
        class_names=labels)
graph=graphviz.Source(dot_data)
graph.render("tree")

