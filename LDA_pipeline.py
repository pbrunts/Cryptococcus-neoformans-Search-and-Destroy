
import sys
import os
import sklearn.datasets as d

x = d.load_files("./txt_categories")



for y in x.keys():
  print (x[y])


print (x.keys())


