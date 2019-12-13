## C. Neoformans Identification using Academic Paper Texts

#### by DeAndre Tomlinson, Emmett Flynn, and Paul Brunts

### Description
This repository contains code for processing and classifying papers along the
lines of whether the underlying studies found evidence of the C. Neoformans
bacteria. These classifications are made by examining the text of the paper,
not by examining the underlying dataset, which is computationally and manually
difficult.

### Pipeline

This repository contains the underlying papers to do appropriate classification.
To run the classification for yourself, first download the repo and run:

`python3 ldaMultipleTopics.py`

This will output two sets of topic weights to the command line in csv form. 
To form your own training an testing files, copy the first of these csv outputs
(it should have roughly 85 documents, or lines) to your training file, and the 
second (roughly 30 documents or lines) to your testing file.

To avoid this step, you can use one of the pre generated datasets. We recommend
that you use the data set in `./csvs/training_stemv2` and 
`./csvs/testing_stemv2`. (These datasets are also hardcoded into the neural 
network code stored in `keranet.py`.

Then choose your classifier. Below are mappings of classifiers to the filename
associated with that classifier. All classifiers are called with the following
convention (with the exception of the neural network) :

`python3 CLASSIFIER PATH_TO_TRAINING_CSV PATH_TO_TESTING_CSV`

For instance, to use the decision tree classifier to train and predict using
the csvs recommended above, the command would be:

`python3 tree.py ./csvs/training_stemv2 ./csvs/testing_stemv2`

Classifer:                File:
SVM                       `svm.py`
KNN                       `KNN.py`
tree                      `tree.py`
forest                    `forest.py`
forest (many iterations)  `forest2.py`


Finally the neural network is called by simply invoking the file:
`python3 keranet.py`


There are many package dependencies, too many to list here, but they include
keras, gensim, and sklearn.

*Optimal Forest2.py parameters -> nestimators = 7, line 48 range should be to 450.
