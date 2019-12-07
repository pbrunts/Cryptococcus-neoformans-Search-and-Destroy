#!/usr/bin/env python

#LDA implementation using gensim
#none of this will have been tested as of 10/14, but 
#as soon as we have the documents parsed and 
#ready to go we will test it out

#note: much of the code is derived from the pipeline
#of priya dwivedi https://github.com/priya-dwivedi/Deep-Learning/blob/master/topic_modeling/LDA_Newsgroup.ipynb

import re
import numpy as np
import pandas as pd
from pprint import pprint
#gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
#spacy
import spacy
#plotting
#import pyLDAvis
#import pyLDAvis.gensim
#import matplotlib.pyplot as plt
#os
import os
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)

import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
import numpy as np

import nltk
nltk.download('stopwords') #the fuck does this do?
#stop words
from nltk.corpus import stopwords
stop_words=stopwords.words('english')
stop_words.extend(['from', 'subject', 're', 'edu', 'use', 'http', 'https',
                     
                  ])

#Functions:
def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))

def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

def make_bigrams(texts):
    return [bigram_mod[doc] for doc in texts]

def make_trigrams(texts):
    return [trigram_mod[bigram_mod[doc]] for doc in texts]


def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """https://spacy.io/api/annotation"""
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent))
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out

def stemWords(texts):
    ps = PorterStemmer()


#!!data is trainingData!!
data=[]
testingData=[]
trainingNameList=[]
testingNameList=[]
positiveNameList=[]
trainingLabelList=[]
testingLabelList=[]
kTopics=4
#let's load in the names of the files as names,
#and see if it matches up to the names of the files
#should be in the same order
for filename in os.listdir('positiveTextFiles'):
    positiveNameList.append(filename)


for filename in os.listdir('trainingFiles'):
    trainingNameList.append(filename)
    filePath='trainingFiles/'+filename
    with open(filePath, 'r') as file:
        x=file.read()
        data.append(x)
        trainingLabelList.append(1 if filename in positiveNameList else 0)

for filename in os.listdir('testingFiles'):
    testingNameList.append(filename)
    filePath='testingFiles/'+filename
    with open(filePath, 'r') as file:
        x=file.read()
        testingData.append(x)
        testingLabelList.append(1 if filename not in positiveNameList else 0)


###PREPROCESSING FOR TRAINING DATA

#for doc in data:
#    print(doc[0:10])

data = [re.sub('\s+', ' ', sent) for sent in data]
data = [re.sub("\'", "", sent) for sent in data]
data = [re.sub('\s+', ' ', sent) for sent in data]
data = [re.sub("\'", "", sent) for sent in data]
data_words = list(sent_to_words(data))



#not entirely sure about nthreshold
# Build the bigram and trigram models
nthreshold=100
bigram = gensim.models.Phrases(data_words, min_count=5, threshold=nthreshold) # higher threshold fewer phrases.
trigram = gensim.models.Phrases(bigram[data_words], threshold=nthreshold)  

# Faster way to get a sentence clubbed as a trigram/bigram
bigram_mod = gensim.models.phrases.Phraser(bigram)
trigram_mod = gensim.models.phrases.Phraser(trigram)


# Remove Stop Words
data_words_nostops = remove_stopwords(data_words)

# Form Bigrams
data_words_bigrams = make_bigrams(data_words_nostops)
# Initialize spacy 'en' model, keeping only tagger component (for efficiency)
######### IF YOU HAVE PROBLEMS WITH THE FOLLOWING CODE LINE #######
# python3 -m spacy download en
nlp = spacy.load('en', disable=['parser', 'ner'])
# Do lemmatization keeping only noun, adj, vb, adv
data_lemmatized = lemmatization(data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])

# Create Dictionary
id2word = corpora.Dictionary(data_lemmatized)

# Create Corpus
texts = data_lemmatized

# Term Document Frequency
corpus = [id2word.doc2bow(text) for text in texts]
#ldamodel
lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=kTopics, 
                                           random_state=100,
                                           update_every=1,
                                           chunksize=100,
                                           passes=10,
                                           alpha='auto',
                                           per_word_topics=True)

doc_lda = lda_model[corpus]



###PREPROCESSING FOR TESTING DATA

testingData = [re.sub('\s+', ' ', sent) for sent in testingData]
testingData = [re.sub("\'", "", sent) for sent in testingData]
testingData = [re.sub('\s+', ' ', sent) for sent in testingData]
testingData = [re.sub("\'", "", sent) for sent in testingData]
testing_data_words = list(sent_to_words(testingData))

#not entirely sure about nthreshold
# Build the bigram and trigram models
testing_bigram = gensim.models.Phrases(testing_data_words, min_count=5, threshold=nthreshold) # higher threshold fewer phrases.
testing_trigram = gensim.models.Phrases(testing_bigram[testing_data_words], threshold=nthreshold)  

# Faster way to get a sentence clubbed as a trigram/bigram
testing_bigram_mod = gensim.models.phrases.Phraser(testing_bigram)
testing_trigram_mod = gensim.models.phrases.Phraser(testing_trigram)


# Remove Stop Words
testing_data_words_nostops = remove_stopwords(testing_data_words)

# Form Bigrams
testing_data_words_bigrams = make_bigrams(testing_data_words_nostops)
# Initialize spacy 'en' model, keeping only tagger component (for efficiency)
# python3 -m spacy download en
testing_nlp = spacy.load('en', disable=['parser', 'ner'])
# Do lemmatization keeping only noun, adj, vb, adv
testing_data_lemmatized = lemmatization(testing_data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])

# Create Dictionary
testing_id2word = corpora.Dictionary(testing_data_lemmatized)

# Create Corpus
testing_texts = testing_data_lemmatized

# Term Document Frequency
testing_corpus = [id2word.doc2bow(text) for text in testing_texts]


def format_topics_sentences_v2(ldamodel=lda_model, corpus=corpus, texts=data):
    # Init output
    sent_topics_df = pd.DataFrame()

    series_list=[]
    for i in range(kTopics):
        series_list.append(0)
    # Get main topic in each document
    for i, row in enumerate(ldamodel[corpus]):
        row = sorted(row[0], key=lambda x: (x[1]), reverse=True)
        # Get the Dominant topic, Perc Contribution and Keywords for each document
        for j, (topic_num, prop_topic) in enumerate(row):
            wp = ldamodel.show_topic(topic_num)
            series_list[topic_num]=prop_topic
            #sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic,4), topic_keywords]), ignore_index=True)
        sent_topics_df=sent_topics_df.append(pd.Series([series_list[i] for i in range(len(series_list))]), ignore_index=True)

    
    #sent_topics_df.columns = ['Topic_0', 'Topic_1', 'Topic_2']
    # Add original text to the end of the output
    #contents = pd.Series(texts, index='contents')
    sent_topics_df['contents'] = texts
    #sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
    return(sent_topics_df)

#looks like I made a second fucntion to format topic percentages;
#frankly I really hope its mapping correctly
#I need to figure out how to attach labels to them


#print(lda_model[id2word.doc2bow(data_lemmatized[5])])
df_topic_sents_keywords = format_topics_sentences_v2(lda_model, corpus, data)
df_topic_sents_keywords['label'] = trainingLabelList
df_topic_sents_keywords['name'] = trainingNameList


df_testing_topics = format_topics_sentences_v2(lda_model, testing_corpus, testingData)
df_testing_topics['label'] = testingLabelList
df_testing_topics['name'] = testingNameList
# Format
#df_dominant_topic = df_topic_sents_keywords.reset_index()
#df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']

# Show
names = ['topic_{}'.format(i) for i in range(len(df_topic_sents_keywords.columns.values.tolist()) - 3)]+['contents', 'label', 'name'] 
df_topic_sents_keywords.columns = names
df_testing_topics.columns = names

# Below we pull into the csv only the columns associated with the topics
# and the labels. (this is the first kTopics columns and the second column
# from the end
csv = df_topic_sents_keywords[names[:kTopics]+[names[-2]]].copy().to_csv()
testing_csv = df_testing_topics[names[:kTopics]+[names[-2]]].copy().to_csv()

# For printing the dataframes
#print(df_topic_sents_keywords) #these are the topics mapped to the documents #it is a pandas dataframe
#print(df_testing_topics) #these are the topics mapped to the documents #it is a pandas dataframe

print('id', end="")
print(csv)
print()
print()
print('id', end="")
print(testing_csv)

# To print the topics
pprint(lda_model.print_topics()) #these are the acutal topics

# Data preprocessing step for the unseen document


#for score in lda_model[bow_vector]:
#    print("Score: {}\t Topic: {}".format(score, lda_model.print_topic(index, 5)))
#vector=lda_model(data_lemmatized[0])
#doc_lda = lda_model[corpus]

#print scores for how good the models are:
# Compute Perplexity
#print('\nPerplexity: ', lda_model.log_perplexity(corpus))  # a measure of how good the model is. lower the better.

# Compute Coherence Score
#coherence_model_lda = CoherenceModel(model=lda_model, texts=data_lemmatized, dictionary=id2word, coherence='c_v')
#coherence_lda = coherence_model_lda.get_coherence()
#print('\nCoherence Score: ', coherence_lda)

# Visualize the topics
#pyLDAvis.enable_notebook()
#vis = pyLDAvis.gensim.prepare(lda_model, corpus, id2word)
#vis
