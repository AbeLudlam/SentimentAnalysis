#Model used for svm model classifier. "python3 sv.py" to run. Currently only uses "sSept_immi.csv" as the testing set and the others as the training set.
import pandas as pd
import re
import nltk
import importlib
#nltk.download("stopwords")
#nltk.download("punkt")
from nltk.tokenize import word_tokenize
from string import punctuation 
from nltk.corpus import stopwords 
import glob
import sys
import unicodedata
import numpy as np
import pickle
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
wordnet_lemmatizer = WordNetLemmatizer()

def normalizer(tweet):
    only_letters = re.sub("[^a-zA-Z]", " ",tweet) 
    tokens = nltk.word_tokenize(only_letters)[2:]
    lower_case = [l.lower() for l in tokens]
    filtered_result = list(filter(lambda l: l not in stop_words, lower_case))
    lemmas = [wordnet_lemmatizer.lemmatize(t) for t in filtered_result]
    return lemmas
decf = pd.read_csv("sDec_immi.csv", encoding= 'utf-8')
necf = pd.read_csv("sNov_immi.csv", encoding= 'utf-8')
oecf = pd.read_csv("sOct_immi.csv", encoding= 'utf-8')
trainingData = pd.concat([decf, necf, oecf])
testDataSet = pd.read_csv("sSept_immi.csv", encoding= 'utf-8')
print("Length: ", len(testDataSet.index))
print(testDataSet.loc[0, "month"])
trainingData['normalized_tweet'] = trainingData.content.apply(normalizer)
testDataSet['normalized_tweet'] = testDataSet.content.apply(normalizer)
def ngrams(input_list):
    #onegrams = input_list
    bigrams = [' '.join(t) for t in list(zip(input_list, input_list[1:]))]
    trigrams = [' '.join(t) for t in list(zip(input_list, input_list[1:], input_list[2:]))]
    return bigrams+trigrams
#tweets['grams'] = tweets.normalized_tweet.apply(ngrams)
#tweets[['grams']].head()
trainingData['grams'] = trainingData.normalized_tweet.apply(ngrams)
testDataSet['grams'] = testDataSet.normalized_tweet.apply(ngrams)
#print(trainingData[['grams']].head(5))
from scipy.sparse import hstack
from sklearn.feature_extraction.text import CountVectorizer
count_vectorizer = CountVectorizer(ngram_range=(1,2))
vectorized_data = count_vectorizer.fit_transform(trainingData.content)
#vectorized_data2 = count_vectorizer.transform(testDataSet.content)
indexed_data = hstack((np.array(range(0,vectorized_data.shape[0]))[:,None], vectorized_data))
vectorized_data2 = count_vectorizer.transform(testDataSet.content)
indexed_data2 = hstack((np.array(range(0,vectorized_data2.shape[0]))[:,None], vectorized_data2))
def sentiment2target(sentiment):
    return {
        'negative': 0,
        'neutral': 1,
        'positive' : 2
    }[sentiment]
from sklearn import metrics
targets = trainingData.Sentiment.apply(sentiment2target)
targets2 = testDataSet.Sentiment.apply(sentiment2target)
#print(metrics.confusion_matrix(targets2, targets2))
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.multiclass import OneVsRestClassifier
clf = OneVsRestClassifier(svm.SVC(gamma=0.01, C=100.0, probability=True, class_weight= 'balanced', kernel='rbf', max_iter = 100000))
print("start to fit")
#scaler = StandardScaler(with_mean=False)
#scaler.fit(indexed_data)

#clf_output = clf.fit(scaler.transform(indexed_data), targets)
clf_output = clf.fit(indexed_data, targets)
print("done fitting")
pred_lin = clf_output.predict(indexed_data)
#print(metrics.confusion_matrix(targets, pred_lin))
print("done predicting")
print(clf_output.score(indexed_data2, targets2))
#clf_output = clf.fit(indexed_data2, targets2)
pred_lin = clf_output.predict(indexed_data2)
print(metrics.confusion_matrix(targets2, pred_lin))
