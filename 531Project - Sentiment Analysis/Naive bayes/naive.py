#The standalone naive bayes classifier that was used in the comparison. Uses "sSept_immi.csv" as the testing set and the others as training datasets. "python3 naive.py" to run. Saves results to SoloNaiveResults.csv
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
import time

#importlib.reload(sys)
#sys.setdefaultencoding('utf8')
class PreProcessTweets2:
        def __init__(self):
            self._stopwords = set(stopwords.words('english') + list(punctuation) + ['AT_USER','URL'])
        
        def processTweets(self, list_of_tweets):
            processedTweets=[]
       # for tweet in list_of_tweets:
        #    processedTweets.append((self._processTweet(tweet["content"]),tweet["Sentiment"]))
            for index,row in testDataSet.iterrows():
        #row.at['Sentiment']=get_tweet_sentiment(row['content'])
                processedTweets.append((self._processTweet(row["content"]),row["Sentiment"]))
        #decf.at[index, 'Sentiment'] = get_tweet_sentiment(row['content'])
            return processedTweets
    
        def _processTweet(self, tweet):
            tweet = tweet.lower() # convert text to lower-case
            tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))', 'URL', tweet) # remove URLs
            tweet = re.sub('@[^\s]+', 'AT_USER', tweet) # remove usernames
            tweet = re.sub(r'#([^\s]+)', r'\1', tweet) # remove the # in #hashtag
            tweet = word_tokenize(tweet) # remove repeated characters (helloooooooo into hello)
            return np.asarray([word for word in tweet if word not in self._stopwords])
class PreProcessTweets:
        def __init__(self):
            self._stopwords = set(stopwords.words('english') + list(punctuation) + ['AT_USER','URL'])
        
        def processTweets(self, list_of_tweets):
            processedTweets=[]
       # for tweet in list_of_tweets:
        #    processedTweets.append((self._processTweet(tweet["content"]),tweet["Sentiment"]))
            for index,row in trainingData.iterrows():
        #row.at['Sentiment']=get_tweet_sentiment(row['content'])
                processedTweets.append((self._processTweet(row["content"]),row["Sentiment"]))
        #decf.at[index, 'Sentiment'] = get_tweet_sentiment(row['content'])
            return processedTweets
    
        def _processTweet(self, tweet):
            tweet = tweet.lower() # convert text to lower-case
            tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))', 'URL', tweet) # remove URLs
            tweet = re.sub('@[^\s]+', 'AT_USER', tweet) # remove usernames
            tweet = re.sub(r'#([^\s]+)', r'\1', tweet) # remove the # in #hashtag
            tweet = word_tokenize(tweet) # remove repeated characters (helloooooooo into hello)
            return np.asarray([word for word in tweet if word not in self._stopwords])
decf = pd.read_csv("sDec_immi.csv", encoding= 'utf-8')
necf = pd.read_csv("sNov_immi.csv", encoding= 'utf-8')
oecf = pd.read_csv("sOct_immi.csv", encoding= 'utf-8')
trainingData = pd.concat([decf, necf, oecf])
testDataSet = pd.read_csv("sSept_immi.csv", encoding= 'utf-8')
print("Length: ", len(testDataSet.index))
print(testDataSet.loc[0, "month"])
trainingData = trainingData[:300]
testDataSet = testDataSet[:100]
tweetProcessor = PreProcessTweets()
tweetProcessor2 = PreProcessTweets2()
preprocessedTrainingSet = tweetProcessor.processTweets(trainingData)
preprocessedTestSet = tweetProcessor2.processTweets(testDataSet)

# ------------------------------------------------------------------------

import nltk 

def buildVocabulary(preprocessedTrainingData):
    all_words = []
    
    for (words, sentiment) in preprocessedTrainingData:
        all_words.extend(words)

    wordlist = nltk.FreqDist(all_words)
    word_features = wordlist.keys()
    
    return word_features

# ------------------------------------------------------------------------

def extract_features(tweet):
    tweet_words=set(tweet)
    features={}
    for word in word_features:
        features['contains(%s)' % word]=(word in tweet_words)
    return features

# ------------------------------------------------------------------------

# Now we can extract the features and train the classifier 
word_features = buildVocabulary(preprocessedTrainingSet)
trainingFeatures=nltk.classify.apply_features(extract_features,preprocessedTrainingSet)
testFeatures = nltk.classify.apply_features(extract_features,preprocessedTestSet)

# ------------------------------------------------------------------------

NBayesClassifier=nltk.NaiveBayesClassifier.train(trainingFeatures)
#classifier_f = open("naivebayes.pickle", "rb")
#NBayesClassifier = pickle.load(classifier_f)
#classifier_f.close()

fea = NBayesClassifier.most_informative_features(10)
lop = pd.DataFrame(columns = ["positive", "negative", "neutral","accuracy", "total", "month", "feature1", "feature2", "feature3", "feature4", "feature5", "feature6", "feature7", "feature8", "feature9", "feature10"])
#pos = 45.32
#neg = 33.23
#neu = 44.213
#fourth = 92.34
#lop.loc[0] = [pos] + [neu] + [fourth] + [len(testDataSet.index)] + [testDataSet.loc[0, "month"]] + [fea[0][0]] + [fea[1][0]] + [fea[2][0]]+ [fea[3][0]] + [fea[4][0]] + [fea[5][0]] + [fea[6][0]] + [fea[7][0]] + [fea[8][0]] + [fea[9][0]]
#lop = pd.DataFrame(columns = ["total", "month", "feature1", "feature2", "feature3", "feature4", "feature5", "feature6", "feature7", "feature8", "feature9", "feature10"])
#lop.loc[0] = [len(testDataSet.index)] + [testDataSet.loc[0, "month"]] + fea[0][0] + fea[1][0] + fea[2][0]+ fea[3][0] + fea[4][0] + fea[5][0] + fea[6][0] + fea[7][0] + fea[8][0] + fea[9][0]
#lop = pd.DataFrame(columns = ["total", "month","feature1", "feature2", "feature3", "feature4", "feature5", "feature6", "feature7", "feature8", "feature9", "feature10"])
#lop.loc[0] = [len(testDataSet.index)] + [testDataSet.loc[0, "month"]]  + [fea[0][0]] + [fea[1][0]] + [fea[2][0]]+ [fea[3][0]] + [fea[4][0]] + [fea[5][0]] + [fea[6][0]] + [fea[7][0]] + [fea[8][0]] + [fea[9][0]]
#print("here")
#for f,w in fea:
#	print(f,w)
print("here")

# ------------------------------------------------------------------------

NBResultLabels = [NBayesClassifier.classify(extract_features(tweet[0])) for tweet in preprocessedTestSet]


#second =(testFeatures.length()/2)
#first = nltk.classify.accuracy(NBayesClassifier, testFeatures[:700])
#fourth = first
#first = None
#third = nltk.classify.accuracy(NBayesClassifier, testFeatures[700:1400])
#fourth = fourth + third
#third = None
#th = nltk.classify.accuracy(NBayesClassifier, testFeatures[1400:2100])
#fourth = fourth + th
#th = None
#thi = nltk.classify.accuracy(NBayesClassifier, testFeatures[2100:2799])
#fourth = fourth + thi
#thi = None
#fourth = (fourth/4)
fourth = 0
#save_classifier = open("naivebayes.pickle","wb")
#pickle.dump(NBayesClassifier, save_classifier)
#save_classifier.close()
four = 0 + len(testDataSet.index)
cnt = 1
count = 0
first = nltk.classify.accuracy(NBayesClassifier, testFeatures[:500])
fourth = first
count = 500
while count < (four - 500):
	third = nltk.classify.accuracy(NBayesClassifier, testFeatures[count:(count+500)])
	fourth = fourth + third
	count = count + 500
	cnt = cnt + 1
fourth = fourth + nltk.classify.accuracy(NBayesClassifier, testFeatures[count:four])
cnt = cnt + 1
fourth = (fourth/cnt)
# ------------------------------------------------------------------------

# get the majority vote
if NBResultLabels.count('positive') > NBResultLabels.count('negative'):
    print("Overall Positive Sentiment")
else: 
    print("Overall Negative Sentiment")
print("Positive Sentiment Percentage = " + str(100*NBResultLabels.count('positive')/len(NBResultLabels)) + "%")
print("Negative Sentiment Percentage = " + str(100*NBResultLabels.count('negative')/len(NBResultLabels)) + "%")
print("Neutral Sentiment Percentage = " + str(100*NBResultLabels.count('neutral')/len(NBResultLabels)) + "%")
print("Accuracy: ", fourth)
NBayesClassifier.show_most_informative_features()
pos = 100*NBResultLabels.count('positive')/len(NBResultLabels)
neg = 100*NBResultLabels.count('negative')/len(NBResultLabels)
neu = 100*NBResultLabels.count('neutral')/len(NBResultLabels)
lop.loc[0] = [pos] + [neg] + [neu] + [fourth] + [len(testDataSet.index)] + [testDataSet.loc[0, "month"]] + [fea[0][0]] + [fea[1][0]] + [fea[2][0]]+ [fea[3][0]] + [fea[4][0]] + [fea[5][0]] + [fea[6][0]] + [fea[7][0]] + [fea[8][0]] + [fea[9][0]]
#NBResultLabels.show_most_informative_features()

lop.to_csv("SoloNaiveResults.csv", index=False, encoding='utf-8-sig')

