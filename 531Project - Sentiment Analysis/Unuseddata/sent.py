#apply sentiment value to each tweet that is encoded.
# -*- coding:utf-8 -*-
import os
import re 
import glob
import sys
import unicodedata
import pandas as pd
from textblob import TextBlob 

reload(sys)
sys.setdefaultencoding('utf8')

#unicodedata.normalize('NFKD', tweet).encode('ascii','ignore').lower()
def clean_tweet(tweet): 
        ''' 
        Utility function to clean tweet text by removing links, special characters 
        using simple regex statements. 
        '''
	#print(tweet)
        return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t]) |(\w+:\/\/\S+)", " ", tweet).split()) 

def get_tweet_sentiment(tweet): 
        ''' 
        Utility function to classify sentiment of passed tweet 
        using textblob's sentiment method 
        '''
	#unicodedata.normalize('NFKD', tweet).encode('ascii','ignore').lower()
        # create TextBlob object of passed tweet text 
	#print(tweet)
        analysis = TextBlob(clean_tweet(tweet)) 
        # set sentiment 
	#print(analysis.sentiment.polarity)
        if analysis.sentiment.polarity > 0: 
            return 'positive'
        elif analysis.sentiment.polarity == 0: 
            return 'neutral'
        else: 
            return 'negative'


decf = pd.read_csv("Marchgun.csv", encoding= 'utf-8')
decf['month'] = 'March'
decf['Sentiment'] = 'neutral'
#print(decf.head(5))
for index,row in decf.iterrows():
	#row.at['Sentiment']=get_tweet_sentiment(row['content'])
	decf.at[index, 'Sentiment'] = get_tweet_sentiment(row['content'])

decf = decf[:200]
decf.to_csv("sMar_gun.csv", index=False, encoding='utf-8-sig')
#print(decf.head(50))


