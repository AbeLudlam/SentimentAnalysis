#used to create the word clouds. Replaced by the functionality in gui2.py

import pandas as pd
import re
import nltk
import importlib
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from nltk.tokenize import word_tokenize
matplotlib.use('TkAgg')
#necf = pd.read_csv("immi_results.csv", encoding= 'utf-8')
necf = pd.read_csv("sSept_immi.csv", encoding= 'utf-8')
#month = []
mnmm = pd.read_csv("sOct_immi.csv", encoding= 'utf-8')
necf= necf[:150]
mnmm = mnmm[:150]
necf.to_csv("mini_immi_training2.csv", index=False, encoding='utf-8-sig')
mnmm.to_csv("mini_immi_training3.csv", index=False, encoding='utf-8-sig')
            

lik = []
for index,row in necf.iterrows():
	lik.append(row["content"])
fin = " ".join(content for content in lik)
stopwor = set(STOPWORDS)
stopwor.update(["immigration", "https", "twitter", "pic", "video", "people", "status", "now", "need", "will", "say"])

#rev = necf.Sentiment.str.cat(sep=' ')
#tokens = word_tokenize(rev)
#freq = set(tokens)
wordcloud = WordCloud(stopwords = stopwor, background_color="white")
#wordcloud.generate_from_frequencies(freq)
lk = wordcloud.generate(fin)

lk.to_file("immigration_oct.png")
plt.imshow(lk , interpolation= 'bilinear')
plt.axis("off")
plt.show()

#accuracy = []
pos = 0 
neg = 0
neu = 0
for index,row in necf.iterrows():
	if(row['Sentiment'] == 'positive'):
		pos=pos+1
	elif(row['Sentiment'] == 'negative'):
		neg= neg+1
	else:
		neu = neu+1
pa= pos/2799
na= neg/2799
nv= neu/2799
#print(pos, pa)
#print(neg, na)
#print(neu, nv)
