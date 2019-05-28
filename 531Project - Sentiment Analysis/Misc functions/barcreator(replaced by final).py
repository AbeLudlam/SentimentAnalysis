#was used to create the monthly bar graph for the Naive bayes model. Replaced by the same functionality in the "gui2.py".

import pandas as pd
import re
import nltk
import importlib
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
secf = pd.read_csv("Veimmi_results.csv", encoding = 'utf-8')
necf = pd.read_csv("nimmi_results.csv", encoding= 'utf-8')
decf = pd.concat([ necf, secf])
decf.to_csv("final_immi_results.csv", index=False, encoding='utf-8-sig')
#necf = pd.read_csv("sSept_immi.csv", encoding= 'utf-8')
month = []
accuracy = []
pos = []
neg = []
neu = []
for index,row in decf.iterrows():
	month.append(row["month"])
#	accuracy.append(row["positive"])
#	accuracy.append(row["negative"])
#	accuracy.append(row["neutral"])
	pos.append(row["positive"])
	neg.append(row["negative"])
	neu.append(row["neutral"])
#plt.pie(accuracy, labels=["positive", "negative", "neutral"],colors=['yellowgreen', 'lightcoral', 'gold'], autopct='%1.1f%%', shadow=True,  startangle=140)
#plt.axis('equal')
#plt.title('Naive Bayes predicted values')
#axes = plt.gca()
#axes.set_ylim([0,100])
#plt.show()
#plt.savefig(fname = "FNaivaccuracy.png")
po, ne = plt.subplots()

#ne.grid('on')
#ne.set_axisbelow(True)
bar_width= .3
opacity = .8
index = np.arange(4)
sep = plt.bar(index, pos, bar_width, alpha=opacity ,color='g', label ="Positive")


oc = plt.bar(index+bar_width, neg, bar_width, alpha=opacity, color='r', label ="Negative")
dec = plt.bar(index+(bar_width*2), neu, bar_width, alpha=opacity, color='y', label ="Neutral")
axes = plt.gca()
axes.set_ylim([0,100])
plt.xlabel('Month')
plt.ylabel('Percentage')
plt.title('Naive Bayes Sentiment Trends')
plt.xticks( index + bar_width, (month))
plt.grid()
ne.set_axisbelow(True)
plt.legend()
plt.savefig(fname = "Fsent_per_n.png")

