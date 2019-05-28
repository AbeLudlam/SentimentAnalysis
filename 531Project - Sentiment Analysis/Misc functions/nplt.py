#Used to generate the pie chart for the results of the Naive bayes model. Can be modified to access the results of the other modifiers and generate the pie charts for them.
import pandas as pd
import re
import nltk
import importlib
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
necf = pd.read_csv("Naive_immi_results.csv", encoding= 'utf-8')
#necf = pd.read_csv("sSept_immi.csv", encoding= 'utf-8')
month = []
accuracy = []
pos = []
neg = []
neu = []
for index,row in necf.iterrows():
#	month.append(row["month"])
	accuracy.append(row["positive"])
	accuracy.append(row["negative"])
	accuracy.append(row["neutral"])
#	pos.append(row["positive"])
#	neg.append(row["negative"])
#	neu.append(row["neutral"])
plt.pie(accuracy, labels=["positive", "negative", "neutral"],colors=['yellowgreen', 'lightcoral', 'gold'], autopct='%1.1f%%', shadow=True,  startangle=140)
plt.axis('equal')
plt.title('Naive Bayes predicted values')
#axes = plt.gca()
#axes.set_ylim([0,100])
#plt.show()
plt.savefig(fname = "Naivaccuracy.png")
#po, ne = plt.subplots()
#bar_width= .3
#opacity = .8
#index = np.arange(4)
#sep = plt.bar(index, pos, bar_width, alpha=opacity, color='g', label ="Positive")


#oc = plt.bar(index+bar_width, neg, bar_width, alpha=opacity, color='r', label ="Negative")
#dec = plt.bar(index+(bar_width*2), neu, bar_width, alpha=opacity, color='y', label ="Neutral")

#plt.xlabel('Month')
#plt.ylabel('Percentage')
#plt.title('Naive Bayes Sentiment Trends')
#plt.xticks( index + bar_width, (month))
#plt.legend()
#plt.savefig(fname = "sent_per_n.png")

