#Used to create "acc_for_all" not necessary replaceable, but features aren't need anymore
import pandas as pd
import re
import nltk
import importlib
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
#lop = pd.DataFrame(columns = ["positive", "negative", "neutral","accuracy", "total", "month"])
#lop.loc[0] = [.566118656] + [.24803431] + [.185847034] + [.4131522516108] + [2799] + ['September'] 
#lop.to_csv("mvm_immi_results.csv", index=False, encoding='utf-8-sig')
necf = pd.read_csv("Naive_immi_results.csv", encoding= 'utf-8')

svf = pd.read_csv("maxent_immi_results.csv", encoding= 'utf-8')
decf = pd.read_csv("svm_immi_results.csv", encoding= 'utf-8')
month = []
accuracy = []
pos = []
neg = []
neu = []
for index,row in necf.iterrows():
	accuracy.append(row["accuracy"])
#	accuracy.append(row["positive"])
#	accuracy.append(row["negative"])
#	accuracy.append(row["neutral"])
#	pos.append(row["positive"])
#	neg.append(row["negative"])
#	neu.append(row["neutral"])
for index,row in svf.iterrows():
	accuracy.append(row["accuracy"])
#	accuracy.append(row["positive"])
#	accuracy.append(row["negative"])
#	accuracy.append(row["neutral"])
#	pos.append(row["positive"])
#	neg.append(row["negative"])
#	neu.append(row["neutral"])
for index,row in decf.iterrows():
	accuracy.append(row["accuracy"])
	
#plt.pie(accuracy, labels=["positive", "negative", "neutral"],colors=['yellowgreen', 'lightcoral', 'gold'], autopct='%1.1f%%', shadow=True,  startangle=140)
#plt.axis('equal')
#plt.title('Naive Bayes predicted values')
#axes = plt.gca()
#axes.set_ylim([0,1])
#plt.show()
#plt.savefig(fname = "FNaivaccuracy.png")
po, ne = plt.subplots()

#ne.grid('on')
#ne.set_axisbelow(True)
#bar_width= .3
#opacity = .8
#index = np.arange(3)
sep = plt.bar(['Naive Bayes','Maximum Entropy', 'SVMs'], accuracy, .3, alpha=.8 ,color='g', label ="Accuracy")
#plt.plot( month, accuracy, color = 'b', label = 'Naive Bayes')

#oc = plt.bar(index+bar_width, neg, bar_width, alpha=opacity, color='r', label ="Negative")
#dec = plt.bar(index+(bar_width*2), neu, bar_width, alpha=opacity, color='y', label ="Neutral")
axes = plt.gca()
axes.set_ylim([0,1])
plt.xlabel('Model')
plt.ylabel('Percentage')
plt.title('Accuracy of all models')
#plt.xticks( index + bar_width, (month))
#plt.grid()
#ne.set_axisbelow(True)
#plt.legend()
#plt.show()

plt.savefig(fname = "Acc_for_all.png")

