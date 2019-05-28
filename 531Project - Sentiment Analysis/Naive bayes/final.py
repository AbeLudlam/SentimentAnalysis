#The final model made for the Naive Bayes classifier. Multifunctional. Here are what the buttons do. 
#1. Use this to run a full 4 cross fold validation of 4 datasets. YOU MUST HAVE 4 DATASETS TO USE THIS FEATURE. All will be used as testing and training data. Results are saved to the files with timestamps and their respective function (Results is the csv file of results, monthly bar is the classifiers predicted sentiment polarity for each month)
#2. Access a "results" csv file for either its monthly sentiment bar graph, or for its accuracy over the months and average accuracy. Accessing either of these functions saves the corresponding image with a timestamped file.
#3. Access a Sentiment csv file (like sSept_immi.csv or mini_immi_training1.csv) to create the word cloud for its respective tweets and save the file with a timestamped mark with clouds.png
from tkinter import *
import pandas as pd
import re
import nltk
import importlib
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from nltk.tokenize import word_tokenize
from tkinter   import filedialog   
from tkinter.filedialog import askopenfilename
import datetime
from string import punctuation 
from nltk.corpus import stopwords 
import glob
import sys
import unicodedata
import numpy as np
import pickle
import time

matplotlib.use('TkAgg')

def bagofwords(inp):
    oecf = pd.read_csv(inp, encoding= 'utf-8')
    lik = []
    for index,row in oecf.iterrows():
        lik.append(row["content"])
    fin = " ".join(content for content in lik)
    stopwor = set(STOPWORDS)
    stopwor.update(["immigration", "https", "twitter", "pic", "video", "people", "status", "now", "need", "will", "say"])
    wordcloud = WordCloud(stopwords = stopwor, background_color="white")
    import datetime
    #print("now")
    fhg = '{0:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now())
    fhg = fhg + 'WCloud.png'
    #print(fhg)
    #lop.to_csv(fhg, index=False, encoding='utf-8-sig')
    lk = wordcloud.generate(fin)
    lk.to_file(fhg)
    #lk = wordcloud.generate(fin)


    plt.imshow(lk , interpolation= 'bilinear')
    plt.axis("off")
    plt.show()
    
    #f #= filedialog.asksaveasfile(mode='w', defaultextension=".png")
    #if f is None:
    #    return
    import datetime
    print("now")
    fhg = '{0:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now())
    fhg = fhg + 'WCloud.png'
    print(fhg)
    #lop.to_csv(fhg, index=False, encoding='utf-8-sig')
    lk = wordcloud.generate(fin)
    lk.to_file(fhg)
    
    return



LARGE_FONT= ("Verdana", 12)
class SeaofBTCapp(Tk):

    def __init__(self, *args, **kwargs):
        
        Tk.__init__(self, *args, **kwargs)
        container = Frame(self)

        container.pack(side="top", fill="both", expand = True)

        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        self.frames = {}

        for F in (StartPage, PageOne, PageTwo, Page3):

            frame = F(container, self)

            self.frames[F] = frame

            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame(StartPage)

    def show_frame(self, cont):

        frame = self.frames[cont]
        frame.tkraise()

        
class StartPage(Frame):

    def __init__(self, parent, controller):
        Frame.__init__(self,parent)
        label = Label(self, text="Start Page", font=LARGE_FONT)
        label.pack(pady=10,padx=10)

        button = Button(self, text="Use Naive Bayes",
                            command=lambda: controller.show_frame(PageOne))
        button.pack()

        button2 = Button(self, text="Display results of datasheet",
                            command=lambda: controller.show_frame(PageTwo))
        button2.pack()
        button3 = Button(self, text="Generate wordcloud of dataset",
                            command=lambda: controller.show_frame(Page3))
        button3.pack()
#def startNB():
#   print(name.get())
#   print(name1.get())
#   print(name2.get())
#   print(name3.get())

 #  return 
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


class PageOne(Frame):

    def __init__(self, parent, controller):
        Frame.__init__(self, parent)
        label = Label(self, text="Enter the four training/testing CSV files", font=LARGE_FONT)
        label.pack(pady=10,padx=10)
        def startNB():
            print(name.get())
            print(name1.get())
            print(name2.get())
            print(name3.get())
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
            decf = pd.read_csv(name3.get(), encoding= 'utf-8')
            necf = pd.read_csv(name2.get(), encoding= 'utf-8')
            oecf = pd.read_csv(name1.get(), encoding= 'utf-8')
            trainingData = pd.concat([decf, necf, oecf])
            testDataSet = pd.read_csv(name.get(), encoding= 'utf-8')
            secf = testDataSet
            tweetProcessor = PreProcessTweets()
            tweetProcessor2 = PreProcessTweets2()
            preprocessedTrainingSet = tweetProcessor.processTweets(trainingData)
            preprocessedTestSet = tweetProcessor2.processTweets(testDataSet)
            word_features = buildVocabulary(preprocessedTrainingSet)
            trainingFeatures=nltk.classify.apply_features(extract_features,preprocessedTrainingSet)
            testFeatures = nltk.classify.apply_features(extract_features,preprocessedTestSet)

# ------------------------------------------------------------------------
            print("here")
            NBayesClassifier=nltk.NaiveBayesClassifier.train(trainingFeatures)
            print("here2")
            fea = NBayesClassifier.most_informative_features(10)
            lop = pd.DataFrame(columns = ["positive", "negative", "neutral","accuracy", "total", "month", "feature1", "feature2", "feature3", "feature4", "feature5", "feature6", "feature7", "feature8", "feature9", "feature10"])
            fourth = 0

            four = 0 + len(testDataSet.index)
            cnt = 0
            count = 0
            if count < (four - 500):
                cnt = 1
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
            NBResultLabels = [NBayesClassifier.classify(extract_features(tweet[0])) for tweet in preprocessedTestSet]

            print("Positive Sentiment Percentage = " + str(100*NBResultLabels.count('positive')/len(NBResultLabels)) + "%")
            print("Negative Sentiment Percentage = " + str(100*NBResultLabels.count('negative')/len(NBResultLabels)) + "%")
            print("Neutral Sentiment Percentage = " + str(100*NBResultLabels.count('neutral')/len(NBResultLabels)) + "%")
            print("Accuracy: ", fourth)
            NBayesClassifier.show_most_informative_features()
            pos = 100*NBResultLabels.count('positive')/len(NBResultLabels)
            neg = 100*NBResultLabels.count('negative')/len(NBResultLabels)
            neu = 100*NBResultLabels.count('neutral')/len(NBResultLabels)
            lop.loc[0] = [pos] + [neg] + [neu] + [fourth] + [len(testDataSet.index)] + [testDataSet.loc[0, "month"]] + [fea[0][0]] + [fea[1][0]] + [fea[2][0]]+ [fea[3][0]] + [fea[4][0]] + [fea[5][0]] + [fea[6][0]] + [fea[7][0]] + [fea[8][0]] + [fea[9][0]]
            
            trainingData = None
            trainingData = pd.concat([decf, necf, secf])
            testDataSet = None
            testDataSet = oecf

            tweetProcessor = None
            tweetProcessor = PreProcessTweets()
            tweetProcessor2 = None
            tweetProcessor2 = PreProcessTweets2()
            preprocessedTrainingSet = None
            preprocessedTrainingSet = tweetProcessor.processTweets(trainingData)
            preprocessedTestSet = None
            preprocessedTestSet = tweetProcessor2.processTweets(testDataSet)
            word_features = None
            word_features = buildVocabulary(preprocessedTrainingSet)
            trainingFeatures = None
            trainingFeatures=nltk.classify.apply_features(extract_features,preprocessedTrainingSet)
            testFeatures = None
            testFeatures = nltk.classify.apply_features(extract_features,preprocessedTestSet)

# ------------------------------------------------------------------------
            NBayesClassifier = None
            NBayesClassifier=nltk.NaiveBayesClassifier.train(trainingFeatures)

            fea = None
            fea = NBayesClassifier.most_informative_features(10)

            NBResultLabels = None
            NBResultLabels = [NBayesClassifier.classify(extract_features(tweet[0])) for tweet in preprocessedTestSet]
            fourth = 0

            four = 0 + len(testDataSet.index)
            if count < (four - 500):
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


            print("Positive Sentiment Percentage = " + str(100*NBResultLabels.count('positive')/len(NBResultLabels)) + "%")
            print("Negative Sentiment Percentage = " + str(100*NBResultLabels.count('negative')/len(NBResultLabels)) + "%")
            print("Neutral Sentiment Percentage = " + str(100*NBResultLabels.count('neutral')/len(NBResultLabels)) + "%")
            print("Accuracy: ", fourth)

            pos = None
            pos = 100*NBResultLabels.count('positive')/len(NBResultLabels)
            neg = None
            neg = 100*NBResultLabels.count('negative')/len(NBResultLabels)
            neu = None
            neu = 100*NBResultLabels.count('neutral')/len(NBResultLabels)
            lop.loc[1] = [pos] + [neg] + [neu] + [fourth] + [len(testDataSet.index)] + [testDataSet.loc[0, "month"]] + [fea[0][0]] + [fea[1][0]] + [fea[2][0]]+ [fea[3][0]] + [fea[4][0]] + [fea[5][0]] + [fea[6][0]] + [fea[7][0]] + [fea[8][0]] + [fea[9][0]]
            print("2 of 4 datasets tested")

            trainingData = None
            trainingData = pd.concat([decf, oecf, secf])
            testDataSet = None
            testDataSet = necf
           
            tweetProcessor = None
            tweetProcessor = PreProcessTweets()
            tweetProcessor2 = None
            tweetProcessor2 = PreProcessTweets2()
            preprocessedTrainingSet = None
            preprocessedTrainingSet = tweetProcessor.processTweets(trainingData)
            preprocessedTestSet = None
            preprocessedTestSet = tweetProcessor2.processTweets(testDataSet)
            word_features = None
            word_features = buildVocabulary(preprocessedTrainingSet)
            trainingFeatures = None
            trainingFeatures=nltk.classify.apply_features(extract_features,preprocessedTrainingSet)
            testFeatures = None
            testFeatures = nltk.classify.apply_features(extract_features,preprocessedTestSet)

# ------------------------------------------------------------------------
            NBayesClassifier = None
            NBayesClassifier=nltk.NaiveBayesClassifier.train(trainingFeatures)


            fea = None
            fea = NBayesClassifier.most_informative_features(10)

            NBResultLabels = None
            NBResultLabels = [NBayesClassifier.classify(extract_features(tweet[0])) for tweet in preprocessedTestSet]
            fourth = 0

            four = 0 + len(testDataSet.index)
            if count < (four - 500):
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


          

            print("Positive Sentiment Percentage = " + str(100*NBResultLabels.count('positive')/len(NBResultLabels)) + "%")
            print("Negative Sentiment Percentage = " + str(100*NBResultLabels.count('negative')/len(NBResultLabels)) + "%")
            print("Neutral Sentiment Percentage = " + str(100*NBResultLabels.count('neutral')/len(NBResultLabels)) + "%")
            print("Accuracy: ", fourth)

            pos = None
            pos = 100*NBResultLabels.count('positive')/len(NBResultLabels)
            neg = None
            neg = 100*NBResultLabels.count('negative')/len(NBResultLabels)
            neu = None
            neu = 100*NBResultLabels.count('neutral')/len(NBResultLabels)
            lop.loc[2] = [pos] + [neg] + [neu] + [fourth] + [len(testDataSet.index)] + [testDataSet.loc[0, "month"]] + [fea[0][0]] + [fea[1][0]] + [fea[2][0]]+ [fea[3][0]] + [fea[4][0]] + [fea[5][0]] + [fea[6][0]] + [fea[7][0]] + [fea[8][0]] + [fea[9][0]]
            print("3 of 4 datasets tested")
 
            trainingData = None
            trainingData = pd.concat([oecf, necf, secf])
            testDataSet = None
            testDataSet = decf

            tweetProcessor = None
            tweetProcessor = PreProcessTweets()
            tweetProcessor2 = None
            tweetProcessor2 = PreProcessTweets2()
            preprocessedTrainingSet = None
            preprocessedTrainingSet = tweetProcessor.processTweets(trainingData)
            preprocessedTestSet = None
            preprocessedTestSet = tweetProcessor2.processTweets(testDataSet)
            word_features = None
            word_features = buildVocabulary(preprocessedTrainingSet)
            trainingFeatures = None
            trainingFeatures=nltk.classify.apply_features(extract_features,preprocessedTrainingSet)
            testFeatures = None
            testFeatures = nltk.classify.apply_features(extract_features,preprocessedTestSet)

# ------------------------------------------------------------------------
            NBayesClassifier = None
            NBayesClassifier=nltk.NaiveBayesClassifier.train(trainingFeatures)

            fea = None
            fea = NBayesClassifier.most_informative_features(10)

            NBResultLabels = None
            NBResultLabels = [NBayesClassifier.classify(extract_features(tweet[0])) for tweet in preprocessedTestSet]
            fourth = 0

            four = 0 + len(testDataSet.index)
            if count < (four - 500):
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



            print("Positive Sentiment Percentage = " + str(100*NBResultLabels.count('positive')/len(NBResultLabels)) + "%")
            print("Negative Sentiment Percentage = " + str(100*NBResultLabels.count('negative')/len(NBResultLabels)) + "%")
            print("Neutral Sentiment Percentage = " + str(100*NBResultLabels.count('neutral')/len(NBResultLabels)) + "%")
            print("Accuracy: ", fourth)

            pos = None
            pos = 100*NBResultLabels.count('positive')/len(NBResultLabels)
            neg = None
            neg = 100*NBResultLabels.count('negative')/len(NBResultLabels)
            neu = None
            neu = 100*NBResultLabels.count('neutral')/len(NBResultLabels)
            lop.loc[3] = [pos] + [neg] + [neu] + [fourth] + [len(testDataSet.index)] + [testDataSet.loc[0, "month"]] + [fea[0][0]] + [fea[1][0]] + [fea[2][0]]+ [fea[3][0]] + [fea[4][0]] + [fea[5][0]] + [fea[6][0]] + [fea[7][0]] + [fea[8][0]] + [fea[9][0]]
            import datetime
            fhg = '{0:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now())
            fhg = fhg + 'NaiveResults.csv'
            print(fhg)
            lop.to_csv(fhg, index=False, encoding='utf-8-sig')
            month = []
            accuracy = []
            pos = []
            neg = []
            neu = []
            for index,row in lop.iterrows():
                month.append(row["month"])
                accuracy.append(row["accuracy"])
#               accuracy.append(row["negative"])
#               accuracy.append(row["neutral"])
                pos.append(row["positive"])
                neg.append(row["negative"])
                neu.append(row["neutral"])
            from tkinter import messagebox
            po, ne = plt.subplots()
            messagebox.showinfo("Accuracy", (accuracy[0]+accuracy[1]+accuracy[2]+accuracy[3])/4)
            
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
            fh = '{0:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now())
            fh = fh + 'NaiveBar.png'
            #print(fh)
            plt.savefig(fname = fh)
           
            plt.show()
            
            
            #pop, nep = plt.subplots()
            

            
            return
 
        button1 = Button(self, text="Back to Home",
                            command=lambda: controller.show_frame(StartPage))
        button1.pack()
        name = StringVar()
        e = Entry(self, width=25, textvariable=name).pack()
        name1 = StringVar()
        e1 = Entry(self, width=25, textvariable=name1).pack()
        name2 = StringVar()
        e2 = Entry(self, width=25, textvariable=name2).pack()
        name3 = StringVar()
        e3 = Entry(self, width=25, textvariable=name3).pack()
        button2 = Button(self, text="submit", command= startNB).pack()
    
#def NaiveBayes(inp1,inp2,inp3,inp4):
#    return
    
#def startNB():
#  NaiveBayes(e.get(),e1.get(),e2.get(),e3.get())
#   return 


class PageTwo(Frame):

    def __init__(self, parent, controller):
        Frame.__init__(self, parent)
        label = Label(self, text="Get monthly data or accuracy from results", font=LARGE_FONT)
        label.pack(pady=10,padx=10)

        button1 = Button(self, text="Back to Home",
                            command=lambda: controller.show_frame(StartPage))
        button1.pack()
        button2= Button(self, text='Get Monthly sentiment of result CSV', command=callbackMonth).pack()
        button3= Button(self, text='Get accuracy of result CSV', command=callbackAcc).pack()
       
def callbackMonth():
    name2= askopenfilename()
    print(name2)
    
    monthGeter(name2)
    
    return name2
def callbackAcc():
    name1= askopenfilename()
    print(name1)
    accGeter(name1)
    monthGeter(name1)
    
    return name1
def monthGeter(nm):
            lof = pd.read_csv(nm, encoding= 'utf-8')
            month = []
            accuracy = []
            pos = []
            neg = []
            neu = []
            for index,row in lof.iterrows():
                month.append(row["month"])
                accuracy.append(row["accuracy"])
#               accuracy.append(row["negative"])
#               accuracy.append(row["neutral"])
                pos.append(row["positive"])
                neg.append(row["negative"])
                neu.append(row["neutral"])

            po, ne = plt.subplots()

            
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
            fh = '{0:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now())
            fh = fh + 'NaiveMonth.png'
            #print(fh)
            plt.savefig(fname = fh)
            
            plt.show()
        

def accGeter(nm):
            lof = pd.read_csv(nm, encoding= 'utf-8')
            month = []
            accuracy = []
            pos = []
            neg = []
            neu = []
            for index,row in lof.iterrows():
                month.append(row["month"])
                accuracy.append(row["accuracy"])
                print(month[index] + " accuracy is ")
                print(accuracy[index])
#               accuracy.append(row["negative"])
#               accuracy.append(row["neutral"])
                
            accuracy.append((accuracy[0]+accuracy[1]+accuracy[2]+accuracy[3])/4)
            print("Average accuracy is: ")
            print(accuracy[4])
            month.append("Average Accuracy")
            o, ne = plt.subplots()

            
            bar_width= .2
            opacity = .8
            index = np.arange(5)
            
            axes = plt.gca()
            axes.set_ylim([0,100])
            
            nlf = plt.bar(index,accuracy,bar_width, alpha=.8, color="b")
            axes2 = plt.gca()
            axes2.set_ylim([0,1])
            plt.xlabel('Month')
            plt.ylabel('Percentage')
            plt.title('Naive Bayes Accuracy')
            plt.xticks( index + bar_width, (month))
            plt.grid()
            ne.set_axisbelow(True)
            fh3 = '{0:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now())
            fh3 = fh3 + 'NaiveAcc.png'
            #print(fh)
            plt.savefig(fname = fh3)
            
            plt.show()
        


class Page3(Frame):

    def __init__(self, parent, controller):
        Frame.__init__(self, parent)
        label = Label(self, text="Generate word cloud file from a tweet dataset", font=LARGE_FONT)
        label.pack(pady=10,padx=10)

        button1 = Button(self, text="Back to Home",
                            command=lambda: controller.show_frame(StartPage))
        button1.pack()

       
        #errmsg = 'Error!'
        button3= Button(self, text='Open Sentiment CSV file', command=callbackBOW).pack(fill=X)

def callbackBOW():
    name= askopenfilename()
    print(name)
    bagofwords(name)
    
    return name

def bowYes():
    return


def answe():
    return
 


app = SeaofBTCapp()
app.mainloop()
