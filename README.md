ll files can be found in this directory, the folders are there to help guide for specific things. 

All python programs are meant to be run in Python3 through the command line (Wanted to turn the final model into an executable, but the nltk library dependency causes problems with that)

example execution, say for the final model, is "python3 final.py"

Each of the core models can be found in "naive.py", "Maxent.py", and "sv.py", for Naive bayes, Maximum entropy, and svm respectively. Naive.py saves its result to "SoloNaiveResults.csv", Maxent.py saves to "maxent_immi_results.csv", and svm saves to 'SVM_results.csv' but only its accuracy. Total percentage of each chosen for SVM is by manually going through the printed confusion matrix 
(view SVM_results.png, the confusion matrix is also in a weird order, with the columns being the predicted results and the rows being the actual, with the first column being "negative", second being "neutral", and third being "positive")

Model_accuracy_results.py looks specifically at "Naive_immi_results.csv","maxent_immi_results.csv", and"svm_immi_results.csv", and plots their accuracy on a bar graph. 

Final.py is the final implementaion of the Naive bayes model and other features. Here are what the options do.

1. The final Naive Bayes model. It must be passed four datasets, where it will perform 4-cross fold validation on the received datasets. Accuracy is shown in a pop a window and a bar graph of the predicted percentages of sentiment for each month is shown. Results are saved to the files with timestamps and their respective function (Results is the csv file of results, Naive bar is the classifiers predicted sentiment polarity for each month). remember to input correct names of 4 different sentiment datasets (sSept_immi.csv and mini_immi_training) are the examples for this. Can also look through Sentiment Datasets for what can be used for testing and training datasets.

2. Get a bar graph displaying the predicted sentiment percentages, or get a bar graph showing the accuracy for each month and the average accuracy overall. The input for these files are the "results" csv files, either from other models are from this model specifically. Creation of these graphs also lead to saving them with a timestamp and a respective suffix to be able to access them again.

3. Create a word cloud based of one of the Sentiment Datasets used for testing and training (such as sDec_immi.csv) As like all other functions of this program, this graph is saved with a timestamp and a respective suffix (cloud).

Be careful canceling and inputting values, the UI is clunky and easily crashes and gets stuck. 

If the main purpose is to test the model, simply use option 1 and make sure type in the correct file names for all four datasets.
