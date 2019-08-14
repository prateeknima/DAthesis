import re
import os
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split as ttsplit
from sklearn.metrics import accuracy_score
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import precision_recall_fscore_support as sc
from sklearn.svm import SVC
from sklearn.metrics import (roc_auc_score, roc_curve, auc, accuracy_score)
from imblearn.over_sampling import RandomOverSampler
from sklearn.naive_bayes import MultinomialNB

my_lemmatizer = WordNetLemmatizer()
vector = CountVectorizer()
stemming = PorterStemmer()
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')


#Getting the stopwords corpora
stopwords_list = stopwords.words('english')
print(stopwords_list[:5])
#Loading the final merged dataset
youtube_train = pd.read_csv("C:/Users/prate/Desktop/ICT_solution/Data/final_data/final_data.csv",delimiter=',')
youtube_train_sen = youtube_train['video_title']
print(youtube_train_sen[1])
#Creating an empty array
final_sent = [];
#Initializing variables
count = 0
a = 0
youtube_train['video_title'][1]
#creating an empty array
arr = []
for sentence in youtube_train['video_title']:
    #if loop to check if the data is not a string i.e. a null value
    if type(sentence) != str:
        #Drop the row if it is null
        youtube_train = youtube_train.drop([a],axis=0)
        print(a)
        #appending the index which has been dropped to the arr variable
        arr.append(a)
        #after dropping the row skip the whole loop
        continue
    #creating an empty string variable
    temp_sent = ""
    print(sentence)
    #tokenizing the sentence
    youtube_words = word_tokenize(sentence)
    filtered_youtube_sentence = ""
    #for loop to go through each tokenized word in the youtube_words variable
    for word in youtube_words:
        #if condition to check the word is not in stopword list and is an alphabet
        if word not in stopwords_list and word.isalpha():
            #stemming the word i.e. transforming it to the root word and adding it to the temp_sent variable
            temp_sent = temp_sent + " " + (stemming.stem(word))
    #appending the sentence to final_sent to get the stemmed data
    final_sent.append(temp_sent);
    #incrementing the variable
    a = a + 1

#converting the data into vectorized format
vect = CountVectorizer(stop_words="english", max_features=10000).fit(final_sent)
print("exit")
len(vect.get_feature_names())
train_vectorized = vect.transform(final_sent)
print("Stage 2 complete")

#splitting the data for training and testing
x_train,x_test,y_train,y_test = ttsplit(train_vectorized,youtube_train['tag'],test_size=0.25)
print("Stage 3 complete")

#Initialising Multinomial Naive Bayes
naive_bayes = MultinomialNB()
#training the data
naive_bayes.fit(x_train,y_train)

#using the test data for predicting the results
prediction = naive_bayes.predict(x_test)
print("Stage 4 complete")
print("Stage 4 complete")
#printing the accuracy score
print(accuracy_score(y_test,prediction))
print("Stage 5 complete")

#printing the overall metrics
from sklearn import metrics
print(metrics.classification_report(y_test, prediction))

#Creating confusion metrics
cm = metrics.confusion_matrix(y_test, prediction)
# Assigning columns names
cm_df = pd.DataFrame(cm,
                     columns=['Predicted Negative', 'Predicted Positive'],
                     index=['Actual Negative', 'Actual Positive'])
# Printing the confusion matrix
cm_df

# Assigning True Positive, True Negative, False Positive, False Negative
TN = cm[1][1]
TP= cm[0][0]
FP = cm[0][1]
FN = cm[1][0]

# calculate the specificity
conf_specificity = (TN / float(TN + FP))
#Print Specificity
conf_specificity