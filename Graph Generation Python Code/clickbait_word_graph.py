import re
import os
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter
from sklearn.model_selection import train_test_split as ttsplit
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import precision_recall_fscore_support as sc
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import random

#import pandas_profiling as pp

my_lemmatizer = WordNetLemmatizer()
vector = CountVectorizer()
stemming = PorterStemmer()
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

stopwords_list = stopwords.words('english')
print(stopwords_list[:5])

#Setting up the working directory
os.chdir('C:\\Users\\prate\\Desktop\\thesis_data_final2')
with open("foo0.csv", "w") as my_empty_csv:
    pass

youtube_train = pd.read_csv("C:/Users/prate/Desktop/thesis_data_final2/final/final1_try_channel.csv",delimiter=',')
youtube_train_sen = youtube_train['video_title']
print(youtube_train_sen[1])

final_sent = [];
count = 0

a = 0
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

#Initializing variables
youtube_targets = youtube_train['tag']
count = 0
count_cb = 0
final_words_clickbait = []


for sentence in final_sent:
    #Gathering the top 100,000 words
    if count_cb > 100000:
      break;
    #If condition for count not in array break
    if count not in arr:
        #Taking only clickbait titles into consideration
        if (youtube_targets[count] == 1):
            count_cb = count_cb + 1
            final_words_clickbait = final_words_clickbait + (word_tokenize(sentence))
    count = count + 1

print("Done")
#Plotting the graph
myplot = nltk.FreqDist(final_words_clickbait)
myplot.plot(20, cumulative=False)
plt.figure(figsize=(16,5))
myplot.plot(50)







