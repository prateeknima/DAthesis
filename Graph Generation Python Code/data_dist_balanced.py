import re
import os
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem import WordNetLemmatizer
from imblearn.over_sampling import RandomOverSampler
import matplotlib.pyplot as plt
import seaborn as sns


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
    #print(sentence)
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

# Oversampling the data to solve the issue of data imbalance
sampler = RandomOverSampler(ratio={1: 661902, 0: 661902},random_state=0)
X_rs, y_rs = sampler.fit_sample(train_vectorized, youtube_train['tag'])
print("Stage Oversampler")

#Distribution of data after oversampling
plt.figure(figsize = (10, 8))
sns.countplot(y_rs)
plt.show()

