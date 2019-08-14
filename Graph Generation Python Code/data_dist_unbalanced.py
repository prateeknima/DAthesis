import re
import os
import pandas as pd
from nltk.corpus import stopwords
import nltk
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem import WordNetLemmatizer
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


#Distribution of data before oversampling
plt.figure(figsize = (10, 8))
sns.countplot(youtube_train['tag'])
plt.show()