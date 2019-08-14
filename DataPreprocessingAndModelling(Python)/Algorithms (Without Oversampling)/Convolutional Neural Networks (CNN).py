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
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np
from keras.models import Sequential
from keras import layers
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from  sklearn import metrics

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

#Converting the sentences to string format
youtube_train_sen = youtube_train['video_title'].values
youtube_train_sen = youtube_train_sen.astype(str)

#splitting the data for training and testing
x_train,x_test,y_train,y_test = ttsplit(youtube_train_sen,youtube_train['tag'].values,test_size=0.25)

#Initializing tokenizer with max number of words set to 5000
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(x_train)

# fit the words to tokenized library
X_train = tokenizer.texts_to_sequences(x_train)
X_test = tokenizer.texts_to_sequences(x_test)

# Adding 1 because of  reserved 0 index
vocab_size = len(tokenizer.word_index) + 1

maxlen = 100

#Convert all sequences to equal lenghts with max lenght set to 100
X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)


def create_embedding_matrix(filepath, word_index, embedding_dim):
    vocab_size = len(word_index) + 1
    # Adding again 1 because of reserved 0 index
    embedding_matrix = np.zeros((vocab_size, embedding_dim))

    with open(filepath,encoding="utf8") as f:
        for line in f:
            word, *vector = line.split()
            if word in word_index:
                idx = word_index[word]
                embedding_matrix[idx] = np.array(vector, dtype=np.float32)[:embedding_dim]
    return embedding_matrix

#Creating a embedding matrix
embedding_dim = 50
embedding_matrix = create_embedding_matrix('C:/Users/prate/Desktop/thesis_data_1/glove.6B.50d.txt' ,tokenizer.word_index,embedding_dim)

embedding_dim = 100

#Initializing the model
model = Sequential()

# Adding the embedding layers with an input size= vocab_size, embedding dim = size of vector space
model.add(layers.Embedding(vocab_size, embedding_dim, input_length=maxlen))

model.add(layers.Conv1D(128, 5, activation='relu')) #converting it to 1D
model.add(layers.GlobalMaxPooling1D())
model.add(layers.Dense(10, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
#fit to the model with a batch size of 10
history = model.fit(X_train, y_train,epochs=2,validation_data=(X_test, y_test),batch_size=1-)

#history

train_acc = model.evaluate(X_train, y_train, verbose=0)
test_acc = model.evaluate(X_test, y_test, verbose=0)

# predict probabilities for test set
yhat_probs = model.predict(X_test, verbose=0)
# predict crisp classes for test set
yhat_classes = model.predict_classes(X_test, verbose=0)
# reduce to 1d array
yhat_probs = yhat_probs[:, 0]
yhat_classes = yhat_classes[:, 0]

# accuracy: (tp + tn) / (p + n)
accuracy = accuracy_score(y_test, yhat_classes)
print('Accuracy: %f' % accuracy)
# precision tp / (tp + fp)
precision = precision_score(y_test, yhat_classes)
print('Precision: %f' % precision)
# recall: tp / (tp + fn)
recall = recall_score(y_test, yhat_classes)
print('Recall: %f' % recall)
# f1: 2 tp / (2 tp + fp + fn)
f1 = f1_score(y_test, yhat_classes)
print('F1 score: %f' % f1)

cm = metrics.confusion_matrix(y_test, yhat_classes)
cm[1][1]

#Creating confusion metrics
cm = metrics.confusion_matrix(y_test, yhat_classes)
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