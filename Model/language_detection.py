import string 
import re
import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import feature_extraction
from sklearn import linear_model
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import pipeline
from sklearn.model_selection import train_test_split
from sklearn import metrics
import itertools

# Custom Libraries



#import data
train_df=pd.read_csv('Data/train_set.csv')
test_df=pd.read_csv('Data/test_set.csv')

#Convert the text into lower case
train_df['text'] = train_df['text'].str.lower() 
test_df['text'] = test_df['text'].str.lower() 

#Remove punctuation marks from the dataset
def punc_remover(text):
    return ''.join([l for l in text if l not in string.punctuation])

train_df['text'] = train_df['text'].apply(punc_remover)
test_df['text'] = test_df['text'].apply(punc_remover)

#separate to features and columns
corpus = train_df['text']
y = train_df['lang_id']
corpus_test=test_df['text']

#label encode the y variable
le = LabelEncoder()
y = le.fit_transform(y)

#convert X varible to numerical using bag of words
cv = CountVectorizer()
X= cv.fit_transform(corpus)
X_pred=cv.transform(corpus_test)

#test train split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)
 
#build model
model = MultinomialNB()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

