# Streamlit dependencies
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
import streamlit as st

# Load the data
train_df = pd.read_csv('Data/train_set.csv')
test_df = pd.read_csv('Data/test_set.csv')

#change names
map_lang = {"xho": "Xhosa", "eng": "English", "nso": "Northen Sotho", "ven": "",
                 "tsn": "Tswana", "nbl": "Ndebele", "zul": "Zulu", "ssw": "Swazi",
                 "tso": "Tsonga", "sot": "Sotho", "afr": "Afrikaans"}

train_df.replace(map_lang,inplace=True)

# Convert the text into lower case
train_df['text'] = train_df['text'].str.lower() 
test_df['text'] = test_df['text'].str.lower() 

# Remove punctuation marks from the dataset
def punc_remover(text):
    return ''.join([l for l in text if l not in string.punctuation])

train_df['text'] = train_df['text'].apply(punc_remover)
test_df['text'] = test_df['text'].apply(punc_remover)

# Separate features and labels
corpus = train_df['text']
y = train_df['lang_id']
corpus_test = test_df['text']

# Label encode the y variable
le = LabelEncoder()
y = le.fit_transform(y)

# Convert X variable to numerical using bag of words
cv = CountVectorizer()
X = cv.fit_transform(corpus)
X_pred = cv.transform(corpus_test)

# Test train split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

# Build model
model = MultinomialNB()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)


# App declaration
def main():
    # Adding elements to the sidebar
    page_options = ["Home", "Data Overview"]
    page_selection = st.sidebar.selectbox("Choose Page", page_options)

    if page_selection == "Home":
        st.title('South Language Detection App')
        st.info('''
        South Africa is a multicultural society that is characterised by its rich linguistic diversity.
        This app detects the language of a given South African language.
        ''')
        
        st.write("---")
        # Create text box
        user_input = st.text_area("Input Text", "Type Here")
        if st.button('Detect Language'):
            processed_input = punc_remover(user_input.lower())
            X_input = cv.transform([processed_input])
            predicted_language = le.inverse_transform(model.predict(X_input))
            if len(predicted_language) > 0:
                st.write(f"The detected language is: {predicted_language[0]}")
            else:
                st.write("Language not recognized")

    if page_selection == "Data Overview":
        st.title("Data Information")
        st.write('''
        This app is still in development. The model needs more hypertuning to improve it's accuracy.
        In the meantime, you can find the script on my [github](https://github.com/ChisomNwankwo/South-African-Language-Identification-Hack-2022)
        ''')


# Run the app
if __name__ == "__main__":
    main()
