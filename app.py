import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer
import os


secret_key = os.getenv("SECRET_KEY")


nltk.download('punkt', force=True)
nltk.download('stopwords', force=True)


try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')

#vectorizer and model
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
            
    text = y[:]
    y.clear()   
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)  
    text = y[:]
    y.clear() 
    
    for i in text:
        y.append(ps.stem(i))        
              
    return " ".join(y)       

st.title("SMS SPAM CLASSIFIER")

input_sms = st.text_input("Enter the message")

if st.button('Predict'):
    # Preprocess
    transformed_sms = transform_text(input_sms)
    # Vectorize
    vector_input = tfidf.transform([transformed_sms])
    # Predict
    result = model.predict(vector_input)[0]
    # Display
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")
