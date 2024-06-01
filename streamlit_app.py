import streamlit as st
import pickle
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import nltk
import string
from nltk.stem.porter import PorterStemmer
from PIL import Image
import os

nltk.download('punkt')
nltk.download('stopwords')

ps = PorterStemmer()

image_path = os.path.join('me2.png')
image = Image.open(image_path)
st.image(image, caption='Spam Classifier')

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    text = [i for i in text if i.isalnum()]
    text = [i for i in text if i not in stopwords.words('english') and i not in string.punctuation]
    text = [ps.stem(i) for i in text]
    return " ".join(text)

spam_vectorizer_path = os.path.join('vectorizer.pkl')
spam_model_path = os.path.join('model.pkl')

with open(spam_vectorizer_path, 'rb') as f:
    spam_tfidf = pickle.load(f)

with open(spam_model_path, 'rb') as f:
    spam_model = pickle.load(f)

if not isinstance(spam_tfidf, TfidfVectorizer):
    raise ValueError("The loaded vectorizer is not a TfidfVectorizer instance")

email_model_path = 'my_model.h5'
tokenizer_path = 'tokenizer.pkl'

if not os.path.exists(email_model_path):
    st.error(f"Model file not found at {email_model_path}")
if not os.path.exists(tokenizer_path):
    st.error(f"Tokenizer file not found at {tokenizer_path}")

email_model = load_model(email_model_path)
with open(tokenizer_path, 'rb') as handle:
    email_tokenizer = pickle.load(handle)

def preprocess_text(text, tokenizer, max_len):
    text = text.lower()
    sequences = tokenizer.texts_to_sequences([text])
    padded_sequences = pad_sequences(sequences, maxlen=max_len, padding='post')
    return padded_sequences

st.title('Email and SMS Spam Classifier')

tab_email, tab_sms = st.tabs(["Email Spam Detection", "SMS Spam Detection"])

with tab_email:
    st.subheader("Email Spam Detection")
    email_input = st.text_area("Enter the email content:")
    if st.button("Check Email"):
        if email_input:
            processed_input = preprocess_text(email_input, email_tokenizer, max_len=500)
            prediction = email_model.predict(processed_input)
            is_spam = (prediction > 0.5).astype("int32")[0][0]
            
            if is_spam:
                st.error("Warning: This email is likely spam!")
            else:
                st.success("This email seems to be safe.")
        else:
            st.error("Please enter email content to check.")

with tab_sms:
    st.subheader("SMS Spam Detection")
    input_sms = st.text_input('Enter the SMS')
    if st.button('Predict Spam SMS'):
        if len(input_sms.split()) < 6:
            st.error("The SMS should contain at least 6 words.")
        else:
            transform_sms = transform_text(input_sms)
            vector_input = spam_tfidf.transform([transform_sms])
            result = spam_model.predict(vector_input)[0]

            if result == 1:
                st.error("Spam SMS")
            else:
                st.success('Not Spam SMS')
