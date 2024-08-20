

import joblib
import numpy as np
import pandas as pd
from pathlib import Path
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer

# Download necessary NLTK data files
nltk.download('stopwords')
nltk.download('punkt')

class PredictionPipeline:
    def __init__(self):
        # Load the model and vectorizer
        self.model = joblib.load(Path('artifacts/model_trainer/model.joblib'))
        self.vectorizer = joblib.load(Path('artifacts/vectorizer.joblib'))

        # Initialize the text cleaning components
        self.stop_words = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()

    def preprocess(self, text):
        # Preprocess the text data
        def clean_text(text):
            text = re.sub('<.*?>', '', text)
            text = re.sub('[^a-zA-Z]', ' ', text).lower()
            words = nltk.word_tokenize(text)
            words = [w for w in words if w not in self.stop_words]
            words = [self.stemmer.stem(w) for w in words]
            return ' '.join(words)

        # Clean the text and transform it into a feature vector
        cleaned_text = clean_text(text)
        text_vector = self.vectorizer.transform([cleaned_text]).toarray()
        return text_vector

    def predict(self, text):
        # Preprocess the text
        data = self.preprocess(text)
        
        # Make prediction
        prediction = self.model.predict(data)
        return prediction
