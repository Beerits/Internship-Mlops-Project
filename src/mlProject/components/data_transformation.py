import os
from mlProject import logger
from sklearn.model_selection import train_test_split
import pandas as pd

# Import necessary modules and libraries for preprocessing
import re
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
nltk.download('stopwords')  # Download stopwords for text cleaning

from tqdm.auto import tqdm  # For progress bars during data processing

from sklearn.feature_extraction.text import CountVectorizer  # For converting text to numerical data

from mlProject.entity.config_entity import DataTransformationConfig  # Importing custom config entity


class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        # Initialize the configuration for data transformation
        self.config = config

    
    def train_test_spliting(self):
        # Read the data from the specified CSV file
        data = pd.read_csv(self.config.data_path)
        
        # Drop duplicate rows based on the 'text' column
        data['text'].drop_duplicates(inplace=True)

        # Clean the text data by removing URLs, mentions, and hashtags
        all_text = ' '.join(data['text'].values)
        all_text = re.sub(r'https?://\S+', '', all_text)  # Remove URLs
        all_text = re.sub(r'@\S+', '', all_text)  # Remove mentions (@username)
        all_text = re.sub(r'#\S+', '', all_text)  # Remove hashtags

        # Tokenize the text and remove stopwords
        words = all_text.split()
        stop_words = set(stopwords.words('english'))
        words = [word for word in words if word not in stop_words]

        # Define a function to clean individual pieces of text
        def clean_text(text):
            # Remove HTML tags and non-alphabetical characters, lowercase the text
            text = re.sub('<.*?>', '', text)
            text = re.sub('[^a-zA-Z]', ' ', text).lower()

            # Tokenize and remove stopwords, then apply stemming
            words = nltk.word_tokenize(text)
            words = [w for w in words if w not in stopwords.words('english')]
            stemmer = PorterStemmer()
            words = [stemmer.stem(w) for w in words]

            # Join the cleaned words back into a string
            text = ' '.join(words)
            return text

        nltk.download('punkt')  # Download necessary data for tokenization

        tqdm.pandas()  # Initialize tqdm for progress tracking

        # Apply the cleaning function to all text data and create a new column 'cleaned_text'
        data['cleaned_text'] = data['text'].progress_apply(clean_text)

        # Convert the cleaned text data into numerical features using CountVectorizer
        cv = CountVectorizer(max_features=5000)
        X = cv.fit_transform(data['cleaned_text']).toarray()  # Transform text into a feature matrix
        y = data['spam']  # Target variable indicating whether the text is spam

        # Save the vectorizer for future use
        joblib.dump(cv, 'artifacts/vectorizer.joblib')

        # Split the data into training and testing sets (80% train, 20% test)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Convert the training and test data into DataFrames and add the target variable back
        train = pd.DataFrame(X_train)
        train['spam'] = y_train
        test = pd.DataFrame(X_test)
        test['spam'] = y_test

        # Save the training and testing datasets as CSV files
        train.to_csv(os.path.join(self.config.root_dir, "train.csv"), index=False)
        test.to_csv(os.path.join(self.config.root_dir, "test.csv"), index=False)

        # Log the completion of the split and the shape of the resulting datasets
        logger.info("Splitted data into training and test sets")
        logger.info(train.shape)
        logger.info(test.shape)

        # Print the shape of the train and test datasets for quick verification
        print(train.shape)
        print(test.shape)
