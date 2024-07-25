import os
from mlProject import logger
from sklearn.model_selection import train_test_split
import pandas as pd

from collections import Counter
import re

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
nltk.download('stopwords')
from nltk.corpus import stopwords

from tqdm.auto import tqdm
import time

from sklearn.feature_extraction.text import CountVectorizer


from mlProject.entity.config_entity import DataTransformationConfig




class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config

    
    ## Note: You can add different data transformation techniques such as Scaler, PCA and all
    #You can perform all kinds of EDA in ML cycle here before passing this data to the model

    # I am only adding train_test_spliting cz this data is already cleaned up


    def train_test_spliting(self):

    

        data = pd.read_csv(self.config.data_path)
        data['text'].drop_duplicates(inplace = True)
        all_text = ' '.join(data['text'].values)
        all_text = re.sub(r'http\S+', '', all_text)
        all_text = re.sub(r'@\S+', '', all_text)
        all_text = re.sub(r'#\S+', '', all_text)

        words = all_text.split()
        stop_words = set(stopwords.words('english'))
        words = [word for word in words if not word in stop_words]

        def clean_text(text):
            text = re.sub('<.*?>', '', text)

            text = re.sub('[^a-zA-Z]', ' ', text).lower()
            words = nltk.word_tokenize(text)
            words = [w for w in words if w not in stopwords.words('english')]
            stemmer = PorterStemmer()
            words = [stemmer.stem(w) for w in words]
            text = ' '.join(words)
            return text

        nltk.download('punkt')

        tqdm.pandas()

        data['cleaned_text'] = data['text'].progress_apply(clean_text)

        cv = CountVectorizer(max_features=5000)
        X = cv.fit_transform(data['cleaned_text']).toarray()
        y = data['spam']

        # Split the data into training and test sets. (0.75, 0.25) split.
        # train, test = train_test_split(data)

        # Split the data into training and test sets. (0.8, 0.2) split.
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Convert the split data back to DataFrames
        train = pd.DataFrame(X_train)
        train['spam'] = y_train
        test = pd.DataFrame(X_test)
        test['spam'] = y_test

        train.to_csv(os.path.join(self.config.root_dir, "train.csv"),index = False)
        test.to_csv(os.path.join(self.config.root_dir, "test.csv"),index = False)

        logger.info("Splited data into training and test sets")
        logger.info(train.shape)
        logger.info(test.shape)

        print(train.shape)
        print(test.shape)
        
