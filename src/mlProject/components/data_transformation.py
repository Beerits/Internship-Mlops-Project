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

from mlProject.entity.config_entity import DataTransformationConfig




class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config

    
    ## Note: You can add different data transformation techniques such as Scaler, PCA and all
    #You can perform all kinds of EDA in ML cycle here before passing this data to the model

    # I am only adding train_test_spliting cz this data is already cleaned up


    def train_test_spliting(self):

    

        data = pd.read_csv(self.config.data_path)

        all_text = ' '.join(data['text'].values)
        all_text = re.sub(r'http\S+', '', all_text)
        all_text = re.sub(r'@\S+', '', all_text)
        all_text = re.sub(r'#\S+', '', all_text)

        words = all_text.split()
        stop_words = set(stopwords.words('english'))
        words = [word for word in words if not word in stop_words]

        word_counts = Counter(words)
        top_words = word_counts.most_common(100)

        # Split the data into training and test sets. (0.75, 0.25) split.
        train, test = train_test_split(data)

        train.to_csv(os.path.join(self.config.root_dir, "train.csv"),index = False)
        test.to_csv(os.path.join(self.config.root_dir, "test.csv"),index = False)

        logger.info("Splited data into training and test sets")
        logger.info(train.shape)
        logger.info(test.shape)

        print(train.shape)
        print(test.shape)
        