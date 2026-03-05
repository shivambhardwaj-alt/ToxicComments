import os
import re
import string
import sys
import pandas as pd
import logging
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from src.exception import CustomException
from src.logger import logging

class DataTransformation:

    CONTRACTIONS = {
        "can't": "cannot", "shan't": "shall not", "won't": "will not", "n't": " not",
        "i'm": "i am", "what's": "what is", "let's": "let us", "'re": " are",
        "'s": " ", "'ve": " have", "'ll": " will", "'scuse": " excuse"
    }

    LABELS = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

    def __init__(self):
        self.transformed_folder = "Train_and_Test"
        os.makedirs(self.transformed_folder, exist_ok=True)

    def clean_text(self, text):
        text = text.lower()
        for k, v in self.CONTRACTIONS.items():
            text = re.sub(k, v, text)
        text = re.sub(r'[^a-z\s]', ' ', text)  
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def initiate_data_transformation(self):

        try:
            logging.info("Data Transformation Started")

            # Load train and test data
            train_data = pd.read_csv(os.path.join("artifacts", "train_data.csv"))
            test_data = pd.read_csv(os.path.join("artifacts", "test_data.csv"))

            logging.info("Cleaning train and test comments")
            train_data['comment_text'] = train_data['comment_text'].apply(self.clean_text)
            test_data['comment_text'] = test_data['comment_text'].apply(self.clean_text)

            logging.info("TF-IDF Vectorization Started")
            tfd = TfidfVectorizer(stop_words="english", ngram_range=(1,2), max_features=5000, min_df=2)
            TEXT_COL = "comment_text"

            X_train_tfd = tfd.fit_transform(train_data[TEXT_COL])
            X_test_tfd = tfd.transform(test_data[TEXT_COL])
            logging.info("TF-IDF Vectorization Completed")
            logging.info(f"Train shape: {X_train_tfd.shape}, Test shape: {X_test_tfd.shape}")



            logging.info("Converting sparse matrices to DataFrames and saving as CSV")
            # Convert sparse matrix to DataFrame safely
            transformed_train_data = pd.DataFrame.sparse.from_spmatrix(
                X_train_tfd, columns=tfd.get_feature_names_out()
            )
            transformed_test_data = pd.DataFrame.sparse.from_spmatrix(
                X_test_tfd, columns=tfd.get_feature_names_out()
            )
            logging.info("Sparse matrices converted to DataFrames")
            logging.info("Adding label columns to transformed data")

            
            for label in self.LABELS:
                transformed_train_data[label] = train_data[label]
                if label in test_data.columns:
                    transformed_test_data[label] = test_data[label]  

            # Save CSV

            logging.info("Saving transformed data to CSV")
            transformed_train_data.to_csv(os.path.join(self.transformed_folder, "transformed_train_data.csv"), index=False)
            transformed_test_data.to_csv(os.path.join(self.transformed_folder, "transformed_test_data.csv"), index=False)

            logging.info("Transformed data saved successfully")

        except Exception as e:
            logging.error(f"Error during data transformation: {e}")
            raise CustomException(e, sys)

if __name__ == "__main__":
    data = DataTransformation()
    data.initiate_data_transformation()