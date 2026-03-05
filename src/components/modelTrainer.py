# here is the code to train the model and evaluate the model performance using the transformed data


import os 
import sys
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from src.exception import CustomException
from src.logger import logging


class ModelTrainer:
    def __init__(self):
        pass
    def intitiate_model_trainer(self):
        try:
            logging.info("Model Training Started")
            # Load transformed train and test data

        except Exception as e : 
            logging.info("Error during model training")

            raise CustomException(e,sys)
    