import os
import sys
import pandas as pd
import numpy as np
from dataclasses import dataclass
from sklearn.model_selection import train_test_split

from src.exception import CustomException
from src.logger import logging


@dataclass
class DataIngestionConfig:
    raw_data_path: str = os.path.join("artifacts", "raw_data.csv")
    train_data_path: str = os.path.join("artifacts", "train_data.csv")
    test_data_path: str = os.path.join("artifacts", "test_data.csv")


class DataIngestion:

    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Data Ingestion Method Started")

        try:
            df = pd.read_csv("./data/trainToxic.csv")

            logging.info("Dataset read successfully")

            # create artifacts folder
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)

            # save raw data
            df.to_csv(self.ingestion_config.raw_data_path, index=False)

            logging.info("Raw data saved")

            # train test split
            train_df, test_df = train_test_split(
                df,
                test_size=0.2,
                random_state=42
            )

            logging.info("Train Test Split Completed")

            # save train data
            train_df.to_csv(self.ingestion_config.train_data_path, index=False)

            # save test data
            test_df.to_csv(self.ingestion_config.test_data_path, index=False)

            logging.info("Train and Test files saved")
            logging.info("Data Ingestion Method Completed")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":

    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()

    