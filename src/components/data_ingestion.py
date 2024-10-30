import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass



@dataclass
class DataIngestionConfig:
    train_data_path: str=os.path.join("archive", "train.csv")
    test_data_path: str=os.path.join("archive", "test.csv")


class DataIngestion:
    def __init__(self) -> None:
        self.ingestion_config=DataIngestionConfig()

    def initiate_data_ingestion(self):

        logging.info("Entered data ingestion method or components")
        try:
            
            df = pd.read_csv('archive/dataset_phishing.csv')

            logging.info("read the dataset as dataframe")

            df['is_phishing'] = df['status'].apply(lambda x: 1 if x=='legitimate' else 0)

            selectedFeatures = ['url',
                                'length_url',
                                'length_hostname',
                                'ip',
                                'nb_dots',
                                'nb_qm',
                                'nb_eq',
                                'nb_slash',
                                'nb_www',
                                'ratio_digits_url',
                                'ratio_digits_host',
                                'tld_in_subdomain',
                                'prefix_suffix',
                                'shortest_word_host',
                                'longest_words_raw',
                                'longest_word_path',
                                'phish_hints',
                                'nb_hyperlinks',
                                'ratio_intHyperlinks',
                                'empty_title',
                                'domain_in_title',
                                'domain_age',
                                'google_index',
                                'page_rank',
                                'is_phishing']

            df = df.loc[:, selectedFeatures]

            os.makedirs(os.path.dirname(self.ingestion_config.test_data_path),exist_ok=True)


            logging.info("train_test_split initiated")

            train_set, test_set = train_test_split(df, test_size=0.30, random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)

            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("ingestion of data is completed")
            print("ingestion of data is completed")
            return{
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            }
        except Exception as E:

            raise CustomException(E,sys)
