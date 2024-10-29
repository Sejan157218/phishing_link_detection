import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd


from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from src.exception import CustomException
from src.logger import logging



@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')


class DataTransformaion:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()


    def  initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info('Read train and test data completed')

            unnecessary_column_name = 'url'
        
            train_df = train_df.drop(columns=[unnecessary_column_name], axis=1)
            test_df = test_df.drop(columns=[unnecessary_column_name], axis=1)

            target_column_name = 'is_phishing'

            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info(
                f'Applying preprocessing on train dataframe and test dataframe'
            )


            train_arr = np.c_[
                input_feature_train_df, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[
                input_feature_test_df, np.array(target_feature_test_df)
            ]

            logging.info(f"Saved preprocessing object . ")

            return (
                train_arr,
                test_arr,
            )


        except Exception as E:
            raise CustomException(E, sys)