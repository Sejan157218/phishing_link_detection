from src.logger import logging
from sklearn.metrics import r2_score, accuracy_score, precision_score, recall_score, f1_score,mean_squared_error, classification_report
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os
from src.exception import CustomException
import sys
from src.utils import load_object


class DataTransformaionForAccuracy:
    def __init__(self):
        self.file_name = ['train.csv', 'test.csv']
        self.preprocessing = 'preprocessor.pkl'


    def data_transformation(self):

        try:
            train_path = os.path.join("archive", self.file_name[0])
            test_path = os.path.join("archive", self.file_name[1])
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
            scaler = load_object(os.path.join("archive", self.preprocessing))

            input_feature_train_df = scaler.transform(input_feature_train_df)
            input_feature_test_df = scaler.transform(input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_df, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[
                input_feature_test_df, np.array(target_feature_test_df)
            ]


            logging.info(f"complete data processing for accuracy")

            return (
                train_arr,
                test_arr,
            )

        except Exception as E:
            raise CustomException(E, sys)


class ShowAccuracy:
    def __init__(self):
        self.model_name = ['Logistic Regression', 'Random Forest', 'Gradient Boosting', 'SVM', 'KNN', 'XGBClassifier']
    
    def checkAccuracy(self, train_array, test_array):
        try:
            logging.info("split training and test input data")
            X_train, y_train, X_test, y_test =(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            report = {}

            model_performance_in_percentage = {}

            for model_name in self.model_name:
                
                model = load_object(os.path.join("archive", f"{model_name[0]}_model.pkl"))

                y_train_pred = model.predict(X_train)

                y_test_pred = model.predict(X_test)

                train_model_score = r2_score(y_train, y_train_pred)

                train_model_score_percentage = train_model_score * 100

                train_model_score = f"{train_model_score:.2f}"
                train_model_score_percentage = f"{train_model_score_percentage:.2f}%"

                test_model_score = r2_score(y_test, y_test_pred)

                test_model_score_percentage = test_model_score * 100

                test_model_score = f"{test_model_score:.2f}"

                test_model_score_percentage = f"{test_model_score_percentage:.2f}%"

                report[model_name] = [train_model_score,test_model_score]

                # report for test data
                
                mse = mean_squared_error(y_test, y_test_pred)
                rmse = np.sqrt(mse)

                test_accuracy_score = accuracy_score(y_test, y_test_pred)
                test_precision_score = precision_score(y_test, y_test_pred)
                test_recall_score = recall_score(y_test, y_test_pred)
                test_f1_score = f1_score(y_test, y_test_pred)

                model_performance_in_percentage[model_name] = [train_model_score_percentage, test_model_score_percentage]

                logging.info(f"---------------------{model_name}-------------------")
                logging.info(f"Mean Squared Error (MSE): {mse}")
                logging.info(f"Root Mean Squared Error (RMSE): {rmse}")
                logging.info(f"R2 score Train: {train_model_score}, Percentage : {train_model_score_percentage}%")
                logging.info(f"R2 score Test: {test_model_score}, Percentage : {test_model_score_percentage}%")
                logging.info(f"Accuracy: {test_accuracy_score}, Percentage : {test_accuracy_score * 100 :.2f}%")
                logging.info(f"Precision: {test_precision_score}, Percentage : {test_precision_score * 100 :.2f}%")
                logging.info(f"Recall: {test_recall_score}, Percentage : {test_recall_score * 100 :.2f}%")
                logging.info(f"F1 Score: {test_f1_score}, Percentage : {test_f1_score * 100 :.2f}%")
                logging.info(f"-----------------------------------------------------------------")

            
            logging.info(f"All model score on both training and test dataset : {report}")
            logging.info(f"All model score on both training and test dataset in percentage: {model_performance_in_percentage}")

            return report
        except Exception as E:
            raise CustomException(E, sys)
     