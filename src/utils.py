import os
import sys

import pickle

from src.exception import CustomException
from src.logger import logging

from sklearn.metrics import r2_score, accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, classification_report
from sklearn.model_selection import GridSearchCV
import numpy as np

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
    except Exception as E:
        raise CustomException(E, sys)
    


def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    try:
        report = {}

        check_model_performance = {}

        for i in range(len(models)):
            model_name = list(models.keys())[i]
            model = list(models.values())[i]
            para = param[list(models.keys())[i]]

            gs = GridSearchCV(model, para, cv=5, n_jobs=-1, scoring='accuracy')
            gs.fit(X_train,y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train,y_train)

            # model.fit(X_train, y_train)

            y_train_pred = model.predict(X_train)

            y_test_pred = model.predict(X_test)

            train_model_score = r2_score(y_train, y_train_pred)

            test_model_score = r2_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = test_model_score

            check_model_performance[list(models.keys())[i]] = [train_model_score, test_model_score]
            trained_model_file_path = os.path.join("archive", f"{list(models.keys())[i][0]}_model.pkl")
            save_object(
                file_path=trained_model_file_path,
                obj = model
            )

            # report for test data
            
            mse = mean_squared_error(y_test, y_test_pred)
            rmse = np.sqrt(mse)

            test_accuracy_score = accuracy_score(y_test, y_test_pred)
            test_precision_score = precision_score(y_test, y_test_pred)
            test_recall_score = recall_score(y_test, y_test_pred)
            test_f1_score = f1_score(y_test, y_test_pred)

            logging.info(f"---------------------{model_name}-------------------")
            logging.info(f"Mean Squared Error (MSE): {mse}")
            logging.info(f"Root Mean Squared Error (RMSE): {rmse}")
            logging.info(f"R2 score Train: {train_model_score}, Percentage : {train_model_score * 100 :.2f}%%")
            logging.info(f"R2 score Test: {test_model_score}, Percentage : {test_model_score * 100 :.2f}%%")
            logging.info(f"Accuracy: {test_accuracy_score}, Percentage : {test_accuracy_score * 100 :.2f}%")
            logging.info(f"Precision: {test_precision_score}, Percentage : {test_precision_score * 100 :.2f}%")
            logging.info(f"Recall: {test_recall_score}, Percentage : {test_recall_score * 100 :.2f}%")
            logging.info(f"F1 Score: {test_f1_score}, Percentage : {test_f1_score * 100 :.2f}%")
            logging.info(f"-----------------------------------------------------------------")
        logging.info(f"All model score on both training and test dataset : {check_model_performance}")
        return report
    except Exception as E:
        # raise CustomException(E, sys)
        pass
    

def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        # raise CustomException(e, sys)
        pass
    