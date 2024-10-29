import os
import sys

import pickle

from src.exception import CustomException
from src.logger import logging

from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV


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
            model = list(models.values())[i]
            para = param[list(models.keys())[i]]

            gs = GridSearchCV(model, para, cv=5, scoring='accuracy')
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
        logging.info(f"All model score on both training and test dataset : {check_model_performance}")
        return report
    except Exception as E:
        raise CustomException(E, sys)