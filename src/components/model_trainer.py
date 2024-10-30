import os
import sys
from dataclasses import dataclass


from sklearn.metrics import r2_score

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

from src.exception import CustomException
from src.logger import logging
from src.utils import evaluate_models, save_object

@dataclass
class ModeltrainerConfig:
    trained_model_file_path = os.path.join("archive", "model.pkl")
    

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModeltrainerConfig()
    
    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("split training and test input data")
            X_train, y_train, X_test, y_test =(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            models = {
                'Logistic Regression': LogisticRegression(),
                'Random Forest': RandomForestClassifier(),
                'Gradient Boosting': GradientBoostingClassifier(),
                'SVM': SVC(),
                'KNN': KNeighborsClassifier(),
                'XGBClassifier': XGBClassifier()
            }

            params={
                'Logistic Regression': {
                    'C': [0.1, 1, 10],
                    'solver': ['liblinear', 'lbfgs'],
                    'max_iter': [100, 200, 500]
                },
                'Random Forest': {
                    'n_estimators': [100, 150],
                    'max_depth': [10, 20],
                    'min_samples_split': [2, 5]
                },
                'Gradient Boosting': {
                    'n_estimators': [100, 200],
                    'learning_rate': [0.01, 0.1],
                    'max_depth': [3, 5],
                },
                'SVM': {
                    'C': [0.1, 1, 10],
                    'kernel': ['linear', 'rbf']
                },
                'KNN': {
                    'n_neighbors': [3, 5, 7, 9],
                    'p': [1, 2]  # 1 for Manhattan, 2 for Euclidean
                },
                'XGBClassifier' : {
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5],
                    'n_estimators': [100, 200]
                }
                
            }

            model_report:dict=evaluate_models(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                                              models=models, param=params)
            print("model_report", model_report)
            ## To get best model score from dict

            best_model_score = max(sorted(model_report.values()))

            ## To get best model name from dict
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            if best_model_score<0.6:
                raise CustomException("No best Model found")
            
            logging.info(f"Best found model {best_model} on both training and test dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj = best_model
            )

            predicted = best_model.predict(X_test)

            r2_square = r2_score(y_test, predicted)

            return r2_square

        except Exception as E:
            raise CustomException(E, sys)