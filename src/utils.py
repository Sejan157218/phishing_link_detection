import os
import sys


import pickle

from src.exception import CustomException
from src.logger import logging

from sklearn.metrics import r2_score, accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, classification_report
from sklearn.model_selection import GridSearchCV
import numpy as np
import pandas as pd

#  ANN
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.regularizers import l2
from keras.layers import Dense, BatchNormalization, Dropout, LSTM, Embedding, Input
from keras.models import Sequential
from keras.utils import to_categorical
from keras.optimizers import Adam
from tensorflow.keras import regularizers
from sklearn.metrics import precision_score, recall_score, confusion_matrix, classification_report, accuracy_score, f1_score
from keras import callbacks
import matplotlib.pyplot as plt




np.random.seed(0)

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



def evaluate_models_ANN(X_train, y_train, X_test, y_test):
    # try:

        # Define the model
        model = Sequential()
        model.add(Input(shape=(23,)))  # Input layer

        # Hidden layers with Dropout and L2 Regularization
        model.add(Dense(units=128, activation='relu', kernel_regularizer=l2(0.01)))
        model.add(Dropout(0.5))  # Dropout with a rate of 50%
        model.add(Dense(units=64, activation='relu', kernel_regularizer=l2(0.01)))
        model.add(Dropout(0.5))
        model.add(Dense(units=32, activation='relu', kernel_regularizer=l2(0.01)))

        # Output layer
        model.add(Dense(units=1, activation='sigmoid'))

        # Compile the model
        opt = Adam(learning_rate=0.00009)
        model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

        # Early Stopping Callback
        early_stopping = tf.keras.callbacks.EarlyStopping(
            min_delta=0.001,  # Minimum improvement to qualify as an epoch improvement
            patience=20,  # Stop after 10 epochs without improvement
            restore_best_weights=True,
        )

        # Train the model
        history = model.fit(
            X_train, y_train,
            batch_size=32,
            epochs=150,
            validation_split=0.2,
            callbacks=[early_stopping]
        )

        # Evaluate the model on the test set
        y_pred = model.predict(X_test) > 0.5  # Predict probabilities and convert to binary

        # Calculate Mean Squared Error (MSE)
        mse = mean_squared_error(y_test, y_pred)

        # Calculate Root Mean Squared Error (RMSE)
        rmse = np.sqrt(mse)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        print(f"Accuracy: {accuracy:.2f}")
        print(f"Precision: {precision:.2f}")
        print(f"Recall: {recall:.2f}")
        print(f"F1 Score: {f1:.2f}")

        logging.info(f"----------------------------------------------------------------")
        logging.info(f"Mean Squared Error (MSE): {mse}")
        logging.info(f"Root Mean Squared Error (RMSE): {rmse}")
        logging.info(f"Accuracy: {accuracy}")
        logging.info(f"Precision: {precision}")
        logging.info(f"Recall: {recall}")
        logging.info(f"F1 Score: {f1}")
        logging.info(f"-----------------------------------------------------------------")

        trained_model_file_path = os.path.join("archive", "ANN_model.pkl")
        trained_model_history_file_path = os.path.join("archive", "ANN_model_history.pkl")
        save_object(
            file_path=trained_model_file_path,
            obj = model
        )

        save_object(
            file_path=trained_model_history_file_path,
            obj = history.history
        )

    #     return None
    # except Exception as E:
    #     # raise CustomException(E, sys)
    #     pass


def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        # raise CustomException(e, sys)
        pass
    