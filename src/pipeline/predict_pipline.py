import os
import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object

from .create_data_for_predict import extract_features


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path=os.path.join("archive","S_model.pkl")
            preprocessor_path=os.path.join('archive','preprocessor.pkl')
            print("Before Loading")
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            print("After Loading")
            data_scaled=preprocessor.transform(features)
            preds=model.predict(data_scaled)
            print("preds", preds)
            return preds[0]
        
        except Exception as e:
            raise CustomException(e,sys)


class CustomData:
    def __init__(self):
        pass
    def get_data_as_data_frame(self, url):
        try:
            
            # # Initialize features dictionary
            features = extract_features(url)
            print("features", features)
            return pd.DataFrame(features, index=[0])

        except Exception as e:
            raise CustomException(e, sys)