from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformaion
from src.components.model_trainer import ModelTrainer


from src.logger import logging

def main():
    logging.info("Starting the project")
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()

    data_transformation = DataTransformaion()
    train_arr, test_arr = data_transformation.initiate_data_transformation(train_data, test_data)

    modelTrainer = ModelTrainer()
    model_score = modelTrainer.initiate_model_trainer(train_arr, test_arr)

    print("model_score", model_score)

if __name__ == "__main__":
    main()