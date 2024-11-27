from src.accuracy_utils import DataTransformaionForAccuracy, ShowAccuracy


from src.logger import logging

def main():
    logging.info("Starting the project")
    obj = DataTransformaionForAccuracy()
    train_arr, test_arr = obj.data_transformation()

    modelTrainer = ShowAccuracy()
    model_score = modelTrainer.checkAccuracy(train_arr, test_arr)

    print("model_score", model_score)

if __name__ == "__main__":
    main()