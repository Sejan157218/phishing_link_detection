from src.pipeline.predict_pipline import CustomData, PredictPipeline


from src.logger import logging

def main():
    get_data = CustomData()
    feature = get_data.get_data_as_data_frame("https://whois.whoisxmlapi.com")
    predict = PredictPipeline()
    predicted_result = predict.predict(feature)
    print("feature", predicted_result)

if __name__ == "__main__":
    main()