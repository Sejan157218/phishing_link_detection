from django.shortcuts import render
from django.views.generic import TemplateView, UpdateView, DetailView
from django.http import HttpResponseRedirect
from django.urls import reverse
import pandas as pd
from src.pipeline.predict_pipline import CustomData, PredictPipeline

def Home(request):
   
    return render(request, 'index.html')

class PredictResult(TemplateView):
    def post(self, request):
        url  = request.POST['url']
        result = ''
        if url:
            get_data = CustomData()
            feature = get_data.get_data_as_data_frame(url)
            print("feature", feature)
            predict = PredictPipeline()
            predicted_result = predict.predict(feature)
            if predicted_result==0:
                result = "This link or website is Not Save"
            else:
                result = "This link or website is Save"
            print("featursssse", predicted_result)

        return render(request,'index.html', {"predicted_result": result})