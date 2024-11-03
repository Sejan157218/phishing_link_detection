from django.shortcuts import render
from django.views.generic import TemplateView
from src.pipeline.predict_pipline import CustomData, PredictPipeline

def Home(request):
   
    return render(request, 'input.html')

class PredictResult(TemplateView):
    template_name = 'result.html'
    def post(self, request):
        url  = request.POST['url']
        result = -1
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
            return render(request,'result.html', {"predicted_result": result})
        else:
            result = "Please provide a url"
            return render(request,'result.html', {"predicted_result": result})