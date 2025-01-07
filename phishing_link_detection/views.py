from django.shortcuts import render
from django.views.generic import TemplateView
from src.pipeline.predict_pipline import CustomData, PredictPipeline
from rest_framework.views import APIView
from rest_framework import status
from rest_framework.response import Response


def Home(request):
   
    return render(request, 'input.html')

class PredictResult(TemplateView):
    template_name = 'result.html'
    def post(self, request):
        url  = request.POST['url']
        print("url________", url)
        result = -1
        if url:
            get_data = CustomData()
            feature = get_data.get_data_as_data_frame(url)
            if feature is None:
                result = "Website not allowing web scraping"
                return render(request,'result.html', {"predicted_result": result})
            # print("feature", feature)
            predict = PredictPipeline()
            predicted_result = predict.predict(feature)
            if predicted_result==0:
                result = "This link or website is Not Safe"
            else:
                result = "This link or website is Safe"
            # print("featursssse", predicted_result)
            return render(request,'result.html', {"predicted_result": result})
        else:
            result = "Please provide a url"
            return render(request,'result.html', {"predicted_result": result})


class PredictApi(APIView):

    def post(self, request, *args, **kwargs):
        request_data = request.data
        url = request_data['url']
        # print("url___________________", url)
        try:
            if url:
                get_data = CustomData()
                feature = get_data.get_data_as_data_frame(url)
        
                if feature is None:
                    return Response({"status":"ok","message": "Website not allowing web scraping"}, status=status.HTTP_423_LOCKED)
                predict = PredictPipeline()
                predicted_result = predict.predict(feature)
                if predicted_result==0:
                    result = "This link or website is Not Safe"
                else:
                    result = "This link or website is Safe"  
                return Response({"status":"ok","message": result}, status=status.HTTP_200_OK)
            else:
                result = "Please provide a url"
                return Response({"status":"ok","message": result}, status=status.HTTP_406_NOT_ACCEPTABLE)
        except Exception as E:
            return Response({"status":"ok","message": "Please try again later"}, status=status.HTTP_400_BAD_REQUEST)