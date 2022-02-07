from django.shortcuts import render, redirect
from django.http import HttpRequest, HttpResponse
from django.http import JsonResponse
import json
import numpy as np
from keras.models import model_from_json


def home(request):
    return render(request, 'home.html')


def classify(request):
    file = open('student_prediction_model.json', 'r')
    model_json = file.read()
    file.close()

    loaded_model = model_from_json(model_json)
    loaded_model.load_weights('student_prediction_model.h5')

    goThroughCourseMaterial = int(request.POST["goThroughCourseMaterial"])
    StudentAbsenceDays = int(request.POST["StudentAbsenceDays"])
    questionsAskedInTheClassroom = int(request.POST["questionsAskedInTheClassroom"])
    print(goThroughCourseMaterial)

    prediction = np.argmax(loaded_model.predict([[goThroughCourseMaterial,StudentAbsenceDays,questionsAskedInTheClassroom]]),axis=1)

    if(prediction[0]==0):
        result = "High"
    elif(prediction[0]==1):
        result = "Low"
    else:
        result = "Medium"
    
    return HttpResponse('<center><h2>Your class performance is </h2><h1 class="deep-purple-text">'+result+'</h1><br><h2>To improve performance you have to go through the course material,<br> present in the class and ask questions to the teacher in the classroom</h2></center>')


