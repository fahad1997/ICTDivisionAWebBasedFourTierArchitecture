from django.contrib import admin
from django.urls import path,include
from . import views

urlpatterns = [
    path('admin/',admin.site.urls),
    path('',views.home,name='Home'),
    path('classify/',views.classify,name="Classify")
]
