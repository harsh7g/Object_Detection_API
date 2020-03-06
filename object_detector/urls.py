from django.urls import path
from . import views

urlpatterns = [
    path('GetImageDetails/', views.o_detect),
    path('GetImageDetailsOI/',views.o_detect_oi),
]