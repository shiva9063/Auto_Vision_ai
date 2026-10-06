from django.urls import path
from . import views

urlpatterns = [
    path("", views.signup_view, name="signup"),
    path("login/", views.login_view, name="login"),
    path('home/',views.home,name='home'),
    path("dataset-load/", views.Dataset_load, name="dataset_load"),
    path('preprocessing/',views.Data_preprocessing,name='data_preprocessing'),
    path('model_training/',views.Model_training,name='model_training'),
    path('prediction/',views.Prediction,name='prediction')
] 