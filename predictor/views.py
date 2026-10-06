from django.shortcuts import render,redirect
from predictor.models import User
from pathlib import Path
from ml.preprocessing import preprocessing_dataset
from ml.train import model_training
from ml.predict import predict
import warnings
warnings.filterwarnings('ignore')

import zipfile
import tarfile
def signup_view(request):

    if request.method == "POST":

        email = request.POST.get("email")
        password = request.POST.get("password")
        confirm_password = request.POST.get("confirm_password")

        if User.objects.filter(email_id=email).exists():
            return render(request,'predictor/signup.html',{'error':'email id already exist'}) 
        else:
            if password != confirm_password:
                return render(
                    request,
                    "predictor/signup.html",
                    {"error": "Password do not match"} )
            else:
                user = User(email_id=email, password=password)
                user.save()
                print("User created")
                print(User.objects.all().values())
                return render(request,'predictor/login.html')
        

    return render(request, "predictor/signup.html")


def login_view(request):

    if request.method == "POST":
        email = request.POST.get('email')
        password = request.POST.get('password')

        if User.objects.filter(
            email_id=email,
            password=password
        ).exists():
            return redirect('home')
        else:
            return render(
                request,
                'predictor/login.html',
                {'error': 'Invalid email or password.'}
            )

    return render(request, 'predictor/login.html')
def home(request):
    return render(request,'predictor/home.html')


def Dataset_load(request):
    if request.method=='POST':
        dataset=request.FILES.get('dataset')
        PROJECT_ROOT = Path(__file__).resolve().parent.parent
        # save dataset in raw dataset folder 
        RAW_PATH = PROJECT_ROOT / "data" / "raw"
        RAW_DATASET_PATH=RAW_PATH/'dataset'
        # Create folder if it doesn't exist
        if not RAW_PATH.exists():
            RAW_PATH.mkdir(parents=True,exist_ok=True) 
        if dataset:

            file_path=RAW_PATH/dataset.name 
            with open(file_path,'wb+') as destination:
                for chunck in dataset.chunks():
                    destination.write(chunck) 
            # extract dataset 
            #zip
            if dataset.name.endswith('.zip'):
                with zipfile.Zipfile(file_path,'r') as zip_ref:
                    zip_ref.extractall(RAW_PATH)

            # tar
            elif dataset.name.endswith('.tar'):
                    with tarfile.open(file_path,'r') as tar_ref:
                        tar_ref.extractall(RAW_PATH)

            # tgz
            elif dataset.name.endswith('.tgz'):
                    with tarfile.open(file_path,'r:gz') as tar_ref:
                        tar_ref.extractall(RAW_PATH)
            dataset_uploaded='Dataset Uploaded'
            return render(request,'predictor/home.html',{'dataset_uploaded':dataset_uploaded})

    return render(request,'predictor/home.html')
def Data_preprocessing(request):
    
    if request.method=='POST':
        preprocess_task=preprocessing_dataset()
        return render(request,'predictor/home.html',{'preprocess_task':preprocess_task})
    return render(request,'predictor/home.html') 
def Model_training(request):
    if request.method=='POST':
        train_task=model_training()
        return render(request,'predictor/home.html',{"train_task":train_task})
    return render(request,'predictor/home.html') 
from pathlib import Path

def Prediction(request):

    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    predict_images = PROJECT_ROOT / "data" / "predict_images"

    predict_images.mkdir(parents=True, exist_ok=True)

    if request.method == 'POST':

        img = request.FILES.get('image')

        if img is None:
            return render(
                request,
                'predictor/home.html',
                {'error': 'Please select an image.'}
            )

        # Save uploaded image
        img_path = predict_images / img.name

        with open(img_path, 'wb+') as destination:
            for chunk in img.chunks():
                destination.write(chunk)

        print("Image saved:", img_path)

        # Send file path to prediction function
        predicted_class, _ = predict(img_path)
        print(predicted_class)
        return render(
            request,
            'predictor/home.html',
            {'predict': predicted_class}
        )

    return render(request, 'predictor/home.html')
