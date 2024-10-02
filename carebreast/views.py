import pyrebase
from django.shortcuts import render, redirect
from django.contrib import auth
from django.shortcuts import render, redirect
from django.core.files.storage import FileSystemStorage
from .prediction.predict import predict_cancer  # Adjust this import based on your file structure

from .forms import ImageUploadForm
import os

import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend
import matplotlib.pyplot as plt

from django.shortcuts import render
from .forms import ImageUploadForm
from .prediction.predict import predict_cancer
from django.conf import settings

def upload_image(request):
    form = ImageUploadForm()  # Initialize the form for the case of invalid submission

    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            # Save the uploaded image
            uploaded_image = request.FILES['image']
            fs = FileSystemStorage()  # Use FileSystemStorage for saving files
            image_path = fs.save(uploaded_image.name, uploaded_image)  # Save the uploaded image
            image_url = fs.url(image_path)  # Get the URL of the uploaded image

            # Call your prediction function
            prediction, heatmap_path = predict_cancer(os.path.join(settings.MEDIA_ROOT, image_path), visualize=True)

            # Pass the result and paths to the template
            context = {
                'prediction': prediction,
                'image_url': image_url,
                'heatmap_url': heatmap_path,
            }
            return render(request, 'result.html', context)

    return render(request, 'welcome.html', {'form': form})

# Firebase config
config = {
    'apiKey': "AIzaSyAM7b26IhvB75SvpN-bHaKHMtAArWJimuw",
    'authDomain': "breastcancerinsight.firebaseapp.com",
    'databaseURL': "https://breastcancerinsight-default-rtdb.firebaseio.com/",
    'projectId': "breastcancerinsight",
    'storageBucket': "breastcancerinsight.appspot.com",
    'messagingSenderId': "338793847759",
    'appId': "1:338793847759:web:d870384012e935507790ba",
    'measurementId': "G-QPGDD2LQTY"
}

firebase = pyrebase.initialize_app(config)
authe = firebase.auth()
database = firebase.database()


def welcome(request):
  # Check if the session contains a user token
  is_authenticated = 'uid' in request.session
  return render(request, "welcome.html", {"is_authenticated": is_authenticated})


# Sign-in page
def signin(request):
  return render(request, "signin.html")

def faq(request):
  return render(request,"faq.html")

# Post sign-in logic
def postsign(request):
  email = request.POST.get('email')
  password = request.POST.get('password')
  try:
    user = authe.sign_in_with_email_and_password(email, password)
    uid = user['localId']  # Get UID for data operations

  except:
    message = "Invalid Credentials!"
    return render(request, "signin.html", {"mess": message})

  session_id = user['idToken']
  request.session['uid'] = uid
  return redirect('welcome')


# Logout logic
def logout(request):
  # Clear the session data
  request.session.flush()
  return redirect('welcome')


# Sign-up page
def signup(request):
  return render(request, "signup.html")


# Post sign-up logic
def postsignup(request):
  name = request.POST.get('name')
  email = request.POST.get('email')
  password = request.POST.get('password')
  try:
    user = authe.create_user_with_email_and_password(email, password)
    uid = user['localId']

  except:
    message = "Unable To Create Account as Password is too weak!"
    return render(request, "signup.html", {"mess": message})

  uid = user['localId']
  data = {"name": name, "email": email, "password": password, "status": "1"}
  database.child("users").child(uid).child("details").set(data)

  session_id = user['idToken']
  request.session['uid'] = uid
  return redirect('welcome')

def account_details(request):
    # Check if the session contains a user token
    if 'uid' not in request.session:
        return redirect('signin')
    
    uid = request.session['uid']
    print(f"User ID from session: {uid}")
    
    try:
        user_details = database.child("users").child(uid).child("details").get().val()
        print(f"User details fetched: {user_details}")
    except Exception as e:
        print(f"Error fetching user details: {e}")
        user_details = None

    if user_details is None:
        return render(request, "account_details.html", {"message": "No user details found"})
    
    return render(request, "account_details.html", {"user_details": user_details})
