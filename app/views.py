from django.shortcuts import render,redirect
from django.conf import settings
from django.core.files.storage import FileSystemStorage
from .models import image
from .forms import ImageForm
import cv2
import numpy as np
import pickle
from pyspark import SparkContext, SparkFiles



def home(request):
    documents = image.objects.all()
    return render(request, 'home.html', { 'documents': documents })


def upload(request):
    if request.method == 'POST' and request.FILES['file']:
        file = request.FILES['file']
        fs = FileSystemStorage()
        filename = fs.save(file.name, file)
        uploaded_file_url = fs.url(filename)
        return render(request, 'upload.html', {
            'uploaded_file_url': uploaded_file_url
        })
    return render(request, 'upload.html')


def model_form_upload(request):
    if request.method == 'POST':
        form = ImageForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()
            return redirect('home')
    else:
        form = ImageForm()
    return render(request, 'model_form_upload.html', {
        'form': form
})

img_dir = './upload/'
result = './result/'

sc = SparkContext()
classifier = "./haarcascade_frontalface_default.xml"
sc.addFile(classifier)
images_RDD = sc.binaryFiles(img_dir)




num_face_rdd = images_RDD.map()
result = num_face_rdd.collect()
pickle.dump(result,open("./face_detection_result.p","wb"))



