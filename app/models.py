from django.db import models

# Create your models here.


class image(models.Model):
    img=models.FileField(upload_to='upload/')