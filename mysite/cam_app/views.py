# Create your views here.
from django.shortcuts import render, redirect
from django.http import HttpResponse
from django.http.response import StreamingHttpResponse
from django.views import View
from django.views.generic.edit import FormView
from os import system
from django.core.files.storage import default_storage
import uuid
from django.template import Context, Template
from django.conf import settings
from django.http import StreamingHttpResponse
from django.http import JsonResponse
from django.contrib.sessions.models import Session
from cam_app.database_operations import *



# import some common libraries
import pickle
import numpy as np
import os, json, cv2, random, glob, uuid
import matplotlib.pyplot as plt
from collections import Counter
from pathlib import Path
from django.views.decorators.clickjacking import xframe_options_sameorigin


str_uuid = uuid.uuid4()  # The UUID for image uploading

class ScannerVideoView(View):
    @xframe_options_sameorigin
    def get(self, request):
        return render(request, 'cam_app/video2.html')



class NoVideoView(View):
    def get(self, request):
        # print(request.POST)
        return render(request, 'cam_app/no_video.html')
    

def get_bbox_count(request):
    result = execute_query_and_fetch_results('data.db', 'SELECT * FROM detection_level order by date_time desc LIMIT 1;')
    
    return JsonResponse({
        'bboxCount': result[0][0],
        'crowdedness': result[0][1]
        })
