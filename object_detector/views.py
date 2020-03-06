from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
import numpy as np
import urllib
import json
import os
import sys
from six import BytesIO
import tarfile
import tensorflow as tf
import zipfile
import pathlib
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image

from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

utils_ops.tf = tf.compat.v1

tf.gfile = tf.io.gfile

PATH_TO_LABELS = 'models/research/object_detection/data/mscoco_label_map.pbtxt'
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

PATH_TO_LABELS_oi = 'models/research/object_detection/data/oid_v4_label_map.pbtxt'
category_index_oi = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS_oi, use_display_name=True)



@csrf_exempt
def o_detect(request):
   
	data = {}
	
	if request.method == "POST":
		
		if request.FILES.get("image", None) is not None:
			
			image = _grab_image(stream=request.FILES["image"])

		
		else:
			
			url = request.POST.get("url", None)

			
			if url is None:
				data["error"] = "No URL provided."
				return JsonResponse(data)

			
			image = _grab_image(url=url)

		input_tensor = tf.convert_to_tensor(image)
		input_tensor = input_tensor[tf.newaxis,...]

		model_name = 'ssd_mobilenet_v1_coco_2017_11_17'
		detection_model = load_model(model_name)

		output_dict = detection_model(input_tensor)
		detection_labels1 = output_dict["detection_classes"].numpy()
		label_scores1=output_dict["detection_scores"].numpy()
		detection_labels=np.copy(detection_labels1[0])
		label_scores=np.copy(label_scores1[0])
		#d_label_names=np.empty((100,),dtype=str)
		count=0
		final_list=np.zeros(91)
		for i in label_scores:
			if i>=0.35:
				final_list[int(detection_labels[count])]=final_list[int(detection_labels[count])]+1
			count=count+1
		final_ans={}
		count=0  		
		for i in final_list:
			if i>0:
				a=category_index[count]['name']
				final_ans[a]=int(i)
			count=count+1
  		
    		
      			
      			
      			

		
	# return a JSON response
	return JsonResponse(final_ans)


@csrf_exempt
def o_detect_oi(request):
    
	data = {}
	
	if request.method == "POST":
		
		if request.FILES.get("image", None) is not None:
			
			image = _grab_image(stream=request.FILES["image"])

		
		else:
			
			url = request.POST.get("url", None)

			
			if url is None:
				data["error"] = "No URL provided."
				return JsonResponse(data)

			
			image = _grab_image(url=url)

		input_tensor = tf.convert_to_tensor(image)
		input_tensor = input_tensor[tf.newaxis,...]

		model_name = 'faster_rcnn_inception_resnet_v2_atrous_oid_v4_2018_12_12'
		detection_model = load_model(model_name)

		output_dict = detection_model(input_tensor)
		detection_labels1 = output_dict["detection_classes"].numpy()
		label_scores1=output_dict["detection_scores"].numpy()
		detection_labels=np.copy(detection_labels1[0])
		label_scores=np.copy(label_scores1[0])
		#d_label_names=np.empty((100,),dtype=str)
		count=0
		final_list=np.zeros(602)
		for i in label_scores:
			if i>=0.35:
				final_list[int(detection_labels[count])]=final_list[int(detection_labels[count])]+1
			count=count+1
		final_ans={}
		count=0  		
		for i in final_list:
			if i>0:
				a=category_index_oi[count]['name']
				final_ans[a]=int(i)
			count=count+1
  		
    		
      			
      			
      			

		
	# return a JSON response
	return JsonResponse(final_ans)

def _grab_image(stream=None, url=None):
	if url is not None:
			resp = urllib.request.urlopen(url)
			data = np.array(Image.open(BytesIO(resp.read())))
	elif stream is not None:
			data = np.array(Image.open(stream))
	
	image = np.asarray(data)
		
 
	# return the image
	return image

def load_model(model_name):
  base_url = 'http://download.tensorflow.org/models/object_detection/'
  model_file = model_name + '.tar.gz'
  model_dir = tf.keras.utils.get_file(
    fname=model_name, 
    origin=base_url + model_file,
    untar=True)

  model_dir = pathlib.Path(model_dir)/"saved_model"

  model = tf.saved_model.load(str(model_dir))
  model = model.signatures['serving_default']

  return model