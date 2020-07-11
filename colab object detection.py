import numpy as np
import os
import six.moves.urllib as urllib
import sys
import zipfile
import pathlib
import cv2



from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

from collections import defaultdict
from matplotlib import pyplot as plt
from PIL import Image, ImageOps

from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

import tensorflow as tf
# patch tf1 into `utils.ops`
utils_ops.tf = tf.compat.v1
# Patch the location of gfile
tf.gfile = tf.io.gfile


config = ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.2
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

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

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = 'models/research/object_detection/data/mscoco_label_map.pbtxt'
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

# If you want to test the code with your images, just add path to the images to the TEST_IMAGE_PATHS.
PATH_TO_TEST_IMAGES_DIR = pathlib.Path('./images/')
TEST_IMAGE_PATHS = sorted(list(PATH_TO_TEST_IMAGES_DIR.glob("*.jpg")))

model_name = 'ssd_mobilenet_v1_coco_2017_11_17'
detection_model = load_model(model_name)

def run_inference_for_single_image(model, image):
    image = np.asarray(image)
    # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
    input_tensor = tf.convert_to_tensor(image)
    # The model expects a batch of images, so add an axis with `tf.newaxis`.
    input_tensor = input_tensor[tf.newaxis,...]

    # Run inference
    output_dict = model(input_tensor)

    # All outputs are batches tensors.
    # Convert to numpy arrays, and take index [0] to remove the batch dimension.
    # We're only interested in the first num_detections.
    num_detections = int(output_dict.pop('num_detections'))
    output_dict = {key:value[0, :num_detections].numpy() 
                 for key,value in output_dict.items()}
    output_dict['num_detections'] = num_detections

    # detection_classes should be ints.
    output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)
    
    return output_dict

def show_inference(model, image):
    # the array based representation of the image will be used later in order to prepare the
    # result image with boxes and labels on it.
    image_np = np.array(image)
    # Detect object
    output_dict = run_inference_for_single_image(model, image_np)
  
    # Get the size of the axis of the image
    y_size = np.size(image_np, axis=0)
    x_size = np.size(image_np, axis=1)

    # find the first instance of an orange
    found = False
    i = 0
    for num in output_dict['detection_classes']:
    	if num == 55:
    		found = True
    		break
    	i += 1
    # return the original image if no orange is found
    if found == False:
    	image_pil = Image.fromarray(image_np)
    	return image_pil

    # un-normalize dimensions of bounding box
    box = output_dict['detection_boxes'][i]
    ymin = int(np.round(box[0] * y_size))
    xmin = int(np.round(box[1] * x_size))
    ymax = int(np.round(box[2] * y_size))
    xmax = int(np.round(box[3] * x_size))

    # crop the image
    print(ymin, xmin, ymax, xmax)
    image_np = image_np[ymin:ymax, xmin:xmax]

    # resize the image to 256x256
    image_pil = Image.fromarray(image_np)
    size = (256, 256)
    image_pil = ImageOps.fit(image_pil, size, Image.ANTIALIAS)
    
    # return image
    return image_pil

def single():
	image = Image.open("./images/oranges1.jpg")
	image = show_inference(detection_model, image)
	image.show()

def cam():
	try:
		# get every frame
	    cap = cv2.VideoCapture(0)
	    while(True):
	        check, img = cap.read()

	        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	        image_pil = Image.fromarray(img)

	        image_pil = show_inference(detection_model, image_pil)

	        img = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR) 
	        # Display the resulting frame
	        cv2.imshow('ORANGESYEEY',img)
	        if cv2.waitKey(1) & 0xFF == ord('q'):
	            break

	    # When everything done, release the capture
	    cap.release()
	    cv2.destroyAllWindows()
	except:
	    print("no cam")

single()