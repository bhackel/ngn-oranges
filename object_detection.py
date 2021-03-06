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


PATH_TO_LABELS = 'models/research/object_detection/data/mscoco_label_map.pbtxt'
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)


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
  # Actual detection.
  output_dict = run_inference_for_single_image(model, image_np)

  # deletar los imagenes que no son naranjas
  del_indexes = np.argwhere(output_dict['detection_classes'] != 55)
  del_indexes = del_indexes.flatten()
  output_dict['detection_classes'] = np.delete(output_dict['detection_classes'], del_indexes)
  output_dict['detection_boxes'] = np.delete(output_dict['detection_boxes'], del_indexes, axis=0)
  output_dict['detection_scores'] = np.delete(output_dict['detection_scores'], del_indexes)

  # deletar box mas de 35% confidence
  del_indexes = np.argwhere(output_dict['detection_scores'] > 0.35)
  output_dict['detection_classes'] = np.delete(output_dict['detection_classes'], del_indexes)
  output_dict['detection_boxes'] = np.delete(output_dict['detection_boxes'], del_indexes, axis=0)
  output_dict['detection_scores'] = np.delete(output_dict['detection_scores'], del_indexes)

  # deletar cajas de carton alrededor de frame
  del_indexes = np.zeros(len(output_dict['detection_boxes']))
  if output_dict['detection_boxes'] != []:
    for i, box in enumerate(output_dict['detection_boxes']):
      ymin, xmin, ymax, xmax = box
      if (xmin < 0.05 or xmax > 0.95 or ymin < 0.05 or ymax > 0.95):
        del_indexes[i] = True

  output_dict['detection_classes'] = np.delete(output_dict['detection_classes'], del_indexes)
  output_dict['detection_boxes'] = np.delete(output_dict['detection_boxes'], del_indexes, axis=0)
  output_dict['detection_scores'] = np.delete(output_dict['detection_scores'], del_indexes)  

  output_dict['num_detections'] = len(output_dict['detection_classes'])
  if len(output_dict['detection_classes']) == 0:
    return Image.fromarray(image_np)

  # Visualization of the results of a detection.
  vis_util.visualize_boxes_and_labels_on_image_array(
      image_np,
      output_dict['detection_boxes'],
      output_dict['detection_classes'],
      output_dict['detection_scores'],
      category_index,
      instance_masks=output_dict.get('detection_masks_reframed', None),
      use_normalized_coordinates=True,
      line_thickness=2,min_score_thresh=.1)

  return Image.fromarray(image_np)


def cam():
  # get every frame
    cap = cv2.VideoCapture(1)
    while(True):
        check, img = cap.read()

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(img)

        image_pil = show_inference(detection_model, image_pil)
        size = (1368,912)
        image_pil = ImageOps.fit(image_pil, size, Image.ANTIALIAS)

        img = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)


        # Display the resulting frame
        cv2.imshow('Fruition',img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()



cam()