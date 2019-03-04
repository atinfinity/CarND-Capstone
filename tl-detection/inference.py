import os
import sys
from glob import glob
from PIL import Image
import numpy as np
import tensorflow as tf

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from collections import defaultdict
from utils import label_map_util
from utils import visualization_utils as vis_util
import time

def load_graph(graph_file):
    """Loads a frozen inference graph"""
    graph = tf.Graph()
    with graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(graph_file, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    return graph

def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)


PATH_TO_GRAPH = r'frozen_model/frozen_inference_graph.pb'
PATH_TO_LABELS = r'data/udacity_label_map.pbtxt'
NUM_CLASSES = 4

detection_graph = load_graph(PATH_TO_GRAPH)

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)
print(category_index)

IMAGE_SIZE = (12, 8)

TEST_IMAGES_DIR = 'testimage'
EXTENSION = '*.*'
print(os.path.join(TEST_IMAGES_DIR, EXTENSION))
TEST_IMAGE_PATHS = glob(os.path.join(TEST_IMAGES_DIR, EXTENSION))
print("number of test images:", len(TEST_IMAGE_PATHS))

with detection_graph.as_default():
    with tf.Session(graph=detection_graph) as sess:
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        detect_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
        detect_scores = detection_graph.get_tensor_by_name('detection_scores:0')
        detect_classes = detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = detection_graph.get_tensor_by_name('num_detections:0')
        
        for img_path in TEST_IMAGE_PATHS:
            image = Image.open(img_path)
            image_np = load_image_into_numpy_array(image)
            image_expanded = np.expand_dims(image_np, axis=0)
            
            start = time.time()

            (boxes, scores, classes, num) = sess.run(
                [detect_boxes, detect_scores, detect_classes, num_detections],
                feed_dict={image_tensor: image_expanded})

            end = time.time()
            print("inference time = ", (end - start) * 1000) 
            
            vis_util.visualize_boxes_and_labels_on_image_array(
                image_np, 
                np.squeeze(boxes),
                np.squeeze(classes).astype(np.int32),
                np.squeeze(scores),
                category_index,
                use_normalized_coordinates=True,
                max_boxes_to_draw=5,
                line_thickness=8)
            plt.figure(figsize=IMAGE_SIZE)
            plt.imshow(image_np)

            basename = os.path.basename(img_path)
            plt.savefig('result/' + basename)
