from styx_msgs.msg import TrafficLight
import numpy as np
import tensorflow as tf
import time
import rospy

class TLClassifier(object):
    def __init__(self):
        self.graph = self.load_graph('light_classification/frozen_model/frozen_inference_graph.pb')
        self.threshold = .5

        self.image_tensor = self.graph.get_tensor_by_name('image_tensor:0')
        self.detect_boxes = self.graph.get_tensor_by_name('detection_boxes:0')
        self.detect_scores = self.graph.get_tensor_by_name('detection_scores:0')
        self.detect_classes = self.graph.get_tensor_by_name('detection_classes:0')
        self.num_detections = self.graph.get_tensor_by_name('num_detections:0')

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(graph=self.graph, config=config)

    def load_graph(self, graph_file):
        """Loads a frozen inference graph"""
        graph = tf.Graph()
        with graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(graph_file, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
        return graph


    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        #TODO implement light color prediction
        state = TrafficLight.UNKNOWN


        image_expanded = np.expand_dims(image, axis=0)
        with self.graph.as_default():

            #start = time.time()
            (boxes, scores, classes, num) = self.sess.run(
                [self.detect_boxes, self.detect_scores, self.detect_classes, self.num_detections],
                feed_dict={self.image_tensor: image_expanded})
            #end = time.time()
            #rospy.loginfo("inference time: %s", str((end - start) * 1000))

        classes = np.squeeze(classes).astype(np.int32)
        scores = np.squeeze(scores)

        if scores[0] > self.threshold:
            if classes[0] == 1:
                state = TrafficLight.GREEN
            elif classes[0] == 2:
                state = TrafficLight.RED
            elif classes[0] == 3:
                state = TrafficLight.YELLOW

        return state
