# import keras
import keras

# import keras_retinanet
from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color

# import miscellaneous modules
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
import time

# set tf backend to allow memory to grow, instead of claiming everything
import tensorflow as tf

def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)

# use this environment flag to change which GPU to use
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# set the modified tf session as backend in keras
keras.backend.tensorflow_backend.set_session(get_session())


class retinaNet:
    def __init__(self, model_path, detection_threshold):
        self.model_path = model_path
        self.label_names = {0: 'Biker', 1: 'Car', 2: 'Bus', 3: 'Cart', 4: 'Skater', 5: 'Pedestrian'}
        self.detection_threshold = detection_threshold

        print("[INFO] loading model...")
        # load retinanet model
        self.model = models.load_model(model_path, backbone_name='resnet50')


    def forward(self, image):

        image = preprocess_image(image)
        image, scale = resize_image(image)

        # process image
        boxes, scores, labels = self.model.predict_on_batch(np.expand_dims(image, axis=0))

        # correct for image scale
        boxes /= scale

        predicted_ann = []

        # visualize detections
        for box, score, label in zip(boxes[0], scores[0], labels[0]):
            # scores are sorted so we can break
            if score < self.detection_threshold:
                break
            record = {}
            boxCoord = []

            boxCoord.append(int(box[1]))#y1
            boxCoord.append(int(box[0]))#x1
            boxCoord.append(int(box[3]))#y2
            boxCoord.append(int(box[2]))#x2

            record['bbox'] = boxCoord
            record['label'] = self.label_names[label]

            predicted_ann.append(record)

        return predicted_ann
