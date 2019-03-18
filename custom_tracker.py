import cv2 
import numpy as np
from deep_sort.deep_sort import nn_matching
from deep_sort.deep_sort.detection import Detection
from deep_sort.deep_sort.tracker import Tracker
from deep_sort.tools import generate_detections as gdet


class DeepSort():
    def __init__(self, model_folder , max_age , max_distance = 0.1 , nn_budget = None , nms_max_overlap = 1.0 , n_init=3):
       # Definition of the parameters
        self.max_distance = max_distance
        self.nn_budget = nn_budget
        self.nms_max_overlap = nms_max_overlap
        
       # deep_sort
        self.encoder = gdet.create_box_encoder(model_folder,batch_size=1)        
        #self.metric = nn_matching.NearestNeighborDistanceMetric("cosine", 0.5 , nn_budget) #max_cosine_distance
        self.metric = nn_matching.NearestNeighborDistanceMetric("euclidean", max_distance, nn_budget)
        self.tracker = Tracker(self.metric , max_age = max_age , n_init=n_init)