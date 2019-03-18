import argparse
import retinaNet
import detect_n_track as dt
import custom_tracker
from  TeamClassifier import TeamClassifier 

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
	help="path to trained RetinaNet model model")
ap.add_argument("-tm", "--tracker_model", required=True,
	help="path to tensorflow frozen graph of tracker (deepsort) model ")
ap.add_argument("-c", "--confidence", type=float, default=0.8,
	help="minimum probability to filter weak detections")
ap.add_argument("-v", "--video", type=str, default = None ,
	help="path to input video file")
ap.add_argument("-i", "--image", type=str, default = None ,
	help="path to image file")
ap.add_argument("-f", "--fps", type=int, default=25,
	help="Frames per sec for output video")
ap.add_argument("-s", "--skip", type=int, default=0,
	help="Frame sampling interval")
ap.add_argument("-sc", "--scale", type=float, default=1,
	help="Scale factor for image")
ap.add_argument("-age", "--max_age", type=int, default=100,
	help="Max num of frames to wait for the occluded object to return")
ap.add_argument("-n_init", "--n_init", type=int, default=1,
	help="Num consecutive frames within which the detection is done")
#ap.add_argument("-ID", "--ID_display", type=bool, default=True,
#	help="Switch the object ID display on or off")
#ap.add_argument("-teams", "--classify_teams", type=bool, default=True,
#	help="Classify teams feature true or false ")
args = vars(ap.parse_args())

net = retinaNet.retinaNet(args['model'],detection_threshold=args['confidence'])
tracker_obj = custom_tracker.DeepSort(args['tracker_model'] , max_age = args['max_age'] , max_distance = 0.5, nn_budget = None , nms_max_overlap = 1.0 , n_init = args['n_init'])


if args['video'] != None : #Run tracker on the video
    #detector.detect_video(net , args['video'],scale=args['scale'],skip=args['skip'])
    dt.detect_n_track_video(net , tracker_obj , args['video'],scale=args['scale'],skip=args['skip'] ,fps = args['fps'])
    
    
        
    
