import argparse
import fasterRCNN
import crops_generator as cg
import custom_tracker
from  TeamClassifier import TeamClassifier 


def str2bool(v):
    print(v)
    if v.lower() =='true':
        return True
    elif v.lower() == 'false':
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-l", "--labelmap", required=True,
	help="labels txt")
ap.add_argument("-m", "--model", required=True,
	help="path to tensorflow frozen graph of object detection model")
ap.add_argument("-tm", "--tracker_model", required=True,
	help="path to tensorflow frozen graph of tracker (deepsort) model ")
ap.add_argument("-c", "--confidence", type=float, default=0.8,
	help="minimum probability to filter weak detections")
ap.add_argument("-f", "--fps", type=int, default=25,
	help="Frames per sec for output video")
ap.add_argument("-s", "--skip", type=int, default=1,
	help="Frame sampling interval")
ap.add_argument("-sc", "--scale", type=float, default=1,
	help="Scale factor for image")
ap.add_argument("-age", "--max_age", type=int, default=100,
	help="Max num of frames to wait for the occluded object to return")
ap.add_argument("-n_init", "--n_init", type=int, default=10,
	help="Num consecutive frames within which the detection is done")
ap.add_argument("-ID", "--ID_display", type=str2bool, default=True,
	help="Switch the object ID display on or off")
ap.add_argument("-teams", "--classify_teams", type=str2bool, default=False,
	help="Classify teams feature true or false ")
ap.add_argument("-reuse", "--occluded_feature_reuse", type=str2bool, default=False,
	help="reusing the occluded features")
ap.add_argument("-crp", "--dump_crops", type=str2bool, default=True,
	help="Dump crops based on IDS")
args = vars(ap.parse_args())

### USER INPUT :######################################################################################################################################
videos = ['/home/ab/projects/amit-priya-shared/Main_repos/deepsort_faster_rcnn/test_video_n_out_crops/092_sub4_video.avi',
          '/home/ab/projects/amit-priya-shared/Main_repos/deepsort_faster_rcnn/test_video_n_out_crops/Clip19_sub1_out_video.avi'
         ] 
############################################################################################################################################################

for v in videos :
    net = fasterRCNN.fasterRCNN(args['model'],args['labelmap'],detection_threshold=args['confidence'])
    tracker_obj = custom_tracker.DeepSort(args['tracker_model'] , max_age = args['max_age'] , max_distance = 0.5, nn_budget = 150 , nms_max_overlap = 1.0 , n_init = args['n_init'])
    team_classifier_obj = TeamClassifier(len_kmeans_counter = 100)    
    cg.detect_n_track_video(net , tracker_obj , team_classifier_obj , v ,scale=args['scale'],skip=args['skip'] , team_classify = args['classify_teams'] , fps = args['fps']
                                ,  reuse_occluded_features = args['occluded_feature_reuse'] , dump_crops = args['dump_crops'])                                


    
    
        
    
