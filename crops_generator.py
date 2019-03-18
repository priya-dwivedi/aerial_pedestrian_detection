import cv2 
import numpy as np
from deep_sort.deep_sort.detection import Detection 
from deep_sort.application_util import preprocessing
from scipy.spatial.distance import cdist

def image_fprop(net , image):
    if type(image) == str :
        image = cv2.imread(image)
    detections = net.forward(image)
    return detections

def overlay_image(image , box):
    (startY, startX, endY, endX) = box.astype("int")
    cv2.rectangle(image, (startX, startY), (endX, endY),(0,255,0), 2) 

def get_xywh(boxes):
    ret_boxes = []
    for box in boxes :
        box = box.astype("int")
        x = int(box[0])  
        y = int(box[1])  
        w = int(box[2]-box[0])
        h = int(box[3]-box[1])
    
        if x < 0 :
            w = w + x
            x = 0
        if y < 0 :
            h = h + y
            y = 0 
        ret_boxes.append([x,y,w,h])
    return ret_boxes

def get_euclidean_dist(c1,c2):
    dist = np.sqrt(sum((c1-c2)**2))
    return dist 
    
def get_centroid(box):
    centroid = np.array([box[0]+box[2] , box[1]+box[3]])/2.0  
    return centroid
    
def mapIds_previous(prev_centroids , prev_ids , current_centroids):
    mapped_ids_list = []
    
    for i , c1 in enumerate(current_centroids):
        #print(i,'i')
        dist_list = []
        for j , c2 in enumerate(prev_centroids):
            dist_list.append(get_euclidean_dist(c1,c2))
        if min(dist_list) < 10 :
            ind = np.where(dist_list==min(dist_list))[0][0]
            mapped_ids_list.append(prev_ids[ind])
        else :
            print(c1,c2)            
            mapped_ids_list.append(None)
       
    return mapped_ids_list  

def getId_features_map(det_centroids , det_features , trackd_bboxes , trackd_IDs , min_dist = 10 ):
    assert(len(trackd_bboxes) == len(trackd_IDs))
    Id_features_map = {}
    print(det_centroids)
    for i in range(len(trackd_IDs)):      
        src = get_centroid(trackd_bboxes[i])
        dst = det_centroids
        dist = cdist([src], dst, metric='euclidean')
        dist = np.ravel(dist)
        if min(dist) <= min_dist :
            ind = np.where(dist == min(dist))[0][0]
            print("src : ",src)
            print("ind : ",ind)            
            Id_features_map[trackd_IDs[i]] = det_features[ind]            
    return Id_features_map        
    
def getId_props(prev_ids , current_ids , prev_centroids , current_centroids , imsize , boundary_offset = 10 ):
    
    missing_ids = set(prev_ids) - set(current_ids)
    new_ids = set(current_ids) - set(prev_ids)
    
    all_ids = list(set(list(prev_ids) + list(current_ids)))
    
    tags = ['unoccluded']*len(all_ids)    
    
    prev_ids = np.array(prev_ids)
    prev_centroids = np.array(prev_centroids)
    
    for ind , i in enumerate(all_ids):
        if i in missing_ids :        
            p_ind = np.where(prev_ids == i )[0]

            x = prev_centroids[p_ind][0][0]
            y = prev_centroids[p_ind][0][1]
    
            if (x >= imsize[0]-boundary_offset) or (x <= boundary_offset) \
                        or (y >= imsize[1]-boundary_offset) or (y <= boundary_offset): #Near to horizontal or vertical boundary 
                tags[ind] = 'out_of_frame'
            else :
                tags[ind] = 'occluded'
            
        elif i in new_ids :
            tags[ind] = 'new'
    
    occlsn_dict = dict(zip(all_ids,tags))
    #if 'occluded' in tags :
    #    import pdb ; pdb.set_trace()            
    return occlsn_dict

def convert2tf_format(boxes):
    """
    Convert boxes from deepsort format (x1,y1,x2,y2) to tf detections format (y1,x1,y2,x2)
    """
    tf_boxes = []
    for box in boxes :
        tf_box = [box[1],box[0],box[3],box[2]]
        tf_boxes.append(tf_box)
    return tf_boxes

    
def pretty_print_dicts(dict1,dict2):
    keys = set(list(dict1.keys()) + list(dict2.keys()))
    print("KEY , OLD DICT       ,         NEW_DICT     ,     DIFF")
    
    for key in keys : 
        diff = 0 
        key1_flag = True
        dict_str = str(key) + ' : '          
        if key in dict1.keys():
            dict_str = dict_str + str(dict1[key]) + '   ' 
            diff = dict1[key]-diff
        else :
            key1_flag = False
            dict_str = dict_str + '----------------   '
            
        if key in dict2.keys():
            key2_flag = True
            dict_str = dict_str + str(dict2[key]) + '   ' 
            diff = dict2[key]-diff
        else :
            key2_flag = False
            dict_str = dict_str + '----------------   '
            
        dict_str = dict_str + ' ====  ' + str(diff)
        print(dict_str+'\n')
        #if max(np.abs(diff)) > 500 and key1_flag == True and key2_flag == True:
            #import pdb ; pdb.set_trace()

def copy_dict(src, dest):
    dest = {}
    for key in src.keys():
        dest[key] = src[key] 
    return dest
    
def check_mismatch(list1,list2):
    assert(len(list1) == len(list2))
    for i in range(len(list1)):
        if (list2[i] != None) and  (list1[i] != None):
            if (list2[i] - list1[i]) != 0:
                return True
    return False
    
def check_vanished(list1,list2):
    print(list1 , list2)
#    import pdb ; pdb.set_trace()
    occlusn_list = []
    for i in range(len(list1)):
        print(list1 , list2)
        if (list2[i] != None) and  (list1[i] != None):
            if (list2[i] - list1[i]) != 0:
                occlusn_list.append('occluded')
                
            else :
                occlusn_list.append('')
    return occlusn_list
    
def filter_repeated_boxes(boxes , thresh = 10):
    centroids = []    
    for box in boxes :
        centroid = get_centroid(box)
        centroids.append(centroid)
    
    #dist =cdist(centroids, centroids, metric='euclidean')
    unique_inds = []
    removed_inds = []
    for i in range(len(centroids)):
        print(removed_inds)
        if i not in removed_inds:
            src = centroids[i]
            dist =cdist([src], centroids, metric='euclidean')
            dist = np.ravel(dist)
            selected_ind = np.where(dist<=thresh)[0]
            if len(selected_ind) > 0 :
                #import pdb;pdb.set_trace()
                unique_inds.append(selected_ind[0])
                if len(selected_ind) > 1 :
                    removed_inds.extend(selected_ind[1:])
                    #import pdb;pdb.set_trace()
    
    return unique_inds

def mkdir(dirpath):
    import os
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)

def get_crop(image , bbox , x_offset = 15 , y_offset = 15):
    #import pdb ; pdb.set_trace()
    bbox = bbox.astype('int') 
    y1 = max(0,bbox[1]-y_offset) 
    y2 = min(bbox[3]+y_offset , image.shape[0])
    x1 = max(0,bbox[0]-x_offset) 
    x2 = min(bbox[2]+x_offset , image.shape[1])
    crop = image[y1:y2 , x1:x2]
    return crop

def write_crops(out_folder , frame_num , image , ID , bbox , y_offset = 15 , x_offset = 15):
    crop = get_crop(image , bbox , x_offset , y_offset)
    out_path = out_folder+'/'+str(ID) 
    #import pdb ; pdb.set_trace()
    out_name = out_path + '/' + str(ID) + '_' + str(frame_num) + '.jpg'
    mkdir(out_path)
    cv2.imwrite(out_name,crop)
            
    
#def check_occlusion():
    
def detect_n_track_video(detect_net ,  tracker_obj , team_classifier_obj, vidname , detection_threshold = 0.8 , 
                             scale = 1.0 , skip = 0 , team_classify = False , fps = 24 , reuse_occluded_features = False , dump_crops = False):
    frame_num = 0
    cap = cv2.VideoCapture(vidname)
    W = int(cap.get(3))
    H = int(cap.get(4)) 
    video_size = [int(W),int(H)]
    length = int(cap.get(7))
    vid_fps = fps or int(round(cap.get(5)))        
    print(video_size)
    print("Video Size :",video_size)
    print("FPS : ",vid_fps)
    tracker = tracker_obj.tracker 
    out_folder = vidname.split('.')[0]
    out_filename = out_folder.split('/')[-1] + '_out.avi'
    #out = cv2.VideoWriter(out_filename,cv2.VideoWriter_fourcc('X','2', '6', '4'), vid_fps, (int(W*scale),int(H*scale)))
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(out_filename,fourcc, vid_fps, (int(W*scale),int(H*scale)))
    prev_ID_centroid_dict = {}
    
    overall_centroid_dict = {}
    overall_ID_bbox_dict = {}
    overall_ID_feature_dict = {}
    
    occluded_boxes_retained = []
    occluded_features_retained = []  
        
    prev_scaled_boxes = []
    prev_boxes = []
    prev_boxes1 = []
    prev_boxes2 = []
    
    while True:
        if frame_num % (skip+1)  == 0 : 
            ret, image = cap.read()
            if not ret:break        
                
            
            #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
            sc_width,sc_height = [int(W*scale) , int(H*scale)]
            if scale != 1:
                print(sc_width,sc_height)
                image = cv2.resize(image, (sc_width,sc_height), interpolation=cv2.INTER_AREA)
            #image[0:260,:] = 0
            detections = image_fprop(detect_net,image)
            num_detections = int(detections['num_detections'])
            print("Num detections:" , detections['num_detections'])
            
            tf_boxes = detections['detection_boxes'][0:num_detections] * np.array([sc_height, sc_width, sc_height, sc_width])
            
            unique_indices = filter_repeated_boxes(tf_boxes , thresh = 20)
            
            #if len(unique_indices) < len(tf_boxes):
                #import pdb ; pdb.set_trace()
            #orig_boxes = tf_boxes[unique_indices][:,[1,0,3,2]]
            orig_boxes = tf_boxes[:,[1,0,3,2]] #[tf_boxes[:,1],tf_boxes[:,0],tf_boxes[:,3],tf_boxes[:,2]]
            deepsort_boxes = get_xywh(orig_boxes) 
            
            features = tracker_obj.encoder(image,deepsort_boxes)
            if len(occluded_boxes_retained) > 0 and reuse_occluded_features == True :
                assert(1==2)
                orig_boxes = list(orig_boxes)+occluded_boxes_retained
                deepsort_boxes = list(deepsort_boxes)+get_xywh(occluded_boxes_retained)
                features = list(features) + occluded_features_retained
                #import pdb ; pdb.set_trace()
            
            
            prev_scaled_boxes.append(deepsort_boxes)
            prev_boxes.append(orig_boxes)
            # score to 1.0 here).
            tr_detections = [Detection(bbox, 1.0, feature) for bbox, feature in zip(deepsort_boxes, features)]  
            
            # Run non-maxima suppression.
            boxes = np.array([d.tlwh for d in tr_detections])
            prev_boxes2.append(boxes)
            scores = np.array([d.confidence for d in tr_detections])
            indices = preprocessing.non_max_suppression(boxes, tracker_obj.nms_max_overlap, scores)
            tr_detections = [tr_detections[i] for i in indices]
                        
           
            print("++++++++++++++++++++++++++++++++++",orig_boxes)
            det_bboxes = []
            det_features = []
            det_centroids = []
            for det in tr_detections:
                bbox = det.to_tlbr()
                det_bboxes.append(bbox)
                det_centroids.append(get_centroid(bbox))
                det_features.append(det.feature)
                print("@@@@@@@@",bbox , det.feature.max() , det.feature.min())
                prev_boxes1.append(bbox)
                
                if team_classify == True  :
                    assert(1==2)
                    #import pdb ; pdb.set_trace()                    
                    if team_classifier_obj.trained == False :
                        team_classifier_obj.scheduleKmeansTrain(image , bbox)  
                    else : 
                        team_id = team_classifier_obj.predict(image , bbox) 
                        print("============== TEAM ID",team_id)
                        #import pdb ; pdb.set_trace() 
                        

                        
            #import pdb ; pdb.set_trace() 
            #prev_tr_detections = tr_detections                       
                        
            # Call the tracker
            tracker.predict()
            tracker.update(tr_detections) 
            
            ID_centroid_dict = {}
            prev_ID_centroid_dict = copy_dict(overall_centroid_dict , prev_ID_centroid_dict) #ID_centroid_dict
            
#            current_centroids = []
#            current_ids = [] 
            
            trackd_bboxes = []
            trackd_centroids = []
            trackd_IDs = []
            image_orig = np.copy(image)
            for track in tracker.tracks:
                if not track.is_confirmed() or track.time_since_update > 1 :   #or track.track_id  != 3:
                    continue

                trackd_bbox = track.to_tlbr()
                trackd_ID = track.track_id
                trackd_centroid = get_centroid(bbox).astype('int')
                
                trackd_bboxes.append(trackd_bbox)
                trackd_IDs.append(trackd_ID)
                trackd_centroids.append(trackd_centroid)
                ID_centroid_dict[trackd_ID] = trackd_centroid                
                ID_features_map = getId_features_map(det_centroids , det_features , [trackd_bbox] , [trackd_ID])
                #import pdb ; pdb.set_trace()
                overall_centroid_dict[trackd_ID] = trackd_centroid 
                overall_ID_bbox_dict[trackd_ID] = trackd_bbox 
                if len(ID_features_map) > 0 :
                    overall_ID_feature_dict[trackd_ID] = ID_features_map[trackd_ID]
                #import pdb ; pdb.set_trace()         
                print("track_id = " , trackd_ID , trackd_bbox , len(track.features))
                
                write_crops(out_folder , frame_num , image_orig , trackd_ID , trackd_bbox , x_offset = 40 , y_offset = 50 )
                
                        
                cv2.rectangle(image, (int(trackd_bbox[0]), int(trackd_bbox[1])), (int(trackd_bbox[2]), int(trackd_bbox[3])),(255,255,255), 2)
                cv2.putText(image, str(trackd_ID),(int(trackd_bbox[0]), int(trackd_bbox[1])),0, 5e-3 * 200, (0,255,0),2) 
                
                #cv2.putText(image, str(ID_centroid_dict[trackd_ID]),(int(trackd_centroid[0]), int(trackd_centroid[1])),0, 5e-3 * 100, (0,255,0),2)
                print(frame_num , track.track_id)
                
            print("==============================",frame_num)
            
            
            if ( frame_num > 1 )  and ( len(trackd_IDs) > 0 ):    
                #ID_features_map  = getId_features_map(det_centroids , det_features , trackd_bboxes , trackd_IDs)               
#                prev_mapped_ids = mapIds_previous(prev_centroids , prev_ids , current_centroids )
                ID_props_dict =  getId_props(list(overall_ID_bbox_dict.keys()) , trackd_IDs , list(overall_centroid_dict.values()) , trackd_centroids , imsize=[sc_width ,sc_height], 
                                                      boundary_offset = 10) 
                                                      
                #import pdb; pdb.set_trace()
                occluded_boxes_retained = []
                occluded_features_retained = []                                                                       
                for ID in ID_props_dict.keys():
                    if ID_props_dict[ID] == 'occluded' : 
                        print("occluded ID : " , ID)
                        if ID in overall_ID_feature_dict.keys():
                            occluded_features_retained.append(overall_ID_feature_dict[ID])
                            occluded_boxes_retained.append(overall_ID_bbox_dict[ID]) 
                        #import pdb; pdb.set_trace()
                    if ID_props_dict[ID] == 'out_of_frame':
                        print('out_of_frame')
                #import pdb; pdb.set_trace()
                                               
            #prev_ids = current_ids
            #prev_centroids = current_centroids
            
            pretty_print_dicts(prev_ID_centroid_dict,ID_centroid_dict)            
#            prev_ID_centroid_dict = overall_centroid_dict #ID_centroid_dict

            
            cv2.imshow("Frame", image)
            out.write(image)
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord("q"):# if the `q` key was pressed, break from the loop
                break
                cap.release()
                out.release() 
                print("-------------- Frame:",frame_num)

            frame_num += 1
    
    print("-------------- Frame:",frame_num)
    print(overall_centroid_dict)
    cap.release()  
    out.release()    
    cv2.destroyAllWindows()            