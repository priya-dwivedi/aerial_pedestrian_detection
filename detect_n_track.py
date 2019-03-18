import cv2 
import numpy as np
from deep_sort.deep_sort.detection import Detection 
from deep_sort.application_util import preprocessing

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
        # box = box.astype("int")
        x = int(box[1])  
        y = int(box[0])  
        w = int(box[3]-box[1])
        h = int(box[2]-box[0])
    
        if x < 0 :
            w = w + x
            x = 0
        if y < 0 :
            h = h + y
            y = 0 
        ret_boxes.append([x,y,w,h])
    return ret_boxes

def writeLog(file_handle , frame_number, ID , bbox , imsize = None ):
    if imsize !=None:
        ymin = max(0,bbox[1])    
        ymax = min(imsize[1],bbox[3])
        xmin = max(0,bbox[0])
        xmax = min(imsize[0],bbox[2])
    else :
        ymin = bbox[1]   
        ymax = bbox[3]
        xmin = bbox[0]
        xmax = bbox[2]       
    #print(xmin,ymin ,xmax , ymax)
    file_handle.write(str(frame_number) +  ',' +
                      str(ID) +  ',' +
                      str(xmin) +  ',' +
                      str(ymin) +  ',' +
                      str(xmax) +  ',' +
                      str(ymax) +'\n')
    
 
def detect_n_track_video(detect_net ,  tracker_obj , vidname , detection_threshold = 0.8 , scale = 1.0 , skip = 0 , fps = None):
    frame_num = 0
    cap = cv2.VideoCapture(vidname)
    W = int(cap.get(3))
    H = int(cap.get(4)) 
    video_size = [int(W),int(H)]
    length = int(cap.get(7))
    vid_fps = int(round(cap.get(5)))
    vid_fps = fps or vid_fps        
    print(video_size)
    print("Video Size :",video_size)
    print("FPS : ",vid_fps)
    tracker = tracker_obj.tracker 
    out_filename = vidname.split('.')[0].split('/')[-1] + '_out.avi'
    out_logname = vidname.split('.')[0].split('/')[-1] + '_out.csv'
    
    
    out = cv2.VideoWriter(out_filename,cv2.VideoWriter_fourcc('X','2', '6', '4'), vid_fps, (int(W*scale),int(H*scale)))
    out_log = open(out_logname , 'w')
    writeLog(out_log , 'frame_number' , 'ID' , ['xmin','ymin','xmax','ymax'] )
    
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

            ## convert to RGB space
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            print("Image shape: ", image.shape)
            # print(image)

            detections = image_fprop(detect_net,image)
            num_detections = int(len(detections))
            print("Num detections:" , num_detections)

            boxes = []
            for det in detections:
                boxes.append(det['bbox'])

            scaled_boxes = get_xywh(boxes) 
            features = tracker_obj.encoder(image,scaled_boxes)
            # score to 1.0 here).
            tr_detections = [Detection(bbox, 1.0, feature) for bbox, feature in zip(scaled_boxes, features)]  
            
            # Run non-maxima suppression.
            boxes = np.array([d.tlwh for d in tr_detections])
            scores = np.array([d.confidence for d in tr_detections])
            indices = preprocessing.non_max_suppression(boxes, tracker_obj.nms_max_overlap, scores)
            tr_detections = [tr_detections[i] for i in indices]
                
            for det in tr_detections:
                bbox = det.to_tlbr()
                
            # Call the tracker
            tracker.predict()
            tracker.update(tr_detections)            
            
            #import pdb; pdb.set_trace() 
            #flag = False            
            for track in tracker.tracks:
                if not track.is_confirmed() or track.time_since_update > 1:
                    continue 
                bbox = track.to_tlbr()
                cv2.rectangle(image, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(255,255,255), 2)
                cv2.putText(image, str(track.track_id),(int(bbox[0]), int(bbox[1])),0, 5e-3 * 200, (0,255,0),2) 
                writeLog(out_log , frame_num , track.track_id , np.array(bbox,dtype= 'int16') , imsize=[sc_width,sc_height])
                #if min(bbox) < 0 :
                #   flag = True

            cv2.imshow("Frame", image)
            out.write(image)
            key = cv2.waitKey(1) & 0xFF
            #if flag == True:
            #    import pdb ; pdb.set_trace()
            if key == ord("q"):# if the `q` key was pressed, break from the loop
                break
                cap.release()
                out.release()
                out_log.close()
                print("-------------- Frame:",frame_num)
                print("\n\n ============ output written as : " , out_filename , " and " ,  out_logname)     
            frame_num += 1
    
    print("-------------- Frame:",frame_num)
    cap.release()  
    out.release() 
    out_log.close()
    cv2.destroyAllWindows()  
    print("\n\n ============ output written as : " , out_filename , " and " ,  out_logname)          