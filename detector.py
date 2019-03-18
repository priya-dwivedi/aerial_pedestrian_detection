import cv2 
import numpy as np

def image_fprop(net , image):
    if type(image) == str :
        image = cv2.imread(image)
    detections = net.forward(image)
    return detections

def overlay_image(image , box):
    (startY, startX, endY, endX) = box.astype("int")
    cv2.rectangle(image, (startX, startY), (endX, endY),(0,255,0), 2) 
 
def detect_video(net,vidname , detection_threshold = 0.8 , scale = 1.0 , skip = 0 ):
    frame_num = 0
    cap = cv2.VideoCapture(vidname)
    W = cap.get(3)
    H = cap.get(4) 
    video_size = [int(W),int(H)]
    length = int(cap.get(7))
    vid_fps = cap.get(5)        
    print(video_size)
    print("Video Size :",video_size)
    print("FPS : ",vid_fps)
    while True:
        if frame_num % (skip+1)  == 0 : 
            ret, image = cap.read()
            if not ret:break        
            #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
            sc_width,sc_height = [int(W*scale) , int(H*scale)]
            if scale != 1:
                print(sc_width,sc_height)
                image = cv2.resize(image, (sc_width,sc_height), interpolation=cv2.INTER_AREA)
            image[0:260,:] = 0
            detections = image_fprop(net,image)
            print("Num detections:" , detections['num_detections'])
            
            for i in range(int(detections['num_detections'])):
                if detections['detection_scores'][i] > detection_threshold :
                    box = detections['detection_boxes'][i] * np.array([sc_height, sc_width, sc_height, sc_width])     
                    #import pdb; pdb.set_trace()  
                    overlay_image(image , box)
            cv2.imshow("Frame", image)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):# if the `q` key was pressed, break from the loop
                break
                cap.release()
                print("-------------- Frame:",frame_num)
            frame_num += 1
    
    print("-------------- Frame:",frame_num)
    cap.release()      
    cv2.destroyAllWindows()            