# Deepsort with RetinaNet

Deepsort forked from : https://github.com/nwojke/deep_sort 

### Requirements :

* python = 3.6

* Tensorflow >= 1.12

* numpy >= 1.14.5

* cv2 >= 3.4.1



### Inputs : 

* Video 

* Trained RetinaNet model
* Trained deepsort model

### Args :

-m , --model :  Path to the trained RetinaNet model

-tm , --tracker_model  : Path to tensorflow frozen graph of tracker (deepsort) model 

-c  , --confidence : Detection threshold for the RetinaNet model .  default = 0.8

-v , --video :  Path to input video file

-f , --fps :  FPS for output video .  default=25 

-s , --skip : Sampling interval for the inference to run . default=0 ie., no skipping of frames

-sc , --scale : Scaling for the input frame . default=1 ie., no scaling

-age , --max_age :  Max frames to wait before discarding a vanished ID . default=100

 -n_init , --n_init  :  Number of frames that a track remains in initialization phase .  default=1 




### Sample Args :

#### For simple deepsort :

python run.py -v video.mov -m snapshots/version8_resplit_test_train/resnet50_csv_12_inference.h5 -tm deep_sort/resources/networks/mars-small128.pb -n_init 5 -age 100 -c 0.5    -s 0 -sc 0.5

______________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________


