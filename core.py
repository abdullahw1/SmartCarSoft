import os
import cv2
import random
import colorsys
import numpy as np
from yolo.utils import draw_bbox
from yolo.configs import *
from tiny_yolo.detect import Yolo
import time
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from deep_sort import generate_detections as gdet
from CollusionDetection.Predictor import time_to_contact
from playsound import playsound


class core():
    def __init__(self):
        self.image = 0
        self.Yolo = Yolo()

    def getImg(self):
        return self.image

    def read_class_names(self, class_file_name):
        # loads class name from a file
        names = {}
        with open(class_file_name, 'r') as data:
            for ID, name in enumerate(data):
                names[ID] = name.strip('\n')
        return names


    def system(self, video_path, output_path, input_size=416, show=False, CLASSES=YOLO_COCO_CLASSES, score_threshold=0.3, iou_threshold=0.45, rectangle_colors='', Track_only = [], display_tm = False, realTime = True ):

        # Definition of the  deep sort parameters
        max_cosine_distance = 0.7
        nn_budget = None

        #initialize deep sort object
        model_filename = 'model_data/mars-small128.pb' # deep sort tensorflow pretrained model
        encoder = gdet.create_box_encoder(model_filename, batch_size=1)
        metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
        tracker = Tracker(metric)

        times, times_2 = [], [] #parameters for finding fps

        if video_path:
            vid = cv2.VideoCapture(video_path) # detect on video
        else:
            vid = cv2.VideoCapture(0) # detect from webcam

        # by default VideoCapture returns float instead of int
        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(vid.get(cv2.CAP_PROP_FPS))
        codec = cv2.VideoWriter_fourcc(*'MPEG') # defining video writer
        out = cv2.VideoWriter(output_path, codec, fps, (width, height)) # output_path must be .avi

        NUM_CLASS = self.read_class_names(CLASSES) # reading coco classes in the form of key value
        num_classes = len(NUM_CLASS)
        key_list = list(NUM_CLASS.keys())
        val_list = list(NUM_CLASS.values())

        # calculating parameters for img processing fucntion
        loop_check, original_frame = vid.read()
        if not loop_check:
            print("\n\nCouldn't read the video")
            return False
        # colors for detection
        hsv_tuples = [(1.0 * x / num_classes, 1., 1.) for x in range(num_classes)]
        detection_colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        detection_colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), detection_colors))
        # random.seed(0)
        random.shuffle(detection_colors) # to shuffle shades of same color
        # random.seed(None)
        newTime = 0
        prevTime = 0
        dummy_time = 1
        t3 = 0
        playsound('system_ready.wav')
        # loop for video
        while True:
            loop_check, original_frame = vid.read() # loop_check is bool value for reading correctly or not
            # cv2.imshow("org",original_frame)
            if not loop_check:
                return True
            prevTime = newTime
            newTime = time.time()
            t1 = time.time()
            bboxes = self.Yolo.predict(original_frame)
            t2 = time.time()
            # extract bboxes to boxes (x, y, width, height), scores and names
            boxes, scores, names = [], [], []
            #tracking
            for bbox in bboxes: #loop to sperate the bounding boxes in the frames
                if len(Track_only) !=0 and NUM_CLASS[int(bbox[5])] in Track_only or len(Track_only) == 0:
                    x1 = int(bbox[0])
                    y1 = int(bbox[1])
                    x2 = int(bbox[2])
                    y2 = int(bbox[3])
                    scoreVal = bbox[4]
                    class_id = int(bbox[5])
                    boxes.append([x1, y1, x2, y2])
                    scores.append(scoreVal)
                    label = NUM_CLASS[class_id]
                    names.append(label)
                    #self.image = cv2.rectangle(original_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

            # Obtain all the detections for the given frame.
            boxes = np.array(boxes)
            names = np.array(names)
            scores = np.array(scores)
            features = np.array(encoder(original_frame, boxes))
            # create deep sort object for detection
            detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature
                          in zip(boxes, scores, names, features)]

            print('Detection boxes: ', boxes)
            if realTime:
                tracked_bboxes = time_to_contact(original_frame, tracker.matchedBoxes, newTime, prevTime,
                                                 key_list, val_list, display_tm=display_tm)
            else:
                tracked_bboxes = time_to_contact(original_frame, tracker.matchedBoxes, dummy_time,
                                                 dummy_time - 0.01666666666, key_list, val_list, display_tm=display_tm)

            # Pass detections to the deepsort object and obtain the track information.
            tracker.predict()
            tracker.update(detections)

            # draw detection on frame
            self.image = draw_bbox(original_frame, tracked_bboxes, detection_colors, NUM_CLASS, tracking=True)

            # calculating fps
            t3 = time.time()
            times.append(t2-t1)
            times_2.append(t3-t1)

            times = times[-20:]
            times_2 = times_2[-20:]

            ms = sum(times)/len(times)*1000
            fps = 1000 / ms
            fps2 = 1000 / (sum(times_2)/len(times_2)*1000)

            print("Time: {:.2f}ms, Detection FPS: {:.1f}, total FPS: {:.1f}".format(ms, fps, fps2))
            if output_path != '': out.write(self.image)
            if show:
                cv2.imshow('Tracked', self.image)

                if cv2.waitKey(25) & 0xFF == ord("q"):
                    cv2.destroyAllWindows()
                    break

        cv2.destroyAllWindows()

