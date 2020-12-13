import cv2
import numpy as np
from yolo.configs import *

# used
def draw_bbox(image, bboxes, colors, NUM_CLASS, show_label=True, Text_colors=(0,0,0),  tracking=False):
    image_h, image_w, _ = image.shape
    bbox_thick = int(0.6 * (image_h + image_w) / 1000) # relative box thickness
    if bbox_thick < 1: bbox_thick = 1 # minimum thickness
    fontScale = 0.75 * bbox_thick # font scaling


    for i, bbox in enumerate(bboxes):
        coor = np.array(bbox[:4], dtype=np.int32) # coordinates
        track_id = bbox[4] # deep sort tracking ID
        class_ind = int(bbox[5]) # yolo class ID
        bbox_color = colors[class_ind] # color against that class ID


        (x1, y1), (x2, y2) = (coor[0], coor[1]), (coor[2], coor[3])

        cv2.rectangle(image, (x1, y1), (x2, y2), bbox_color, bbox_thick*2) # bounding box

        if show_label:
            label = "{}".format(NUM_CLASS[class_ind]) # class name
            if tracking:
                label +=  " " + str(track_id)
            if len(bbox) == 7:
                label += " TTC:" + str(round(bbox[6], 2))
            # font size for relative label box size
            (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_COMPLEX_SMALL, fontScale, thickness=bbox_thick)  # text size

            # filled rectangle for label
            cv2.rectangle(image, (x1, y1), (x1 + text_width, y1 - text_height - baseline), bbox_color, thickness=cv2.FILLED)

            # label above rectangle
            cv2.putText(image, label, (x1, y1-4), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                        fontScale, Text_colors, bbox_thick, lineType=cv2.LINE_AA)

    return image


