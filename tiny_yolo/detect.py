from __future__ import division

from .models import *
from .utils.utils import *
from .utils.datasets import *

import os
import sys
import time
import datetime
import argparse

import cv2

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

class Yolo:
    def __init__(self):
        self.model_def = 'tiny_yolo/config/yolov3.cfg'
        self.weights_path = 'tiny_yolo/weights/yolov3.weights'
        self.class_path = 'tiny_yolo/data/coco.names'
        self.conf_thres = 0.8
        self.nms_thres = 0.4
        self.batch_size = 1
        self.n_cpu = 0
        self.img_size = 416
        # self.checkpoint_model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = Darknet(self.model_def, img_size=self.img_size).to(self.device)
        if self.weights_path.endswith(".weights"):
            # Load darknet weights
            self.model.load_darknet_weights(self.weights_path)
        else:
            # Load checkpoint weights
            self.model.load_state_dict(torch.load(self.weights_path))
        # Set up model
        self.model.eval()  # Set in evaluation mode
        self.classes = load_classes(self.class_path)  # Extracts class labels from file
        self.Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    def image2torch(self, img):
        img = transforms.ToTensor()(img)
        img, _ = pad_to_square(img, 0)
        img = resize(img, self.img_size).unsqueeze(0)
        return img

    def predict(self, img):
        pytorch_img = self.image2torch(img)
        with torch.no_grad():
                d = self.model(pytorch_img)
                d = non_max_suppression(d, self.conf_thres, self.nms_thres)
        detections = rescale_boxes(d[0], self.img_size, img.shape[:2])
        unique_labels = detections[:, -1].cpu().unique()
        n_cls_preds = len(unique_labels)
        results = []
        for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
            results.append([x1, y1, x2, y2, cls_conf.item(), cls_pred])
        return results
