import glob
from ultralytics import YOLO
import cv2
import argparse
import numpy as np
import os
from configs.local_variable import COLOR_MAP,CLASS_MAP
parser = argparse.ArgumentParser(description='parser example')
parser.add_argument('--checkpoint', default="checkpoint/best_hole.pt", type=str, help='checkpoint ')
parser.add_argument('--iou_thershold', default=0.4, type=int, help='sigma ')
parser.add_argument('--confidence_thershold', default=0.2, type=int, help='sigma ')
args = parser.parse_args()
from utils.draw import draw_cross
file_list = glob.glob(r'dataset/*.bmp')



def function():
    for id,image in enumerate(file_list):
        img = cv2.imread('dataset/Image_20230731231258253.bmp') # read picture
        model = YOLO(args.checkpoint) # load model
        '''load some parameter'''
        iou_thershold = args.iou_thershold
        confidence_thershold = args.confidence_thershold
        predict_result = model.predict(img,conf = confidence_thershold,iou = iou_thershold,agnostic_nms = True)
        for result in predict_result:
            for index,bbox in enumerate(result.boxes):
                bbox = bbox.data.numpy().T
                label = int(bbox[5])
                # blot_bbox.insert(label,bbox[:4].T.squeeze()) # insert one result which is to give the sequence
                x1, y1, x2, y2 = int(bbox[0]),int(bbox[1]),int(bbox[2]),int(bbox[3])
                weight, height = x2-x1,y2-y1
                cx,cy = x1 + weight//2, y1 + height//2
                bbox_cache = bbox.copy()  # shallow copy
                image_new = cv2.rectangle(img,(x1, y1),(x2, y2),COLOR_MAP[label],2)
                image_new = cv2.putText(image_new,CLASS_MAP[label], (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.75, COLOR_MAP[label], 2)
                image_new = draw_cross(image_new,[cx,cy],COLOR_MAP[label],3)

        cv2.imshow("blot picture", image_new)
        cv2.waitKey(0)




if __name__ == '__main__':
    function()