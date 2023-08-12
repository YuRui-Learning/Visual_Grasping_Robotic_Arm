import glob
from ultralytics import YOLO
import cv2
import argparse
import numpy as np
from configs.local_variable import COLOR_MAP,CLASS_MAP
from utils.draw import draw_cross
from utils.fillter import Blot_class
from utils.sort import Sort_bbox
from input.dictonary import Blot_bbox
from sort.function import send_sort
import os

parser = argparse.ArgumentParser(description='parser example')
parser.add_argument('--checkpoint', default="checkpoint/best.pt", type=str, help='checkpoint ')
parser.add_argument('--sigma', default=0.1, type=int, help='sigma ')
parser.add_argument('--iou_thershold', default=0.3, type=int, help='sigma ')
parser.add_argument('--confidence_thershold', default=0.4, type=int, help='sigma ')
args = parser.parse_args()

file_list = glob.glob(r'dataset/*.bmp')



def function():
    for id,image in enumerate(file_list):
        picture_num = id
        img = cv2.imread(image)
        model = YOLO(args.checkpoint)
        sigma = args.sigma
        iou_thershold = args.iou_thershold
        confidence_thershold = args.confidence_thershold
        predict_result = model.predict(img,conf = confidence_thershold,iou = iou_thershold)
        blot_class = Blot_class() # to solve class not fit
        blot_bbox = Blot_bbox() # store every class bbox
        sort_bbox = Sort_bbox() # sort the bbox
        label_list = [] # 缓存label
        for result in predict_result:
            for index,bbox in enumerate(result.boxes):
                bbox = bbox.data.numpy().T
                label = int(bbox[5])
                blot_bbox.insert(label,bbox[:4].T.squeeze())

                x1, y1, x2, y2 = int(bbox[0]),int(bbox[1]),int(bbox[2]),int(bbox[3])
                weight, height = x2-x1,y2-y1
                cx,cy = x1 + weight//2, y1 + height//2
                bbox_cache = bbox.copy()  # shallow copy
                # store cx，cy，sort id，inout
                bbox_cache[4],bbox_cache[5],= cx, cy
                sort_bbox.insert(bbox_cache[:8].T.squeeze())
                if bbox[4] > 0.65:
                    blot_class.update(weight,height,label)
                elif bbox[4] < 0.65:
                    height_wight_rate = weight / blot_class.dictonry[label]['width_mean'] + height / blot_class.dictonry[label]['height_mean']
                    while height_wight_rate < 2 - sigma or height_wight_rate > 2 + sigma:
                        label += 1
                        label %= 3
                        if label == int(bbox[5]):
                            break
                        height_wight_rate = weight / blot_class.dictonry[label]['width_mean'] + height / \
                                            blot_class.dictonry[label]['height_mean']
                    print("index:%d,weight:%d,height:%d,class:%d"%(index,weight,height,label))
                label_list.append(label) # 这个时候的label是正确的label
                image_new = cv2.rectangle(img,(x1, y1),(x2, y2),COLOR_MAP[label],2)
                image_new = cv2.putText(image_new,CLASS_MAP[label], (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.75, COLOR_MAP[label], 2)
                image_new = draw_cross(image_new,[cx,cy],COLOR_MAP[label],3)
            input_bbox = sort_bbox.call()
            image_new,bbox_new = sort_bbox.sort(input_bbox,img)
            # x1,y1,x2,y2,label,cx,cy,sort_id,inout
            label_ndarray = np.array(label_list)
            bbox_new = np.insert(bbox_new, 4, label_ndarray, axis=1)
            sort_list = send_sort(bbox_new)

            for id,data in enumerate(sort_list):
                num = str(id)
                cx = int(data[5])
                cy = int(data[6])

                image_new = cv2.putText(image_new, num, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                                        (255,255,255), 2)

        picture_name = str(picture_num) + ".jpg"
        save_path = os.path.join("output", picture_name)
        cv2.imwrite(save_path, image_new)
        # cv2.imshow("blot picture", image_new)
        # cv2.waitKey(0)


def infer_one_picture(imgpath,args):
    img = cv2.imread(imgpath)
    model = YOLO(args.checkpoint)
    sigma = args.sigma
    iou_thershold = args.iou_thershold
    confidence_thershold = args.confidence_thershold
    predict_result = model.predict(img,conf = confidence_thershold,iou = iou_thershold)
    blot_class = Blot_class() # to solve class not fit
    blot_bbox = Blot_bbox() # store every class bbox
    sort_bbox = Sort_bbox() # sort the bbox
    label_list = [] # 缓存label
    for result in predict_result:
        for index,bbox in enumerate(result.boxes):
            bbox = bbox.data.numpy().T
            label = int(bbox[5])
            blot_bbox.insert(label,bbox[:4].T.squeeze())

            x1, y1, x2, y2 = int(bbox[0]),int(bbox[1]),int(bbox[2]),int(bbox[3])
            weight, height = x2-x1,y2-y1
            cx,cy = x1 + weight//2, y1 + height//2
            bbox_cache = bbox.copy()  # shallow copy
            # store cx，cy，sort id，inout
            bbox_cache[4],bbox_cache[5],= cx, cy
            sort_bbox.insert(bbox_cache[:8].T.squeeze())
            if bbox[4] > 0.65:
                blot_class.update(weight,height,label)
            elif bbox[4] < 0.65:
                height_wight_rate = weight / blot_class.dictonry[label]['width_mean'] + height / blot_class.dictonry[label]['height_mean']
                while height_wight_rate < 2 - sigma or height_wight_rate > 2 + sigma:
                    label += 1
                    label %= 3
                    if label == int(bbox[5]):
                        break
                    height_wight_rate = weight / blot_class.dictonry[label]['width_mean'] + height / \
                                        blot_class.dictonry[label]['height_mean']
                print("index:%d,weight:%d,height:%d,class:%d"%(index,weight,height,label))
            label_list.append(label) # 这个时候的label是正确的label
            image_new = cv2.rectangle(img,(x1, y1),(x2, y2),COLOR_MAP[label],2)
            image_new = cv2.putText(image_new,CLASS_MAP[label], (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.75, COLOR_MAP[label], 2)
            image_new = draw_cross(image_new,[cx,cy],COLOR_MAP[label],3)
        input_bbox = sort_bbox.call()
        image_new,bbox_new = sort_bbox.sort(input_bbox,img)
        # x1,y1,x2,y2,label,cx,cy,sort_id,inout
        label_ndarray = np.array(label_list)
        bbox_new = np.insert(bbox_new, 4, label_ndarray, axis=1)
        sort_list = send_sort(bbox_new)

        for id,data in enumerate(sort_list):
            num = str(id)
            cx = int(data[5])
            cy = int(data[6])

            image_new = cv2.putText(image_new, num, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                                    (255,255,255), 2)
    return sort_list, image_new



if __name__ == '__main__':
    function()
    # imagepath = 'dataset/Image_20230731225958746.bmp'
    # sort_list,image_new = infer_one_picture(imagepath,args)
    # cv2.imshow("blot picture", image_new)
    # cv2.waitKey(0)