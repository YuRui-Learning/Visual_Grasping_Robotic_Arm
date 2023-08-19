import glob
from ultralytics import YOLO
import cv2
import argparse
import numpy as np
from configs.local_variable import COLOR_MAP,CLASS_MAP,SIZE_RATE
from utils.draw import draw_cross
from utils.fillter import Blot_class
from input.dictonary import Blot_bbox
from sort.function import send_sort
from hough.get_center import hough_center
import os

parser = argparse.ArgumentParser(description='parser example')
parser.add_argument('--checkpoint', default="checkpoint/best_hole_new.pt", type=str, help='checkpoint ')
parser.add_argument('--sigma', default=0.2, type=int, help='sigma ')
parser.add_argument('--iou_thershold', default=0.3, type=int, help='iou_theshold')
parser.add_argument('--confidence_thershold', default=0.2, type=int, help='confidence_thershold ')
parser.add_argument('--sort_solution', default='pca', type=str, help='pca or corner use in sort')
parser.add_argument('--center_solution', default='none', type=str, help='hough or yolo use in get center')
parser.add_argument('--save_screw_picture', default=False, type=bool, help='whether to save every screw picture')
parser.add_argument('--picture_num', default='cluster', type=str, help='cluster or single use to infer number')
args = parser.parse_args()

file_list = glob.glob(r'dataset_0818/*.bmp')



def function(model):
    for id,image in enumerate(file_list):
        picture_num = id # store in this sort sequence
        img = cv2.imread(image) # read picture
        '''load some param'''
        pictutre_every_num = 0  # save every screw picture
        sigma = args.sigma  # size threshold
        iou_thershold = args.iou_thershold
        confidence_thershold = args.confidence_thershold
        sort_solution = args.sort_solution  # use in get every screw solution
        center_solution = args.center_solution  # use in get every screw center
        flag_save_screw = args.save_screw_picture
        '''choose to use which solution to sort'''
        if sort_solution == 'pca':
            from pca.sort import Sort_bbox
        elif sort_solution == 'corner':
            from utils.sort import Sort_bbox
        '''infernence'''
        predict_result = model.predict(img, conf=confidence_thershold, iou=iou_thershold, agnostic_nms=True)
        '''init some class'''
        blot_class = Blot_class()  # to solve class not fit, only store L M S
        sort_bbox = Sort_bbox()  # sort the bbox only sort in L M S
        '''some list to cache data'''
        label_list = []  # cache all label
        screw_label = []
        for result in predict_result:
            for index, bbox in enumerate(result.boxes):
                '''get base info'''
                bbox = bbox.data.numpy().T
                label = int(bbox[5])
                x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                weight, height = x2 - x1, y2 - y1
                cx, cy = x1 + weight // 2, y1 + height // 2
                '''img split'''
                x1_new, x2_new, y1_new, y2_new = int(x1 / 1.05), int(x2 / 0.95), int(y1 / 1.05), int(y2 / 0.95)
                img_split = img[y1_new:y2_new, x1_new:x2_new]
                '''color get '''
                x1_color, x2_color, y1_color, y2_color = int(x1_new * 1.05), int(x2_new * 0.95), int(
                    y1_new * 1.05), int(y2_new * 0.95)
                img_split_color = img[y1_color:y2_color, x1_color:x2_color]
                unique_colors, counts = np.unique(img_split_color, return_counts=True)
                if counts.mean() < 60:
                    label = 3
                '''save every screw picture'''
                pictutre_every_num += 1
                if flag_save_screw is True:
                    picture_every_name = str(pictutre_every_num) + ".jpg"
                    save_path = os.path.join("output/screw", picture_every_name)
                    cv2.imwrite(save_path, img_split)
                '''yolo v8 predcit'''
                if center_solution == 'yolo':
                    img_split_predict_result = model.predict(img_split, conf=confidence_thershold, iou=iou_thershold,
                                                             agnostic_nms=True)
                    xywh = img_split_predict_result[0].boxes.xywh
                    if len(img_split_predict_result[0].boxes) != 0:
                        cx, cy = x1_new + xywh[0][0], y1_new + xywh[0][1]
                '''hough'''
                if center_solution == 'hough':
                    split_cx, split_cy = hough_center(img_split)
                    if split_cx != 0 and split_cy != 0:
                        cx, cy = x1_new + split_cx, y1_new + split_cy

                '''store cx，cy，sort id，inout'''
                bbox_cache = bbox.copy()  # shallow copy
                bbox_cache[4], bbox_cache[5], = cx, cy
                sort_bbox.insert(bbox_cache[:8].T.squeeze())  # insert every class
                '''fit size result'''
                if label < 3:  # use in L M S
                    if pictutre_every_num == 1:
                        update_thershold = 0.75
                    else:
                        update_thershold = 0.86
                    if bbox[4] > update_thershold:  # update the result
                        blot_class.update(weight, height, label)
                    # fit the result when M and S low confidence,only use when blot_class num is not 0
                    if blot_class.dictonry[label]['number'] != 0:
                        height_wight_rate = weight / blot_class.dictonry[label]['width_mean'] + height / \
                                            blot_class.dictonry[label]['height_mean']
                        if 2 - sigma < height_wight_rate < 2 + sigma:
                            blot_class.update(weight, height, label)
                    else:
                        height_wight_rate = weight / (blot_class.dictonry[0]['width_mean'] / SIZE_RATE[label][0]) + \
                                            height / (blot_class.dictonry[0]['height_mean'] / SIZE_RATE[label][1])
                        # satisfy threshold updata blot_class mean dict
                        if 2 - sigma < height_wight_rate < 2 + sigma:
                            blot_class.update(weight, height, label)
                    '''find the label in loop'''
                    while height_wight_rate < 2 - sigma or height_wight_rate > 2 + sigma:
                        label += 1
                        label %= 3  # label % 3 == init label -->break
                        if label == int(bbox[5]):
                            break
                        if blot_class.dictonry[label]['number'] != 0:
                            height_wight_rate = weight / blot_class.dictonry[label]['width_mean'] + height / \
                                                blot_class.dictonry[label]['height_mean']
                            if 2 - sigma < height_wight_rate < 2 + sigma:
                                blot_class.update(weight, height, label)
                        else:
                            height_wight_rate = weight / (blot_class.dictonry[0]['width_mean'] / SIZE_RATE[label][0]) + \
                                                height / (blot_class.dictonry[0]['height_mean'] / SIZE_RATE[label][1])
                            # meanwhile if satisify update dictionary
                            if 2 - sigma < height_wight_rate < 2 + sigma:
                                blot_class.update(weight, height, label)
                '''get true label ,meanwhile filter the hole'''
                label_list.append(label)  # 这个时候的label是正确的label
                if label != 3:  # screw store and visual
                    screw_label.append(label)
            '''sort part code API '''
            input_bbox = sort_bbox.call()  # list
            image_new, bbox_new = sort_bbox.sort(input_bbox, img, label_list)  # snake matrix to unifrom ndarray
            label_ndarray = np.array(screw_label)
            bbox_new = np.insert(bbox_new, 4, label_ndarray, axis=1)  # x1,y1,x2,y2,label,cx,cy,sort_id,inout
            ''' last result ndarray which u can get every screw in sequence'''
            sort_list = send_sort(bbox_new)
            '''add some info to img'''
            for id, data in enumerate(sort_list):
                num = str(id)
                cx = int(data[5])
                cy = int(data[6])
                x1, y1, x2, y2 = int(data[0]), int(data[1]), int(data[2]), int(data[3])
                label = int(data[4])
                image_new = cv2.rectangle(img, (x1, y1), (x2, y2), COLOR_MAP[label], 2)
                image_new = cv2.putText(image_new, CLASS_MAP[label], (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                                        COLOR_MAP[label], 2)
                image_new = draw_cross(image_new, [cx, cy], COLOR_MAP[label], 3)
                image_new = cv2.putText(image_new, num, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                                        (255, 255, 255), 2)

        picture_name = str(picture_num) + ".jpg"
        save_path = os.path.join("output", picture_name)
        cv2.imwrite(save_path, image_new)
        # cv2.imshow("blot picture", image_new)
        # cv2.waitKey(0)

def infer_one_picture(imgpath,args,model):
    '''
    :param imgpath: every picture path
    :param args: argparse(checkpoint, sigma, iou and con_threshold, sort and center solution)
    :param model: YOLO V8model
    :return:
     sort_list：x1,y1,x2,y2,label,cx,cy,sort_id,inout, sequence every ndarry
     img_new:result picture
    '''
    img = cv2.imread(imgpath)  # read picture
    '''load some param'''
    pictutre_every_num = 0  # save every screw picture
    sigma = args.sigma  # size threshold
    iou_thershold = args.iou_thershold
    confidence_thershold = args.confidence_thershold
    sort_solution = args.sort_solution  # use in get every screw solution
    center_solution = args.center_solution  # use in get every screw center
    flag_save_screw = args.save_screw_picture
    '''choose to use which solution to sort'''
    if sort_solution == 'pca':
        from pca.sort import Sort_bbox
    elif sort_solution == 'corner':
        from utils.sort import Sort_bbox
    '''infernence'''
    predict_result = model.predict(img, conf=confidence_thershold, iou=iou_thershold, agnostic_nms=True)
    '''init some class'''
    blot_class = Blot_class()  # to solve class not fit, only store L M S
    sort_bbox = Sort_bbox()  # sort the bbox only sort in L M S
    '''some list to cache data'''
    label_list = []  # cache all label
    screw_label = []
    for result in predict_result:
        for index, bbox in enumerate(result.boxes):
            '''get base info'''
            bbox = bbox.data.numpy().T
            label = int(bbox[5])
            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            weight, height = x2 - x1, y2 - y1
            cx, cy = x1 + weight // 2, y1 + height // 2
            '''img split'''
            x1_new, x2_new, y1_new, y2_new = int(x1 / 1.05), int(x2 / 0.95), int(y1 / 1.05), int(y2 / 0.95)
            img_split = img[y1_new:y2_new, x1_new:x2_new]
            '''color get '''
            x1_color,x2_color,y1_color,y2_color = int(x1_new * 1.05),int(x2_new * 0.95),int(y1_new * 1.05),int(y2_new * 0.95)
            img_split_color = img[y1_color:y2_color, x1_color:x2_color]
            unique_colors, counts = np.unique(img_split_color, return_counts=True)
            if counts.mean() < 60:
                label = 3
            '''save every screw picture'''
            pictutre_every_num += 1
            if flag_save_screw is True:

                picture_every_name = str(pictutre_every_num) + ".jpg"
                save_path = os.path.join("output/screw", picture_every_name)
                cv2.imwrite(save_path, img_split)
            '''yolo v8 predcit'''
            if center_solution == 'yolo':
                img_split_predict_result = model.predict(img_split, conf=confidence_thershold, iou=iou_thershold,
                                                         agnostic_nms=True)
                xywh = img_split_predict_result[0].boxes.xywh
                if len(img_split_predict_result[0].boxes) != 0:
                    cx, cy = x1_new + xywh[0][0], y1_new + xywh[0][1]
            '''hough'''
            if center_solution == 'hough':
                split_cx, split_cy = hough_center(img_split)
                if split_cx != 0 and split_cy != 0:
                    cx, cy = x1_new + split_cx, y1_new + split_cy

            '''store cx，cy，sort id，inout'''
            bbox_cache = bbox.copy()  # shallow copy
            bbox_cache[4], bbox_cache[5], = cx, cy
            sort_bbox.insert(bbox_cache[:8].T.squeeze())  # insert every class
            '''fit size result'''
            if label < 3:  # use in L M S
                if pictutre_every_num == 1:
                    update_thershold = 0.75
                else:
                    update_thershold = 0.86
                if bbox[4] > update_thershold:  # update the result
                    blot_class.update(weight, height, label)
                # fit the result when M and S low confidence,only use when blot_class num is not 0
                if blot_class.dictonry[label]['number'] != 0:
                    height_wight_rate = weight / blot_class.dictonry[label]['width_mean'] + height / \
                                        blot_class.dictonry[label]['height_mean']
                    if 2 - sigma < height_wight_rate < 2 + sigma:
                        blot_class.update(weight, height, label)
                else:
                    height_wight_rate = weight / (blot_class.dictonry[0]['width_mean'] / SIZE_RATE[label][0]) + \
                                        height / (blot_class.dictonry[0]['height_mean'] / SIZE_RATE[label][1])
                    # satisfy threshold updata blot_class mean dict
                    if 2 - sigma < height_wight_rate < 2 + sigma:
                        blot_class.update(weight, height, label)
                '''find the label in loop'''
                while height_wight_rate < 2 - sigma or height_wight_rate > 2 + sigma:
                    label += 1
                    label %= 3  # label % 3 == init label -->break
                    if label == int(bbox[5]):
                        break
                    if blot_class.dictonry[label]['number'] != 0:
                        height_wight_rate = weight / blot_class.dictonry[label]['width_mean'] + height / \
                                            blot_class.dictonry[label]['height_mean']
                        if 2 - sigma < height_wight_rate < 2 + sigma:
                            blot_class.update(weight, height, label)
                    else:
                        height_wight_rate = weight / (blot_class.dictonry[0]['width_mean'] / SIZE_RATE[label][0]) + \
                                            height / (blot_class.dictonry[0]['height_mean'] / SIZE_RATE[label][1])
                        # meanwhile if satisify update dictionary
                        if 2 - sigma < height_wight_rate < 2 + sigma:
                            blot_class.update(weight, height, label)
            '''get true label ,meanwhile filter the hole'''
            label_list.append(label)  # 这个时候的label是正确的label
            if label != 3:  # screw store and visual
                screw_label.append(label)
        '''sort part code API '''
        input_bbox = sort_bbox.call()  # list
        image_new, bbox_new = sort_bbox.sort(input_bbox, img, label_list)  # snake matrix to unifrom ndarray
        label_ndarray = np.array(screw_label)
        bbox_new = np.insert(bbox_new, 4, label_ndarray, axis=1)  # x1,y1,x2,y2,label,cx,cy,sort_id,inout
        ''' last result ndarray which u can get every screw in sequence'''
        sort_list = send_sort(bbox_new)
        '''add some info to img'''
        for id, data in enumerate(sort_list):
            num = str(id)
            cx = int(data[5])
            cy = int(data[6])
            x1, y1, x2, y2 = int(data[0]), int(data[1]), int(data[2]), int(data[3])
            label = int(data[4])
            image_new = cv2.rectangle(img, (x1, y1), (x2, y2), COLOR_MAP[label], 2)
            image_new = cv2.putText(image_new, CLASS_MAP[label], (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                                    COLOR_MAP[label], 2)
            image_new = draw_cross(image_new, [cx, cy], COLOR_MAP[label], 3)
            image_new = cv2.putText(image_new, num, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                                    (255, 255, 255), 2)
    return sort_list, image_new



if __name__ == '__main__':
    picture_num = args.picture_num
    model = YOLO(args.checkpoint)  # load model
    if picture_num == 'cluster':
        function(model)
    if picture_num == 'single':
        imagepath = 'dataset_0818/103.bmp'
        sort_list,image_new = infer_one_picture(imagepath,args,model)
        cv2.imshow("blot picture", image_new)
        cv2.waitKey(0)