import numpy as np
import cv2
import math
import pandas as pd


class Sort_bbox():

    def __init__(self):
        self.bbox = []
        self.function_dict = {}
        self.corner = {}
        self.sort_count = 0
        self.count_flag = {0:False, 1:False, 2:False, 3:False, 4:False,5:False,
                           6:False, 7:False, 8:False, 9:False, 10:False, 11:False,
                           12:False, 13:False,14:False,15:False,16:False}

    def insert(self,bbox):
        self.bbox.append(bbox)

    def call(self):
        return np.array(self.bbox)

    ''' y = k(x-a1)+b'''
    def linear_regression(self,line):
        x1,x2 = line[0][0] , line[1][0]
        b = x1[1]
        k = (x2[1] - x1[1]) / (x2[0] - x1[0])
        a1 = x1[0]
        return k,b,a1

    def calculate_distance(self,line):
        x1, x2 = line[0][0][0], line[1][0][0]
        y1, y2 = line[0][0][1], line[1][0][1]
        dis = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        return dis


    def sort(self,bbox,img,labellist):
        # store list
        cx_list = []
        cy_list = []
        for i in range(len(bbox)):
            cx_list.append(bbox[i][4])
            cy_list.append(bbox[i][5])
        # find max
        zero_matrix = np.zeros(len(bbox))
        bbox = np.insert(bbox, len(bbox[0]), zero_matrix, axis=1)  # sort_id
        bbox = np.insert(bbox, len(bbox[0]), zero_matrix, axis=1)  # in or out
        rowmin_id = bbox.argmin(axis=0)
        rowmax_id = bbox.argmax(axis=0)
        # four corner
        leftcorner = bbox[rowmin_id[4]] # cx min
        topcorner = bbox[rowmin_id[5]] # cy min
        rightcorner = bbox[rowmax_id[4]] # cx min
        downcorner = bbox[rowmax_id[5]] # cy max
        # sotre in dict
        self.corner[0] = bbox[rowmin_id[4]]
        self.corner[1] = bbox[rowmin_id[5]]
        self.corner[2] = bbox[rowmax_id[4]]
        self.corner[3] = bbox[rowmax_id[5]]
        # four line
        line1 = np.array([[leftcorner[4], leftcorner[5]], [topcorner[4], topcorner[5]]], np.int32).reshape((-1, 1, 2))
        line2 = np.array([[topcorner[4], topcorner[5]], [rightcorner[4], rightcorner[5]]], np.int32).reshape((-1, 1, 2))
        line3 = np.array([[rightcorner[4], rightcorner[5]], [downcorner[4], downcorner[5]]], np.int32).reshape((-1, 1, 2))
        line4 = np.array([[downcorner[4], downcorner[5]], [leftcorner[4], leftcorner[5]]], np.int32).reshape((-1, 1, 2))
        # y = k(x-a1)+b
        self.function_dict[0] = self.linear_regression(line1)
        self.function_dict[1] = self.linear_regression(line2)
        self.function_dict[2] = self.linear_regression(line3)
        self.function_dict[3] = self.linear_regression(line4)
        # after get four conrner delete hole
        delete_count = 0
        for id,label in enumerate(labellist):
            if label == 3:
                bbox = np.delete(bbox, id - delete_count, axis=0)
                delete_count += 1
        # predcit
        for i in range(len(self.function_dict)):
            k, b, a1 = self.function_dict[i]
            x = bbox.T[4]
            y_true = bbox.T[5]
            y_pred = k *(x -a1) + b
            dif = abs(y_pred - y_true) # 寻找每一个线性预测误差
            index_list = np.where(dif < 50) # 找到满足的索引
            index_dict = {} # 用于存放index 和距离的
            for _,index in enumerate(list(index_list[0])): # 计算回归后值差距较小的索引与角点的距离
                x_dis = self.corner[i][4] - bbox[index][4] # 横向距离
                y_dis = self.corner[i][5] - bbox[index][5]  # 纵向距离
                distance = x_dis**2 + y_dis**2
                index_dict[index] = distance
            sort_dict = sorted(index_dict.items(), key=lambda x: x[1]) # 根据字典value排序
            for _,data in enumerate(sort_dict):
                index = data[0] # 取字典的键做为变量
                if self.count_flag[index] is False: # 根据flag位遍历
                    self.count_flag[index] = True
                    bbox[index][6] = self.sort_count # 根据距离依次赋值
                    bbox[index][7] = 0  # 显示外圈
                    label = int(bbox[index][5])
                    x1, y1 = int(bbox[index][0]), int(bbox[index][1])
                    image_new = cv2.putText(img,str(int(bbox[index][6])), (x1+10, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.75, [0,0,0], 2)
                    self.sort_count += 1
                else:
                    continue
        # middle line
        dis1 = self.calculate_distance(line1)
        dis2 = self.calculate_distance(line2)
        dis3 = self.calculate_distance(line3)
        dis4 = self.calculate_distance(line4)
        # 找到较短边给其赋线性回归的函数
        if dis1 > dis2 and dis3 > dis4: #用 line 1 和line 3
            k = (self.function_dict[0][0] + self.function_dict[2][0])/2
            b = (self.function_dict[0][1] + self.function_dict[2][1]) / 2
            a1 = (self.function_dict[0][2] + self.function_dict[2][2]) / 2
        else:
            k = (self.function_dict[1][0] + self.function_dict[3][0])/2
            b = (self.function_dict[1][1] + self.function_dict[3][1]) / 2
            a1 = (self.function_dict[1][2] + self.function_dict[3][2]) / 2
        x = bbox.T[4]
        y_true = bbox.T[5]
        y_pred = k * (x - a1) + b
        dif = abs(y_pred - y_true)  # 寻找每一个线性预测误差
        index_list = np.where(dif < 50)  # 找到满足的索引
        index_dict = {}  # 用于存放index 和距离的
        for _, index in enumerate(list(index_list[0])):  # 计算回归后值差距较小的索引与角点的距离
            x_dis = self.corner[i][4] - bbox[index][4]  # 横向距离
            y_dis = self.corner[i][5] - bbox[index][5]  # 纵向距离
            distance = x_dis ** 2 + y_dis ** 2
            index_dict[index] = distance
        sort_dict = sorted(index_dict.items(), key=lambda x: x[1])  # 根据字典value排序
        for _, data in enumerate(sort_dict):
            index = data[0]  # 取字典的键做为变量
            if self.count_flag[index] is False:  # 根据flag位遍历
                self.count_flag[index] = True
                bbox[index][6] = self.sort_count  # 根据距离依次赋值
                bbox[index][7] = 1  # 显示外圈
                label = int(bbox[index][5])
                x1, y1 = int(bbox[index][0]), int(bbox[index][1])
                image_new = cv2.putText(img, str(int(bbox[index][6])), (x1 + 10, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                                        [0, 0, 0], 2)
                self.sort_count += 1
            else:
                continue


        imgnew = cv2.polylines(image_new, pts=[line1, line2, line3, line4], isClosed=False, color=[0,0,0], thickness=2, lineType=cv2.LINE_8)

        data = {'x': cx_list,
                'y': cy_list}
        # 创建DataFrame
        df = pd.DataFrame(data)
        # 保存数据为CSV文件
        df.to_csv('pca/data.csv', index=False)

        return imgnew,bbox

