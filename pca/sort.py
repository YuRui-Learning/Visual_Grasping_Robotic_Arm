import numpy as np
import cv2
import math
import pandas as pd
from sklearn.linear_model import LinearRegression
import pandas as pd
import matplotlib.pyplot as plt
from numpy import *
import numpy as np


class Sort_bbox():

    def __init__(self):
        self.bbox = []
        self.function_dict = {}
        self.corner = {}
        self.sort_count = 0
        self.count_flag = {0:False, 1:False, 2:False, 3:False, 4:False,5:False,
                           6:False, 7:False, 8:False, 9:False, 10:False, 11:False,
                           12:False, 13:False,14:False,15:False,16:False}
        self.k_dict = {"++":0,
                        "+-":0,
                        "--":0,
                        "-+":0}

        self.dis_center = {"++":[],
                            "+-":[],
                            "--":[],
                            "-+":[]}
        self.index_center = {"++":[],
                            "+-":[],
                            "--":[],
                            "-+":[]}

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

    def linearmodel(self,recondata):
        # inear model fit
        model = LinearRegression()
        x = np.array(recondata[:, 0])
        y = np.array(recondata[:, 1])
        model.fit(x, y)
        # 获取拟合参数
        slope = model.coef_[0]
        slope_v = -1 / slope
        return slope, slope_v

    def calculate_distance(self,line):
        x1, x2 = line[0][0][0], line[1][0][0]
        y1, y2 = line[0][0][1], line[1][0][1]
        dis = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        return dis

    def line1(self,x, slope, cx, cy):
        return slope * (x - cx) + cy

    def line2(self,x, slope_v, cx, cy):
        return slope_v * (x - cx) + cy

    def get_four_corner(self,data,slope,cx,cy):
        slope_v = -1 / slope
        for id, data_item in enumerate(data):
            diff_line1 = self.line1(data_item[0], slope, cx, cy) - data_item[1]
            diff_line2 = self.line2(data_item[0], slope_v, cx, cy) - data_item[1]
            strflag = ""
            if diff_line1 < 0:
                strflag += "+"
            else:
                strflag += "-"
            if diff_line2 < 0:
                strflag += "+"
            else:
                strflag += "-"
            x_dis = data_item[0] - cx  # 横向距离
            y_dis = data_item[1] - cy  # 纵向距离
            distance = x_dis ** 2 + y_dis ** 2
            self.dis_center[strflag].append(distance) # get every conrner result
            self.index_center[strflag].append(id) # append id
            strflag += str(id)

    def get_four_line(self,datamatrix, slope,cx,cy,bbox):
        '''
        :param datamatrix: data matrix
        :param slope:
        :param cx:
        :param cy:
        :param bbox: bounding bbox
        :return:
        '''
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(datamatrix[:, 0].flatten().A[0], datamatrix[:, 1].flatten().A[0], marker='^', s=90)
        x_min, x_max = min(datamatrix[:, 0]) - 200, max(datamatrix[:, 0]) + 200
        y_min, y_max = min(datamatrix[:, 1]) - 200, max(datamatrix[:, 1]) + 200
        ax.set_xlim([x_min, x_max])
        ax.set_ylim([y_min, y_max])
        print(x_min[0])
        x = np.linspace(int(x_min[0]), int(x_max[0]), 10)
        # 计算y的值
        y = slope * (x - cx) + cy
        # 绘制直线
        plt.plot(x, y)

        x = np.linspace(int(x_min[0]), int(x_max[0]), 10)
        # get vertival
        slope_v = -1 / slope
        y = slope_v * (x - cx) + cy
        # 绘制直线
        plt.plot(x, y)
        data = np.array(datamatrix)
        function_idx = 0
        for key in self.dis_center.keys():
            max_dict_index = self.dis_center[key].index(max(self.dis_center[key])) # find max dis in every four dict
            max_index = self.index_center[key][max_dict_index] # find index
            a1, b = data[max_index]
            k = self.k_dict[key]
            x = np.linspace(int(x_min[0]), int(x_max[0]), 10)
            y = k * (x - a1) + b
            self.function_dict[function_idx] = k,b,a1
            self.corner[function_idx] = bbox[max_index]
            function_idx += 1
            plt.plot(x, y)

    def pcafunction(self,dataMat, topNfeat=9999999):
        meanVals = mean(dataMat, axis=0)
        meanRemoved = dataMat - meanVals  # remove mean
        covMat = cov(meanRemoved, rowvar=0)
        eigVals, eigVects = linalg.eig(mat(covMat))
        eigValInd = argsort(eigVals)  # sort, sort goes smallest to largest
        eigValInd = eigValInd[:-(topNfeat + 1):-1]  # cut off unwanted dimensions
        redEigVects = eigVects[:, eigValInd]  # reorganize eig vects largest to smallest
        lowDDataMat = meanRemoved * redEigVects  # transform data into new dimensions
        reconMat = (lowDDataMat * redEigVects.T) + meanVals
        return lowDDataMat, reconMat

    def sort(self,bbox,img,labellist):
        zero_matrix = np.zeros(len(bbox))
        bbox = np.insert(bbox, len(bbox[0]), zero_matrix, axis=1)  # sort_id
        bbox = np.insert(bbox, len(bbox[0]), zero_matrix, axis=1)  # in or out

        # store list
        cx_list = []
        cy_list = []
        for i in range(len(bbox)):
            cx_list.append(bbox[i][4])
            cy_list.append(bbox[i][5])

        data = {'x': cx_list,
                'y': cy_list}
        # 创建DataFrame
        df = pd.DataFrame(data)
        # find max
        # 打印DataFrame
        datamatrix = mat(np.array(df))
        # pca
        _, recondata = self.pcafunction(datamatrix, 1)
        cx = datamatrix[:, 0].mean()
        cy = datamatrix[:, 1].mean()
        slope, slope_v = self.linearmodel(recondata)
        data = np.array(datamatrix)
        self.get_four_corner(data, slope, cx, cy)
        self.k_dict["++"], self.k_dict["+-"], self.k_dict["--"], self.k_dict["-+"] = slope, slope_v, slope, slope_v
        self.get_four_line(datamatrix,slope,cx,cy,bbox)
        plt.show()

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
            dif_stand = 100
            if abs(k) > 20: # fit K 过大
                dif_stand *= abs(k / 10)
            index_list = np.where(dif < dif_stand) # 找到满足的索引
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
        k,a1,b = slope,cx,cy
        x = bbox.T[4]
        y_true = bbox.T[5]
        y_pred = k * (x - a1) + b
        dif = abs(y_pred - y_true)  # 寻找每一个线性预测误差
        index_list = np.where(dif < 100)  # 找到满足的索引
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

        line1 = np.array([[0, 0], [1, 1]], np.int32).reshape((-1, 1, 2))
        imgnew = cv2.polylines(image_new, pts=[line1], isClosed=False, color=[0,0,0], thickness=2, lineType=cv2.LINE_8)


        return imgnew,bbox

if __name__ == '__main__':
    sort_bbox = Sort_bbox()
    sort_bbox.sort()
