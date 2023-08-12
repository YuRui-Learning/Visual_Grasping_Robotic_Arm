# import cv2
# import numpy as np
#
# #定义形状检测函数
# def ShapeDetection(img):
#     imgContour = img.copy()
#     imgGray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)  # 转灰度图
#     imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)  # 高斯模糊
#     img = cv2.Canny(img, 60, 60)  # Canny算子边缘检测
#     contours,hierarchy = cv2.findContours(imgContour,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)  #寻找轮廓点
#     for obj in contours:
#         area = cv2.contourArea(obj)  #计算轮廓内区域的面积
#         cv2.drawContours(img, obj, -1, (255, 0, 0), 4)  #绘制轮廓线
#         perimeter = cv2.arcLength(obj,True)  #计算轮廓周长
#         approx = cv2.approxPolyDP(obj,0.02*perimeter,True)  #获取轮廓角点坐标
#         CornerNum = len(approx)   #轮廓角点的数量
#         x, y, w, h = cv2.boundingRect(approx)  #获取坐标值和宽度、高度
#
#         #轮廓对象分类
#         if CornerNum ==3: objType ="triangle"
#         elif CornerNum == 4:
#             if w==h: objType= "Square"
#             else:objType="Rectangle"
#
#         cv2.rectangle(imgContour,(x,y),(x+w,y+h),(0,0,255),2)  #绘制边界框
#         cv2.putText(imgContour,objType,(x+(w//2),y+(h//2)),cv2.FONT_HERSHEY_COMPLEX,0.6,(0,0,0),1)  #绘制文字
#         cv2.imshow("blot picture", imgContour)
#         cv2.waitKey(0)
#
#
# import cv2
# import numpy as np
#
# #定义形状检测函数
# def ShapeDetection(img):
#     contours,hierarchy = cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)  #寻找轮廓点
#     for obj in contours:
#         area = cv2.contourArea(obj)  #计算轮廓内区域的面积
#         perimeter = cv2.arcLength(obj,True)  #计算轮廓周长
#         approx = cv2.approxPolyDP(obj,0.02*perimeter,True)  #获取轮廓角点坐标
#         CornerNum = len(approx)   #轮廓角点的数量
#         x, y, w, h = cv2.boundingRect(approx)  # 获取坐标值和宽度、高度
#         if CornerNum >= 4 and area >800 and perimeter >100:
#             objType="Rectangle"
#             cv2.rectangle(imgContour,(x,y),(x+w,y+h),(0,0,255),2)  #绘制边界框
#
# path = '../dataset/Image_20230731230658775.bmp'
# img = cv2.imread(path)
# imgContour = img.copy()
#
# imgGray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)  #转灰度图
# imgCanny = cv2.Canny(imgGray,60,60)  #Canny算子边缘检测
# ShapeDetection(imgCanny)  #形状检测
#
# cv2.imshow("Original img", img)
# cv2.imshow("imgCanny", imgCanny)
# cv2.imshow("shape Detection", imgContour)
#
# cv2.waitKey(0)
#

# -*- coding:utf-8 -*-
import cv2

# 读取图像
image = cv2.imread('../dataset/Image_20230731230658775.bmp')

# 将图像转化为灰度图
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 对图像进行二值化
ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

# 检测图像中的轮廓
contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

# 遍历所有轮廓
max_area = 0
for contour in contours:
  # 计算轮廓的面积
  area = cv2.contourArea(contour)

  # 如果面积更大，则更新最大面积
  if area > max_area:
    max_area = area
