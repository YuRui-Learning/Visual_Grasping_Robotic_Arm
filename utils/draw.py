import cv2
import numpy as np

def draw_cross(image, center, color, cross_radius=5):
    """
    在传入的图像中绘制十字架
    :param image: 传入的需要绘制十字架的图片
    :param cross_radius: 需要绘制的十字架的半径
    :return: 返回绘制好十字架的图像
    """
    center_x = center[0]  # 图像的中心点x坐标
    center_y = center[1]
    cross_x1 = center_x - cross_radius  # 十字的左顶点
    cross_x2 = center_x + cross_radius
    cross_y1 = center_y - cross_radius  # 十字的上顶点
    cross_y2 = center_y + cross_radius
    # 使用cv2.polylines()画多条直线也可以用来画多边形
    line1 = np.array([[cross_x1, center_y], [cross_x2, center_y]], np.int32).reshape((-1, 1, 2))
    line2 = np.array([[center_x, cross_y1], [center_x, cross_y2]], np.int32).reshape((-1, 1, 2))
    cv2.polylines(image, pts=[line1, line2], isClosed=False, color=color , thickness=2, lineType=cv2.LINE_8)
    return image
