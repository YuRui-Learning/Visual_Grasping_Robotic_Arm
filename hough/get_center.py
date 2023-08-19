import cv2
import numpy as np

def hough_center(img):
    '''
    only to get center
    :param img: picture
    :return: cx,cy
    '''

    '''change picture'''
    GrayImage=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    GrayImage= cv2.medianBlur(GrayImage,5)
    th2 = cv2.adaptiveThreshold(GrayImage,255,cv2.ADAPTIVE_THRESH_MEAN_C,
                        cv2.THRESH_BINARY,3,5)
    '''kernel function'''
    kernel = np.ones((5,5),np.uint8)
    erosion = cv2.erode(th2,kernel,iterations=1)
    '''deal picture'''
    imgray=cv2.Canny(erosion,30,100)
    circles = cv2.HoughCircles(imgray, cv2.HOUGH_GRADIENT, 1, 20,
                             param1=50, param2=50, minRadius=20, maxRadius=100)

    '''get result and show'''
    cx = 0
    cy = 0
    show_flag = False
    if circles is not None:
        circles = np.uint16(np.around(circles))
        cx,cy = circles[0][0][0],circles[0][0][1]
        if show_flag:
            cv2.circle(img, (cx, cy), circles[0][0][2], (255, 0, 0), 2)
            # draw the center of the circle
            cv2.circle(img, (cx, cy), 2, (0, 0, 255), 3)
            cv2.imshow('detected circles', img)
            cv2.waitKey(0)
            cv2.imshow('detected circles', img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        print(len(circles[0, :]))


    return cx,cy


