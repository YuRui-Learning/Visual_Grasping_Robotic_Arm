import numpy as np
import random
import cv2

class Blot_class():
    '''
    this function is to store the mean and update the result
    param: weight height label
    '''
    def __init__(self):
        self.dictonry = {
            0:{
                'width_mean':0,
                'height_mean':0,
                'number':0
            },
            1: {
                'width_mean': 0,
                'height_mean': 0,
                'number': 0
            },
            2: {
                'width_mean': 0,
                'height_mean': 0,
                'number': 0
            }
        }

    def update(self,wight,height,label):
        self.dictonry[label]['number'] += 1
        n = self.dictonry[label]['number']
        if n != 1:
            self.dictonry[label]['width_mean'] = self.dictonry[label]['width_mean']*(n - 1) // n + wight//n
            self.dictonry[label]['height_mean'] = self.dictonry[label]['height_mean']* (n - 1) // n + height // n
        else:
            self.dictonry[label]['width_mean'] = wight
            self.dictonry[label]['height_mean'] = height

    def prior(self):
        self.dictonry[0]['width_mean'] = 99
        self.dictonry[0]['height_mean'] = 108
        self.dictonry[0]['number'] = 1
        self.dictonry[1]['width_mean'] = 86
        self.dictonry[1]['height_mean'] = 95
        self.dictonry[1]['number'] = 1
        self.dictonry[2]['width_mean'] = 70
        self.dictonry[2]['height_mean'] = 78
        self.dictonry[2]['number'] = 1