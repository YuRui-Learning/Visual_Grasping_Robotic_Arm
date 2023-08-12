import numpy as np
import random
import cv2

class Blot_class():
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
