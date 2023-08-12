
class Blot_bbox():

    def __init__(self):
        self.blot_bbox_dict = {
            0:[],
            1:[],
            2:[],

        }
    def insert(self,label,bbox):
        self.blot_bbox_dict[label].append(bbox)