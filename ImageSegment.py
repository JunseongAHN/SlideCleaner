import os
import numpy as np
import matplotlib.pyplot as plt
import imageio
import openslide
import cv2
from PIL import Image
from matplotlib import cm

class SegmentImage():
    def __init__(self, image_path, segment_level):
        
        self.image_path = image_path
        self.whole_slide = openslide.OpenSlide(self.image_path)
        #make sure that it is negative level
        lowest_dim = self.whole_slide.level_dimensions[-segment_level]
        lowest_level = self.whole_slide.level_count -segment_level
        # print(lowest_level, lowest_dim)
        whole_slide_png = self.whole_slide.read_region((0,0), lowest_level, lowest_dim)
        self.img = np.array(whole_slide_png)
        self.img = cv2.GaussianBlur(self.img, (5, 5), 0)
        # asdf
        #check the image is valid
        # try:
        #     print('Original size',self.img.shape)
        # except AttributeError:
        #     print("shape not found")
        self.segment = self.connectedBackgroundImage()
        self.mask =  self.segment.astype(bool)

    def segmentImage(self):
        #---------------------------------
        # Segmentation
        gray = cv2.cvtColor(self.img, cv2.COLOR_RGB2GRAY)
        ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        return thresh
    
    def connectedBackgroundImage(self):
        """
        Connects the background and object
        """
        thresh = self.segmentImage()
        #output = cv2.connectedComponentsWithStats(thresh, 4, cv2.CV_32S)

        kernel = np.ones((8, 8), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

        # sure background area
        sure_bg = cv2.dilate(opening, kernel, iterations=3)
        # Finding sure foreground area
        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

        # Finding unknown region
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg, sure_fg)
        return sure_bg
    
    def save_mask(self, file_name):
        im = Image.fromarray((self.segment))
        im.save(file_name)



if __name__ == "__main__":
    processed_img = SegmentImage("../../data/training/normal/normal_002.tif", 4)
    processed_img.save_mask("checking.png")