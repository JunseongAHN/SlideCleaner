import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import openslide

class QualityPatch():
    def __init__(self, original_img_path,label_img_path,patch_level,patch_size):
        """
        parameter:
            original_img_path(str): the source of image
            label_img_path(str): label image
            patch_level(int): the level that the patch belongs to
            patch_size(tuple): size of patch(x,y)

        attributes:
            self.slide(Openslide): the slide that the patch belongs to 
            self.original_img_path(str) : the path of the lide
            self.label_img_path(str) : label_img_path
            self.patch_level(int) : the level that the patch belongs to
            self.patch_size = patch_size

            self.scale(int) : the magnification of the slide that the patch belongs to with level_max baseline
            self.label(np array) : the image of label
            self.label_size(tuple) : the size of label
            self.adj_patch_size_label(tuple) : considering the slide is rescaled to self.label_size the size is zero, it is 1
        """
        self.slide = openslide.OpenSlide(original_img_path)
        slide_width, slide_height = self.slide.dimensions
        self.label = (cv2.imread(label_img_path,cv2.IMREAD_GRAYSCALE)/255)
        self.patch_coors = [(w,h) for w in range(0, slide_width - patch_size[0], patch_size[0]) for h in range(0, slide_height - patch_size[1],patch_size[1])]

        self.original_img_path = original_img_path
        self.label_img_path = label_img_path
        self.patch_level = patch_level
        self.patch_size = patch_size
        self.label = self.label.T
        self.level_dim = self.slide.level_dimensions[patch_level]

        self.label_size = self.label.shape
        self.scale = (self.label_size[0]/self.level_dim[0], self.label_size[1]/self.level_dim[1])
        self.adj_patch_size_label = self.calculateAdjPatchSize()

    def calculateLabelCoordinates(self, patch_location):
        return (int(self.scale[0]*patch_location[0]/2**(self.patch_level)), int(self.scale[1]*patch_location[1]/2**(self.patch_level)))
  
    def calculateAdjPatchSize(self):
        return (int(self.scale[0] * self.patch_size[0])+1, int(self.scale[1] * self.patch_size[1])+1)

    def patchQualityInsurance(self, patch_location):
        label_coordinates = self.calculateLabelCoordinates(patch_location)
        percent = (np.sum(self.label[label_coordinates[0]:label_coordinates[0]+self.adj_patch_size_label[0],label_coordinates[1]:label_coordinates[1]+self.adj_patch_size_label[1]]))/(self.adj_patch_size_label[0]*self.adj_patch_size_label[1])

        return percent

    def getLabelWithPatchLocation(self, patch_location):
        patch_image = np.ones(self.adj_patch_size_label)/2
        label_with_patch_location = self.label.copy()
        label_coordinates = self.calculateLabelCoordinates(patch_location)
        label_with_patch_location[label_coordinates[0]:label_coordinates[0]+self.adj_patch_size_label[0],label_coordinates[1]:label_coordinates[1]+self.adj_patch_size_label[1]] = patch_image
        return label_with_patch_location.T
    
    def getReleventPatches(self):
        relevent_patches = []


        for i, coor in enumerate(self.patch_coors):
            percent = self.patchQualityInsurance(coor)
            if percent > .5:
                relevent_patches.append([coor,percent])
            if i % 10000 == 0:
                print(i, "/",len(self.patch_coors), "dic len", len(relevent_patches), " from", len(self.patch_coors) )
        return relevent_patches

    def checkingfunction(self, checking_coors=(40000,90000)):
        if checking_coors[0] < 0 or checking_coors[0] < 0 or\
            self.slide.level_dimensions[self.patch_level][0] < (checking_coors[0] / 2**(self.patch_level) + self.patch_size[0]) or\
            self.slide.level_dimensions[self.patch_level][1] < ((checking_coors[1] / 2**(self.patch_level) + self.patch_size[1])):
            raise ValueError("the patch location with patch size is not valid.")
        
        image = self.slide.read_region(checking_coors, self.patch_level, self.patch_size)
        percent = self.patchQualityInsurance(checking_coors)

        fig, ax = plt.subplots(nrows=1, ncols=3)
        plt.tight_layout()
        ax[0].set_title("tissue percentage %.02f"%percent)
        ax[0].axis('off')
        ax[0].imshow(image)
        ax[1].set_title("tissue label")
        ax[1].axis('off')
        ax[1].imshow(self.label.T, cmap='gray')
        ax[2].set_title("label with patch")
        ax[2].axis('off')
        ax[2].imshow(self.getLabelWithPatchLocation(checking_coors))
        plt.savefig("test/check_read_region"+str(self.patch_level)+'.png')
        plt.close('all')
