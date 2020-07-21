import os
from ImageSegment import SegmentImage
from QualityInsurance import QualityPatch
import json

slide_paths = ["../../data/training/normal/", "../../data/training/tumor/"]
label_path = "result/label/"
cleaned_patch_path = "result/cleaned_patch/"
cleaned_patch_file_name = ['normal', 'tumor']

label_level = 4
patch_size = (225, 225)
patch_level = 0


def main():
    cleaned_patch = [{}] * len(slide_paths)

    for i, slide_path in enumerate(slide_paths):
        for (root, dirs, files) in os.walk(slide_path, topdown=True):
            files.sort()
            for myfile in files:
                myslide_path = slide_path + myfile
                label_image_path = label_path + myfile
                SegmentImage(myslide_path, label_level).save_mask(label_image_path)
                qp = QualityPatch(myslide_path, label_image_path, patch_level, patch_size)
                relevent_patch = qp.getReleventPatches()
                cleaned_patch[i][myfile] = relevent_patch

        with open(cleaned_patch_path + "cleaned_" + cleaned_patch_file_name[i] + '.json', 'w') as fp:
            json.dump(cleaned_patch[i], fp)
if __name__ == "__main__":
    main()
