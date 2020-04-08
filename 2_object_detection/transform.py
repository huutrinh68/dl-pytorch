from utils.augmentation import Compose, ConvertFromInts, ToAbsoluteCoords, \
    PhotometricDistort, Expand, RandomSampleCrop, RandomMirror, \
    ToPercentCoords, Resize, SubtractMeans

from make_datapath import make_datapath_list
from extract_inform_annotation import Anno_xml
from lib import *

class DataTransform():
# class DataTransform:
# class DataTransform(object):
    def __init__(self, input_size, color_mean):
        self.data_transform = {
            "train": Compose([
                ConvertFromInts(), # convert image from int to float 32
                ToAbsoluteCoords(), # back annotation to normal type
                PhotometricDistort(), # change color by random
                Expand(color_mean), 
                RandomSampleCrop(), # randomcrop image
                RandomMirror(), # xoay ảnh ngược lại
                ToPercentCoords(), # chuẩn hoá annotation data về dạng [0-1]
                Resize(input_size),
                SubtractMeans(color_mean) # Subtract mean của BGR
            ]), 
            "val": Compose([
                ConvertFromInts(), # convert image from int to float 32
                Resize(input_size),
                SubtractMeans(color_mean)
            ])
        }

    def __call__(self, img, phase, boxes, labels):
        return self.data_transform[phase](img, boxes, labels)

if __name__ == "__main__":
    classes = ["aeroplane", "bicycle", "bird",  "boat", "bottle", 
               "bus", "car", "cat", "chair", "cow", "diningtable",
               "dog", "horse", "motorbike", "person", "pottedplant",
               "sheep", "sofa", "train", "tvmonitor"]

    # prepare train, valid, annotation list
    root_path = "./data/VOCdevkit/VOC2012/"
    train_img_list, train_annotation_list, val_img_list, val_annotation_list = make_datapath_list(root_path)

    # read img
    img_file_path = train_img_list[0]
    img = cv2.imread(img_file_path) # Height, Width, Channel(BGR)
    height, width, channels = img.shape

    # annotation information
    trans_anno = Anno_xml(classes)
    anno_info_list = trans_anno(train_annotation_list[0], width, height)

    # plot original image
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) # mặc định của matplotlib là RGB
    plt.show()

    # prepare data transform
    color_mean = (104, 117, 123)
    input_size = 300
    transform = DataTransform(input_size, color_mean)

    # transform train img
    phase = "train"
    img_transformed, boxes, labels = transform(img, phase, anno_info_list[:,:4], anno_info_list[:, 4])
    plt.imshow(cv2.cvtColor(img_transformed, cv2.COLOR_BGR2RGB)) # mặc định của matplotlib là RGB
    plt.show()

    # transform val img
    phase = "val"
    img_transformed, boxes, labels = transform(img, phase, anno_info_list[:,:4], anno_info_list[:, 4])
    plt.imshow(cv2.cvtColor(img_transformed, cv2.COLOR_BGR2RGB)) # mặc định của matplotlib là RGB
    plt.show()











