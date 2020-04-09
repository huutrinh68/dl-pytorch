from lib import *
from make_datapath import make_datapath_list
from transform import DataTransform
from extract_inform_annotation import Anno_xml


class MyDataset(data.Dataset):
    def __init__(self, img_list, anno_list, phase, transform, anno_xml):
        self.img_list = img_list
        self.anno_list = anno_list
        self.phase = phase
        self.transform = transform
        self.anno_xml = anno_xml
    
    def __len__(self):
        return len(self.img_list)
    
    def __getitem__(self, index):
        img, gt, height, width = self.pull_item(index)

        return img, gt
    
    def pull_item(self, index):
        img_file_path = self.img_list[index]
        img = cv2.imread(img_file_path) #BGR
        height, width, channels = img.shape

        # get anno information
        anno_file_path = self.anno_list[index]
        ann_info = self.anno_xml(anno_file_path, width, height)

        # preprocessing
        img, boxes, labels = self.transform(img, self.phase, ann_info[:, :4], ann_info[:, 4])

        # BGR -> RGB, (height, width, channels) -> (channels, height, width)
        img = torch.from_numpy(img[:,:,(2,1,0)]).permute(2,0,1)

        # ground truth
        gt = gt = np.hstack((boxes, np.expand_dims(labels, axis=1)))

        return img, gt, height, width


if __name__ == "__main__":
    classes = ["aeroplane", "bicycle", "bird",  "boat", "bottle", 
               "bus", "car", "cat", "chair", "cow", "diningtable",
               "dog", "horse", "motorbike", "person", "pottedplant",
               "sheep", "sofa", "train", "tvmonitor"]

    # prepare train, valid, annotation list
    root_path = "./data/VOCdevkit/VOC2012/"
    train_img_list, train_annotation_list, val_img_list, val_annotation_list = make_datapath_list(root_path)

    # prepare data transform
    color_mean = (104, 117, 123)
    input_size = 300

    train_dataset = MyDataset(train_img_list, train_annotation_list, phase="train",
    transform=DataTransform(input_size, color_mean), anno_xml=Anno_xml(classes))

    # print(len(train_dataset))
    print(train_dataset.__getitem__(1))





