import os.path as osp
import torch.utils.data as data
from PIL import Image
from utils.augmentation import Compose, Scale, RandomRotation, RandomMirror, Resize, Normalize_Tensor
import matplotlib.pyplot as plt

def make_datapath_list(rootpath):
    original_image_template = osp.join(rootpath, 'JPEGImages', '%s.jpg')
    annotation_image_template = osp.join(rootpath, 'SegmentationClass', '%s.png')

    #train, val
    train_ids = osp.join(rootpath, 'ImageSets/Segmentation/train.txt')
    val_ids = osp.join(rootpath, 'ImageSets/Segmentation/val.txt')

    train_img_list = list()
    train_anno_list = list()

    for line in open(train_ids):
        img_id = line.strip()
        img_path = (original_image_template % img_id)
        anno_path = (annotation_image_template % img_id)

        train_img_list.append(img_path)
        train_anno_list.append(anno_path)

    
    val_img_list = list()
    val_anno_list = list()

    for line in open(val_ids):
        img_id = line.strip()
        img_path = (original_image_template % img_id)
        anno_path = (annotation_image_template % img_id)

        val_img_list.append(img_path)
        val_anno_list.append(anno_path)


    return train_img_list, train_anno_list, val_img_list, val_anno_list


class DataTransform():
    def __init__(self, input_size, color_mean, color_std):
        self.data_transform = {
            "train": Compose([
                Scale(scale=[0.5, 1.5]),
                RandomRotation(angle=[-10, 10]),
                RandomMirror(),
                Resize(input_size),
                Normalize_Tensor(color_mean, color_std)
                ]),
            "val": Compose([
                Resize(input_size),
                Normalize_Tensor(color_mean, color_std)
            ])
        }
    
    def __call__(self, phase, img, anno_class_img):
        return self.data_transform[phase](img, anno_class_img)
        

class MyDataset(data.Dataset):
    def __init__(self, img_list, anno_list, phase, transform):
        self.img_list = img_list
        self.anno_list = anno_list
        self.phase = phase
        self.transform = transform

    def __len__(self):
        return len(self.img_list)
    
    def __getitem__(self, index):
        img, anno_class_img = self.pull_item(index)
        return img, anno_class_img
    
    def pull_item(self, index):
        # original image
        img_file_path = self.img_list[index]
        img = Image.open(img_file_path)

        # annotation image
        anno_file_path = self.anno_list[index]
        anno_class_img = Image.open(anno_file_path) #PIL -> (height, width, channel(RGB))

        img, anno_class_img = self.transform(self.phase, img, anno_class_img)

        return img, anno_class_img



if __name__ == "__main__":
    rootpath = "./data/VOCdevkit/VOC2012/"
    train_img_list, train_anno_list, val_img_list, val_anno_list = make_datapath_list(rootpath)

    # print(len(train_img_list))
    # print(len(val_img_list))

    # print(train_img_list[0]) #path of original image (RGB)
    # print(train_anno_list[0]) #path of segmentation image (Color pallet)
    
    color_mean = (0.485, 0.456, 0.406)
    color_std = (0.229, 0.224, 0.225)

    train_dataset = MyDataset(train_img_list, train_anno_list, phase="train", transform=DataTransform(input_size=475, color_mean=color_mean, color_std=color_std))
    val_dataset = MyDataset(val_img_list, val_anno_list, phase="val", transform=DataTransform(input_size=475, color_mean=color_mean, color_std=color_std))

    # print("val_dataset_img: {}".format(val_dataset.__getitem__(0)[0].shape))
    # print("val_dataset_anno_class_img: {}".format(val_dataset.__getitem__(0)[1].shape))
    # print("val_dataset: {}".format(val_dataset.__getitem__(0)))

    batch_size = 4
    train_dataloader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    dataloader_dict = {
        "train": train_dataloader,
        "val": val_dataloader
    }

    batch_iterator = iter(dataloader_dict["train"])

    images, anno_class_images = next(batch_iterator)
    # print(images.size())
    # print(anno_class_images.size())

    image = images[0].numpy().transpose(1,2,0) #(chanel(RGB), height, witdh) => (height, width, channel(RGB))
    plt.imshow(image)
    plt.show()

    anno_class_image = anno_class_images[0].numpy()
    plt.imshow(anno_class_image)
    plt.show()

