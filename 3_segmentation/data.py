import os.path as osp

def make_datapath_list(rootpath):
    original_image_template = osp.join(rootpath, 'JPEGImages', '%s.jpg')
    annotation_image_template = osp.join(rootpath, 'SegmetationClass', '%s.png')

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


if __name__ == "__main__":
    rootpath = "./data/VOCdevkit/VOC2012/"
    train_img_list, train_anno_list, val_img_list, val_anno_list = make_datapath_list(rootpath)

    print(len(train_img_list))
    print(len(val_img_list))

    print(train_img_list[0]) #path of original image (RGB)
    print(train_anno_list[0]) #path of segmentation image (Color pallet)





