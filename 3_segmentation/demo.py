import cv2 
from model import PSPNet
import torch 
from data import make_datapath_list, DataTransform
from PIL import Image
import numpy as np


# load trained model
net = PSPNet(n_classes=21)
state_dict = torch.load('./pspnet50_30.pth', map_location={'cuda:0': 'cuda:0'})
net.load_state_dict(state_dict)
net.eval()

# 1500 images for training
# 1500 images for validation
rootpath = "./data/VOCdevkit/VOC2012/"
train_img_list, train_anno_list, val_img_list, val_anno_list = make_datapath_list(rootpath)
color_mean = (0.485, 0.456, 0.406)
color_std = (0.229, 0.224, 0.225)

anno_file_path = val_anno_list[1]
anno_class_img = Image.open(anno_file_path)
p_palatte = anno_class_img.getpalette()

transform = DataTransform(input_size=475, color_mean=color_mean, color_std=color_std)

cap = cv2.VideoCapture(0)
img_width, img_height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
phase = 'val'

while(True):
    ret, img = cap.read()
    img = Image.fromarray(img)
    origin = img.copy()

    img, anno_class_img = transform(phase, img, anno_class_img)
    x = img.unsqueeze(0)
    outputs = net(x)
    y = outputs[0]

    y = y[0].detach().numpy()
    y = np.argmax(y, axis=0)

    anno_class_img = Image.fromarray(np.uint8(y), mode='P')
    anno_class_img = anno_class_img.resize((img_width, img_height), Image.NEAREST)
    anno_class_img.putpalette(p_palatte)

    trans_img = Image.new('RGBA', anno_class_img.size, (0, 0, 0, 0))
    anno_class_img = anno_class_img.convert('RGBA')

    for x in range(img_width):
        for y in range(img_height):
            pixel = anno_class_img.getpixel((x, y))
            r, g, b, a = pixel 
            if r==0 and g == 0 and b == 0:
                continue
            else:
                trans_img.putpixel((x, y), (r, g, b, 150))
    
    result = Image.alpha_composite(origin.convert('RGBA'), trans_img)

    # convert pil -> cv2
    img = np.array(result, dtype=np.uint8)

    # display the result frame
    cv2.imshow('PSPNet50', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break 

cap.release()
cv2.destroyAllWindows()


