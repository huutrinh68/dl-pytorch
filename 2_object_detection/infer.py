from lib import *
from model import SSD
from transform import DataTransform
from show_result import SSDPredictShow


voc_classes = ['aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor']


# network
cfg = {
    "num_classes": 21, #VOC data include 20 class + 1 background class
    "input_size": 300, #SSD300
    "bbox_aspect_num": [4, 6, 6, 6, 4, 4], # Tỷ lệ khung hình cho source1->source6`
    "feature_maps": [38, 19, 10, 5, 3, 1],
    "steps": [8, 16, 32, 64, 100, 300], # Size of default box
    "min_size": [30, 60, 111, 162, 213, 264], # Size of default box
    "max_size": [60, 111, 162, 213, 264, 315], # Size of default box
    "aspect_ratios": [[2], [2,3], [2,3], [2,3], [2], [2]]
}

net = SSD(phase="inference", cfg=cfg)
# SSD の学習済みの重みを設定
net_weights = torch.load('./data/weights/ssd300_50.pth', map_location={'cuda:0': 'cpu'})

net.load_state_dict(net_weights)


image_file_path = "./data/cowboy-757575_640.jpg"
img = cv2.imread(image_file_path) # [ 高さ ][ 幅 ][ 色 BGR]
height, width, channels = img.shape # 画像のサイズを取得
# 2. 元画像の表示
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show()
# 3. 前処理クラスの作成
color_mean = (104, 117, 123) # (BGR) の色の平均値
input_size = 300 # 画像の input サイズを 300 300 にする
transform = DataTransform(input_size, color_mean)

phase = "val"
img_transformed, boxes, labels = transform(img, phase, "", "") # アノテーションはないので、"" にする
img_tensor = torch.from_numpy(img_transformed[:, :, (2, 1, 0)]).permute(2, 0, 1)

net.eval() # ネットワークを推論モードへ
x = img_tensor.unsqueeze(0) # ミニバッチ化:torch.Size([1, 3, 300, 300])
detection = net(x)
# print(detections.shape)
# print(detections)


# ssd = SSDPredictShow(eval_categories=voc_classes, net=net)
# ssd.show(image_file_path, data_confidence_level=0.6)
top_k=10

plt.figure(figsize=(10,10))
colors = plt.cm.hsv(np.linspace(0, 1, 21)).tolist()
plt.imshow(img)  # plot the image for matplotlib
currentAxis = plt.gca()

detections = detection.data
# scale each detection back up to the image
scale = torch.Tensor(img.shape[1::-1]).repeat(2)
for i in range(detections.size(1)):
    j = 0
    while detections[0,i,j,0] >= 0.01:
        score = detections[0,i,j,0]
        label_name = voc_classes[i-1]
        display_txt = '%s: %.2f'%(label_name, score)
        pt = (detections[0,i,j,1:]*scale).cpu().numpy()
        coords = (pt[0], pt[1]), pt[2]-pt[0]+1, pt[3]-pt[1]+1
        color = colors[i]
        currentAxis.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=color, linewidth=2))
        currentAxis.text(pt[0], pt[1], display_txt, bbox={'facecolor':color, 'alpha':0.5})
        j+=1