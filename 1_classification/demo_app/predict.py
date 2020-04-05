from lib import * 
from config import *
from image_transform import *
from utils import *

class_index = ["ants", "bees"]
        
class Predictor():
    def __init__(self, class_index):
        self.class_index = class_index
    
    def predict_max(self, out):
        maxid = np.argmax(out.detach().numpy())
        predicted_label_name = self.class_index[maxid]
        
        return predicted_label_name

predictor = Predictor(class_index)

def predict(img):
    # prepare network
    use_pretrained = True
    net = models.vgg16(pretrained=use_pretrained)
    net.classifier[6] = nn.Linear(in_features=4096, out_features=2)

    net.eval()

    # prepare model
    model = load_model(net, save_path)

    # prepare input
    transform = ImageTransform(resize, mean, std)
    img = transform(img, phase="test")
    img = img.unsqueeze_(0)

    # predict
    out = model(img)
    response = predictor.predict_max(out)

    return response

