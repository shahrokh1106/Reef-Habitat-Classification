import cv2
import torch
import os

WEIGHTS_PATH = "Urchin-Detector/models/yolov5m_helio/weights/best.pt"
NUM_TO_LABEL = ["Evechinus chloroticus","Centrostephanus rodgersii", "Heliocidaris erythrogramma"]
LABEL_TO_NUM = {label: i for i, label in enumerate(NUM_TO_LABEL)}
NUM_TO_COLOUR = [(74,237,226), (24,24,204), (3,140,252)]
yolo5_path = "Urchin-Detector"


def load_model(weights_path=WEIGHTS_PATH, cuda=True):
    """Load and return a yolo model"""
    model = torch.hub.load(os.path.join(yolo5_path, "yolov5"), "custom", path=weights_path, source="local")
    model.cuda() if cuda else model.cpu()
    return model

def plat_scaling(x):
    #Platt scaling function for highres-ro v3 model
    cubic = -7.3848* x**3 +13.5284 * x**2 -6.2952 *x + 1.0895
    linear = 0.566 * x + 0.027

    return cubic if x >=0.45 else linear
class UrchinDetector_YoloV5:
    """Wrapper class for the yolov5 model"""
    def __init__(self, weight_path=WEIGHTS_PATH, conf=0.45, iou=0.6, img_size=1280, cuda=None, classes=NUM_TO_LABEL, plat_scaling = False):
        self.weight_path = weight_path
        self.conf = conf
        self.iou = iou
        self.img_size = img_size
        self.cuda = cuda if not (cuda is None) else torch.cuda.is_available()
        self.scaling = plat_scaling

        self.model = load_model(self.weight_path, self.cuda)
        self.model.conf = self.conf
        self.model.iou = self.iou
        
        self.classes = classes

    def update_parameters(self, conf=0.45, iou=0.6):
        self.conf = conf
        self.model.conf = conf
        self.iou = iou
        self.model.iou = iou

    def predict(self, im):
        results = self.model(im, size = self.img_size)
        if self.scaling:
            with torch.inference_mode():
                for pred in results.pred[0]:
                    pred[4] = plat_scaling(pred[4])
            results.__init__(results.ims, pred=results.pred, files=results.files, times=results.times, names=results.names, shape=results.s)
        return results

    def __call__(self, im):
        return self.xywhcl(im)
    
    def xywhcl(self, im):
        pred = self.predict(im).xywh[0].cpu().numpy()
        return [box for box in pred]
    
def annotate_image(im, prediction, num_to_label, num_to_colour, draw_labels=True):
        """Draws xywhcl boxes onto a single image. Colours are BGR"""
        thickness = 2
        font_size = 0.75

        label_data = []
        for pred in prediction:
            top_left = (int(pred[0]) - int(pred[2])//2, int(pred[1]) - int(pred[3])//2)
            bottom_right = (top_left[0] + int(pred[2]), top_left[1] + int(pred[3]))
            label = num_to_label[int(pred[5])]
            label = f"{label[0]}. {label.split()[1]}"

            colour = num_to_colour[int(pred[5])]

            #Draw boudning box
            im = cv2.rectangle(im, top_left, bottom_right, colour, thickness)

            label_data.append((f"{label} - {float(pred[4]):.2f}", top_left, colour))

        #Draw text over boxes
        if draw_labels:
            for data in label_data:
                text_size = cv2.getTextSize(data[0], cv2.FONT_HERSHEY_SIMPLEX, font_size, thickness)[0]
                text_box_top_left = (data[1][0], data[1][1] - text_size[1])
                text_box_bottom_right = (data[1][0] + text_size[0], data[1][1])
                im = cv2.rectangle(im, text_box_top_left, text_box_bottom_right, data[2], -1)
                im = cv2.putText(im, data[0], data[1], cv2.FONT_HERSHEY_SIMPLEX, font_size, (0, 0, 0), thickness - 1, cv2.LINE_AA)
        return im

def GetYoloModel():
    model = UrchinDetector_YoloV5()
    return model
