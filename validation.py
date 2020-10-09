import numpy as np
import cv2
from pytorch.yolo_models.utils_yolo import *
from pytorch.yolo_models.darknet import Darknet
import torch

def read_image(path):
    image = cv2.imread(path)
    return image

def predict_convert(image_var, model, class_names, reverse=False):
    pred, _ = model(image_var)
    boxes = []
    img_vis = []
    pred_vis = []
    vis = []
    i = 0
    boxes.append(nms(pred[0][i] + pred[1][i] + pred[2][i], 0.4))
    img_vis.append((image_var[i].cpu().data.numpy().transpose(1, 2, 0) * 255).astype(np.uint8))

    pred_vis.append(plot_boxes(Image.fromarray(img_vis[i]), boxes[i], class_names=class_names))
    vis = np.array(pred_vis[i][0])
    return np.array(vis), np.array(boxes)

def visual(path):
    namesfile = './pytorch/yolo_models/data_yolo/coco.names'
    class_names = load_class_names(namesfile)
    single_model = Darknet('./pytorch/yolo_models/cfg/yolov3.cfg')
    single_model.load_weights('/home/ningfei/Documents/meshadv/data/yolov3.weights')
    model = single_model
    model = model.cuda()
    model.eval()

    background = cv2.imread(path)
    background = cv2.resize(background, (single_model.height, single_model.width))
    background = background[:, :, ::-1] / 255.0
    background = background.astype(np.float32)

    background_tensor = torch.from_numpy(background.transpose(2, 0, 1)).cuda()
    final_image = torch.clamp(background_tensor, 0, 1).unsqueeze(0)
    vis, _ = predict_convert(final_image, model, class_names)

    return vis

if __name__ == '__main__':
    path = './adv_truck.png'
    vis_ = visual(path)
    new_im = Image.fromarray(vis_)
    new_im.save("result.png")

