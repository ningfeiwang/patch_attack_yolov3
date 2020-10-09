import numpy as np
import cv2
from pytorch.yolo_models.utils_yolo import *
from pytorch.yolo_models.darknet import Darknet
import torch
from PIL import Image

from torch.optim import Adam
import tqdm
import torch.nn.functional as F
import random



def read_image(path):
    image = cv2.imread(path)
    return image


def predict_convert(image_var, model, class_names, reverse=False):
    # print(image_var.shape)
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

def tensor_size(t):
    return t.size()[1]*t.size()[2]*t.size()[3]


def resize2d(img, size):
    return (F.adaptive_avg_pool2d(img, size))

def get_image(path, x, y, size):
    adv_stopsign = Image.open(path).convert('RGB')
    adv_stopsign = np.asarray(adv_stopsign)
    patch_x = size
    patch_y = size
    adv_img = adv_stopsign.copy()
    adv_img[x:x + patch_x, y: y + patch_y, :] = 255
    return adv_img


def attack(path):
    namesfile = './pytorch/yolo_models/data_yolo/coco.names'
    class_names = load_class_names(namesfile)
    single_model = Darknet('./pytorch/yolo_models/cfg/yolov3.cfg')
    single_model.load_weights('/home/ningfei/Documents/meshadv/data/yolov3.weights')
    model = single_model
    model = model.cuda()
    model.eval()

    adv_stopsign = Image.open(path).convert('RGB')
    adv_stopsign = np.asarray(adv_stopsign)
    patch_x = 315
    patch_y = 315
    adv_img = adv_stopsign.copy()

    bx = adv_img[248:248 + patch_x, 812: 812 + patch_y, :].copy()
    bx = bx / 255.0
    bx = bx.astype(np.float32)
    bx = torch.from_numpy(bx.transpose(2, 0, 1)).cuda()
    bx = torch.clamp(bx, 0, 1).unsqueeze(0)
    adv_img = adv_stopsign.copy() / 255.0
    adv_img = adv_img.astype(np.float32)
    background_tensor = torch.from_numpy(adv_img.transpose(2, 0, 1).copy()).cuda()
    final_image_ = torch.clamp(background_tensor, 0, 1).unsqueeze(0)
    learning_rate = 0.005
    bx = Variable(bx.data, requires_grad=True)
    opt = Adam([bx], lr=learning_rate, amsgrad=True)
    num_class = 80
    threshold = 0.1
    batch_size = 1
    best_it = 0
    best_loss = 1000000.0

    for i in tqdm.tqdm(range(50)):

        final_image = final_image_ + (0.001) * torch.randn(final_image_.shape).cuda()

        final_image[:, :, 248:248 + patch_x, 812: 812 + patch_y] = bx[:,:,:,:]

        final_image = resize2d(final_image, (416, 416))
        final_image = torch.clamp(final_image[0], 0, 1)[None]
        final, outputs = model(final_image)
        adv_total_loss = None
        num_pred = 0.0
        removed = 0.0
        for index, out in enumerate(outputs):
            num_anchor = out.shape[1] // (num_class + 5)
            out = out.view(batch_size * num_anchor, num_class + 5, out.shape[2], out.shape[3])
            cfs = torch.nn.functional.sigmoid(out[:, 4])
            mask = (cfs >= threshold).type(torch.cuda.FloatTensor)
            num_pred += torch.numel(cfs)

            removed += torch.sum((cfs < threshold).type(torch.FloatTensor)).data.cpu().numpy()

            loss = 2.0 * torch.sum((cfs - 0) ** 2)

            h_x = bx.size()[2]
            w_x = bx.size()[3]
            count_h = tensor_size(bx[:, :, 1:, :])
            count_w = tensor_size(bx[:, :, :, 1:])
            h_tv = torch.pow((bx[:, :, 1:, :] - bx[:, :, :h_x - 1, :]), 2).sum()
            w_tv = torch.pow((bx[:, :, :, 1:] - bx[:, :, :, :w_x - 1]), 2).sum()
            tv_loss = 4.0 * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

            if adv_total_loss is None:
                adv_total_loss = loss
            else:
                adv_total_loss += loss
            adv_total_loss += tv_loss


        total_loss = torch.clamp(adv_total_loss, min=0)
        print('total loss: ', total_loss)
        opt.zero_grad()
        total_loss.backward(retain_graph=True)
        opt.step()
        if adv_total_loss < best_loss:
            best_loss = adv_total_loss
            best_bx = bx.data.cpu().numpy()
            best_final_img = final_image.data.cpu().numpy()

    return best_bx,best_final_img, model, class_names

if __name__ == '__main__':
    path = './truck_sample.png'
    best_bx, best_final, model, class_names = attack(path)

    np.savez('./truck_sample.npz', bx=best_bx, img = best_final)

    vis_, _ = predict_convert(torch.tensor(best_final).cuda(), model, class_names)
    adv_stopsign = Image.open(path).convert('RGB')
    adv_stopsign = np.asarray(adv_stopsign)
    patch_x = 315
    patch_y = 315
    adv_img = adv_stopsign.copy()
    adv_img[248:248 + patch_x, 812: 812 + patch_y, :] = (best_bx[0] * 255).transpose(1,2,0).astype(np.uint8)
    new_im = Image.fromarray(adv_img)
    new_im.save("adv_truck.png")
    new_im = Image.fromarray( (best_bx[0] * 255).transpose(1, 2, 0).astype(np.uint8))
    new_im.save("patch.png")