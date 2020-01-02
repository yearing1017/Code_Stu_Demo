import os 
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from onehot import onehot
import cv2
from test_data import test_dataloader

colors = [(0,0,0),(0,128,128),(128,0,128),(128,0,0),(128,128,128)]

def label2color(colors, n_classes, predict):
    for i in range(4):
        seg_color = np.zeros((predict.shape[1], predict.shape[2], 3))
        for c in range(n_classes):
            seg_color[:, :, 0] += ((predict[i,:,:] == c) *
                                (colors[c][0])).astype('uint8')
            seg_color[:, :, 1] += ((predict[i,:,:] == c) *
                                (colors[c][1])).astype('uint8')
            seg_color[:, :, 2] += ((predict[i,:,:] == c) *
                                (colors[c][2])).astype('uint8')
        seg_color = seg_color.astype(np.uint8)
    return seg_color


def predict():

    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    model = torch.load('checkpoints/deeplabv3_model_100.pt')

    model.eval()
    with torch.no_grad():
        for index, (image, label) in enumerate(test_dataloader):
            image = image.to(device)
            #label = label.to(device)
            predict = model(image) #(4,5,640,640)
            predict_index = torch.argmax(predict, dim=1, keepdim=False).numpy()  #(4, 640,640)




