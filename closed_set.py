#!/usr/bin/env python3

#################################################################################################
# Author: Seungjoon Rhie                                                                        #
# Purpose: To detect people in images, crop them out, and sort                                  #
# Reference: 1) https://www.learnopencv.com/faster-r-cnn-object-detection-with-pytorch/         #
#            2) https://www.learnopencv.com/faster-r-cnn-object-detection-with-pytorch/         #
# Usage: python detection.py 'data-path-to-images' 'num-of-ids'                                 #
# REQUIRED: PyTorch running on GPU                                                              #
#################################################################################################

import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision
from torchvision import transforms as T
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import cv2
import os
import argparse
import random
import json
from net import *

### Get args
parser = argparse.ArgumentParser()
parser.add_argument('data_path', help='Path to images')
parser.add_argument('num_id', help='Number of ids generating')
args = parser.parse_args()


### Load the pretrained model for detection
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.cuda()
model.eval()

COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A']


if os.path.exists("closed_set/gallery") == False:
    os.mkdir("closed_set/gallery")

if os.path.exists("closed_set/temp") == False:
    os.mkdir("closed_set/temp")

if os.path.exists("closed_set/result") == False:
    os.mkdir("closed_set/result")

### Load the pretrained model for cosine similarity
model2 = models.resnet18(pretrained=True)
model2.cuda()
layer = model2._modules.get('avgpool')
model2.eval()

scaler = transforms.Resize((224, 224))
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
to_tensor = transforms.ToTensor()

cos = nn.CosineSimilarity(dim=1, eps=1e-6)


### Load the model for attribute extractor


model_dict = {
    'resnet18'  :  ResNet18_nFC,
    'resnet34'  :  ResNet34_nFC,
    'resnet50'  :  ResNet50_nFC,
    'densenet'  :  DenseNet121_nFC,
    'resnet50_softmax'  :  ResNet50_nFC_softmax,
}
num_cls_dict = { 'market':30, 'duke':23 }
num_ids_dict = { 'market':751, 'duke':702 }


def load_network(network):
    save_path = os.path.join('./checkpoints', "duke", "resnet50", 'net_last.pth')
    network.load_state_dict(torch.load(save_path))
    return network

def load_network2(network):
    save_path = os.path.join('./checkpoints', "market", "resnet50", 'net_last.pth')
    network.load_state_dict(torch.load(save_path))
    return network

def load_network3(network):
    save_path = os.path.join('./checkpoints', "new", 'resnet50_400x300.pth')
    network.load_state_dict(torch.load(save_path))
    return network

def load_image(path):
    src = Image.open(path)
    y = src.size[1]
    transform_att = T.Compose([
        T.Resize(size=(y, y)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    src = transform_att(src)
    src = src.unsqueeze(dim=0)

    return src

model3 = model_dict["resnet50"](num_cls_dict["duke"])
model3 = load_network(model3)

model3.eval()

model4 = model_dict["resnet50"](num_cls_dict["market"])
model4 = load_network2(model4)

model4.eval()

model5 = model_dict["resnet50"](num_cls_dict["market"])
model5 = load_network3(model5)

model5.eval()

### Getting prediction vector
def get_prediction(img_path, threshold):
    img = Image.open(img_path)
    transform = T.Compose([T.ToTensor()])
    img = transform(img)
    img = img.cuda()
    pred = model([img])
    pred = pred
    pred_class = [COCO_INSTANCE_CATEGORY_NAMES[i] for i in list(pred[0]['labels'].cuda())]
    pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].detach().cuda())]
    pred_score = list(pred[0]['scores'].detach().cuda())
    pred_t = [pred_score.index(x) for x in pred_score if x > threshold][-1]
    pred_boxes = pred_boxes[:pred_t+1]
    pred_class = pred_class[:pred_t+1]
    pred_score = pred_score[:pred_t+1]

    return pred_boxes, pred_class, pred_score


#### Getting vector for cosine similarity
def get_vector(image_name):
    img = Image.open(image_name)
    transform = Variable(normalize(to_tensor(scaler(img))).unsqueeze(0))
    transform = transform.cuda()
    embed = torch.zeros(1, 512, 1, 1)

    def copy_data(m, i, o):
        embed.copy_(o.data)

    lay = layer.register_forward_hook(copy_data)
    model2(transform)
    lay.remove()

    return embed


### Crop images
def crop_image(img, coord):
    x = coord[i][0][0]
    w = coord[i][1][0] - coord[i][0][0]
    y = coord[i][0][1]
    h = coord[i][1][1] - coord[i][0][1]
    cropped = img[int(y):int(y + h), int(x):int(x + w)]

    return cropped


### Begin Sorting
ct_frame = 0
color_val = list()
att_id = list()

feature_list = list()
att_duke_list = list()
att_market_list = list()
att_market_2_list = list()

for _ in range(0, int(args.num_id)): color_val.append((random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))

for path, dirs, files in os.walk("closed_set/gallery/"):
    if len(files) != 0:
        att_id.append(path[19:])

        gal_pic_feature = get_vector(path + "/" + files[0])
        feature_list.append(gal_pic_feature)

        gal_pic_att = load_image(path + "/" + files[0])
       # gal_pic_att = gal_pic_att.cuda()

        gal_pic_att_duke = model3.forward(gal_pic_att)
        att_duke_list.append(gal_pic_att_duke)

        gal_pic_att_market = model4.forward(gal_pic_att)
        att_market_list.append(gal_pic_att_market)

        gal_pic_att_market_2 = model4.forward(gal_pic_att)
        att_market_list.append(gal_pic_att_market_2)

for pic in os.listdir(args.data_path):
    result_conf = list()
    box_list = list()
    check_gal = [False for i in range(int(args.num_id))]
    ct_ppl = 0

    boxes, pred_cls, pred_score = get_prediction(args.data_path + pic, 0.5)
    img_crop = cv2.imread(args.data_path + pic)  # Image for cropping
    img_box = cv2.imread(args.data_path + pic)  # Image for boundary boxes

    for i in range(len(boxes)):
        check_common = 0
        if pred_cls[i] == "person":
            box_list.append(i)
            ct_ppl += 1
            cropped = crop_image(img_crop, boxes)
            cv2.imwrite("closed_set/temp/test.jpg", cropped)
            cur_pic_feature = get_vector("closed_set/temp/test.jpg")
            cur_pic_att = load_image("closed_set/temp/test.jpg")
        #    cur_pic_att = cur_pic_att.cuda()

            cur_pic_att_duke = model3.forward(cur_pic_att)
            cur_pic_att_market = model4.forward(cur_pic_att)
            cur_pic_att_market_2 = model5.forward(cur_pic_att)

            for id in range(int(args.num_id)):
                cos_sim_feature = cos(feature_list[id], cur_pic_feature)

                cos_sim_att_duke = cos(att_duke_list[id], cur_pic_att_duke)
                cos_sim_att_market = cos(att_market_list[id], cur_pic_att_market)
                cos_sim_att_market_2 = cos(att_market_list[id], cur_pic_att_market_2)

                final_cos = cos_sim_feature.item() + cos_sim_att_duke.item() + cos_sim_att_market.item() + cos_sim_att_market_2.item()

                result_conf.append(final_cos)
    if ct_ppl <= int(args.num_id):
        check_cur = [False for i in range(ct_ppl)]
        ct_check = 0
        while ct_check != ct_ppl:
            max_index = result_conf.index(max(result_conf))
            gal_who = max_index % int(args.num_id)
            cur_who = int(max_index / int(args.num_id))
            if check_gal[gal_who] is False | check_cur[cur_who] is False:
                cv2.rectangle(img_box, boxes[box_list[cur_who]][0], boxes[box_list[cur_who]][1], color=color_val[gal_who], thickness=3)
                cv2.putText(img_box, att_id[gal_who], boxes[box_list[cur_who]][0], cv2.FONT_HERSHEY_COMPLEX_SMALL, fontScale=2, color=color_val[gal_who], thickness=2)
                check_gal[gal_who] = True
                check_cur[cur_who] = True
                result_conf[max_index] = 0
                ct_check += 1
            else:
                result_conf[max_index] = 0

    cv2.imwrite("closed_set/result/frame_%04d.jpg" % ct_frame, img_box)
    ct_frame += 1
    print("%d th frame done" % ct_frame)