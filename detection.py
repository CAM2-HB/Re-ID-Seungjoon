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
parser.add_argument('dataset', default='duke', type=str, help='dataset')
parser.add_argument('model', default='resnet50', type=str, help='model')
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

if os.path.exists("test_result") == False:
    os.mkdir("test_result")

if os.path.exists("test_result_boxes") == False:
    os.mkdir("test_result_boxes")

if os.path.exists("test_result/temp") == False:
    os.mkdir("test_result/temp")


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

dataset_dict = {
    'market'  :  'Market-1501',
    'duke'  :  'DukeMTMC-reID',
}
model_dict = {
    'resnet18'  :  ResNet18_nFC,
    'resnet34'  :  ResNet34_nFC,
    'resnet50'  :  ResNet50_nFC,
    'densenet'  :  DenseNet121_nFC,
    'resnet50_softmax'  :  ResNet50_nFC_softmax,
}
num_cls_dict = { 'market':30, 'duke':23 }
num_ids_dict = { 'market':751, 'duke':702 }

transform_att = T.Compose([
    T.Resize(size=(288, 144)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def load_network(network):
    save_path = os.path.join('./checkpoints', args.dataset, args.model, 'net_last.pth')
    network.load_state_dict(torch.load(save_path))
    return network

def load_network2(network):
    save_path = os.path.join('./checkpoints', "market", "resnet50", 'net_last.pth')
    network.load_state_dict(torch.load(save_path))
    return network

def load_image(path):
    src = Image.open(path)
    src = transform_att(src)
    src = src.unsqueeze(dim=0)

    return src

model3 = model_dict[args.model](num_cls_dict[args.dataset])
model3 = load_network(model3)
model3.eval()

model4 = model_dict["resnet50"](num_cls_dict["market"])
model4 = load_network2(model4)
model4.eval()


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
ct_people = 0
ct_frame = 0
ct_file = 0
ct_color = 0
file_list = list()
color_val = list()
att_id = list()

for _ in range(0, int(args.num_id)): color_val.append((random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))
for x in range(0, int(args.num_id)): att_id.append("Person " + str(x))

for pic in os.listdir(args.data_path):
    check = [False for i in range(int(args.num_id))]
    boxes, pred_cls, pred_score = get_prediction(args.data_path + pic, 0.5)
    img_crop = cv2.imread(args.data_path + pic)  # Image for cropping
    img_box = cv2.imread(args.data_path + pic)  # Image for boundary boxes

    for i in range(len(boxes)):
        check_common = 0
        if pred_cls[i] == "person":
#            print(pred_score[i])
            if ct_file == 0:
                ### Cropping
                cropped = crop_image(img_crop, boxes)
                os.mkdir("test_result/%02d" % ct_people)
                file_list.append("0" + str(ct_people))
                cv2.imwrite("test_result/%02d/test_%04d.jpg" % (ct_people, ct_file), cropped)

                ### Boundary Boxes
                cv2.rectangle(img_box, boxes[i][0], boxes[i][1], color=color_val[ct_color], thickness=3)
                cv2.putText(img_box, att_id[ct_color], boxes[i][0], cv2.FONT_HERSHEY_COMPLEX_SMALL, fontScale=2, color=color_val[ct_color], thickness=2)
                ct_color += 1
                ct_people += 1
            else:
                cropped = crop_image(img_crop, boxes)
                cv2.imwrite("test_result/temp/test.jpg", cropped)
                cur_pic = get_vector("test_result/temp/test.jpg")

                for path, dirs, files in os.walk("test_result/"):
                    if (len(files) != 0) & (path != "test_result/temp"):
                        comp_pic = get_vector(path + "/" + files[-1])
                        cos_sim = cos(cur_pic, comp_pic)

                        if cos_sim.item() > 0.8:
                            check_common = 1
                            src_cur_att = load_image("test_result/temp/test.jpg")
                            out_cur_att = model3.forward(src_cur_att)
                            src_comp_att = load_image(path + "/" + files[-1])
                            out_comp_att = model3.forward(src_comp_att)
                            cos_sim = cos(out_cur_att, out_comp_att)

                            out_cur_att2 = model4.forward(src_cur_att)
                            out_comp_att2 = model4.forward(src_comp_att)
                            cos_sim2 = cos(out_cur_att2, out_comp_att2)

                            if (cos_sim.item() > 0.9) | (cos_sim2.item() > 0.9):
                                if check[int(path[-2:])] is False:
                                    check[int(path[-2:])] = True
                                    ### Cropping
                                    cv2.imwrite(path + "/" + "test_%04d.jpg" % ct_file, cropped)
                                    os.remove("test_result/temp/test.jpg")

                                    ### Boundary Boxes
                                    cv2.rectangle(img_box, boxes[i][0], boxes[i][1], color=color_val[int(path[-1])], thickness=3)
                                    cv2.putText(img_box, att_id[int(path[-2:])], boxes[i][0], cv2.FONT_HERSHEY_COMPLEX_SMALL, fontScale=2, color=color_val[int(path[-1])], thickness=2)
                                    break
                            else:
                                print(cos_sim.item())
                else:
                    if check_common == 0:
                        ### Cropping
                        os.mkdir("test_result/%02d" % ct_people)
                        file_list.append("0" + str(ct_people))
                        cv2.imwrite("test_result/%02d/test_%04d.jpg" % (ct_people, ct_file), cropped)
                        ct_people += 1
                        os.remove("test_result/temp/test.jpg")

                        ### Boundary Boxes
                        cv2.rectangle(img_box, boxes[i][0], boxes[i][1], color=color_val[ct_color], thickness=3)
                        cv2.putText(img_box, att_id[ct_color], boxes[i][0], cv2.FONT_HERSHEY_COMPLEX_SMALL, fontScale=2, color=color_val[ct_color], thickness=2)
                        ct_color += 1

    cv2.imwrite("test_result_boxes/frame_%04d.jpg" % ct_file, img_box)
    ct_file += 1

