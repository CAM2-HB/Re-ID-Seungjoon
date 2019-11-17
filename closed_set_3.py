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
import sys
sys.path.append('../')
from openpose.body.estimator import BodyPoseEstimator
from net import *

### Get args
parser = argparse.ArgumentParser()
parser.add_argument('data_path', help='Path to images')
args = parser.parse_args()

### Joint
estimator = BodyPoseEstimator(pretrained=True)

### Load the pretrained model for detection
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
#model.cuda()
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
#model2.cuda()
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
    save_path = os.path.join('./checkpoints', "new", 'resnet50_400x300.pth')
    #save_path = os.path.join('./checkpoints', "market", "resnet50", 'net_last.pth')
    network.load_state_dict(torch.load(save_path))

    return network

def load_image(path):
    src = Image.open(path)
    y = src.size[1]
    transform_att = T.Compose([
        T.Resize(size=(int(y), int(y))),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    src = transform_att(src)
    src = src.unsqueeze(dim=0)

    return src

model3 = model_dict["resnet50"](num_cls_dict["market"])
model3 = load_network(model3)
#model3.cuda()
model3.eval()


### Getting prediction vector
def get_prediction(img_path, threshold):
    img = Image.open(img_path)
    transform = T.Compose([T.ToTensor()])
    img = transform(img)
    #img = img.cuda()
    pred = model([img])
    pred = pred
    pred_class = [COCO_INSTANCE_CATEGORY_NAMES[i] for i in list(pred[0]['labels'])]
    #pred_class = [COCO_INSTANCE_CATEGORY_NAMES[i] for i in list(pred[0]['labels'].cuda())]
    pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].detach())]
    #pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].detach().cuda())]
    pred_score = list(pred[0]['scores'].detach())
    #pred_score = list(pred[0]['scores'].detach().cuda())
    pred_t = [pred_score.index(x) for x in pred_score if x > threshold][-1]
    pred_boxes = pred_boxes[:pred_t+1]
    pred_class = pred_class[:pred_t+1]
    pred_score = pred_score[:pred_t+1]

    return pred_boxes, pred_class, pred_score

def get_vector(image_name):
    img = Image.open(image_name)
    transform = Variable(normalize(to_tensor(scaler(img))).unsqueeze(0))
    #transform = transform.cuda()
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
first_time = 1
id_att = list()
id_feature = list()
#id_pose = list()

for _ in range(0, 100): color_val.append((random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))

for pic in os.listdir(args.data_path):          ### Every image in data set
    result_conf = list()
    box_list = list()
    #pose_list = list()
    ct_ppl = 0
    check_gal = [False for i in range(100)]

    boxes, pred_cls, pred_score = get_prediction(args.data_path + pic, 0.5)
    img_crop = cv2.imread(args.data_path + pic)  # Image for cropping
    img_box = cv2.imread(args.data_path + pic)  # Image for boundary boxes

    for i in range(len(boxes)):        ### Every detected object
        if pred_cls[i] == "person":        ### If it's person
            cropped = crop_image(img_crop, boxes)        ### Crop it
            cv2.imwrite("closed_set/temp/test.jpg", cropped)        ### Save it into temp folder
            image_src = cv2.imread("closed_set/temp/test.jpg")       ### Get the number of joints
            keypoints = estimator(image_src)
            count = 0                                                ### Number of joints
            if len(keypoints) != 0:
                for ct in keypoints[0]:
                    if ct[0] + ct[1] + ct[2] != 0:
                        count += 1
            else:
                break

            if (count > 7) & (len(keypoints) < 2):                   ## If valid object
                cur_pic = load_image("closed_set/temp/test.jpg")
                #cur_pic = cur_pic.cuda()
                cur_pic_att = model3.forward(cur_pic)
                if first_time == 1:
                    ### Save feature
                    feature = get_vector("closed_set/temp/test.jpg")
                    id_feature.append(feature)

                    ### Save pose
                    #keypoints.astype(float)
                    #keypoints = keypoints[0][:, :-1]
                    #tensor = torch.from_numpy(keypoints)
                    #tensor = tensor.type(torch.FloatTensor)
                    #id_pose.append(tensor)

                    ### Save attribute
                    id_att.append(cur_pic_att)                    ## Will save attribute vector

                    cv2.rectangle(img_box, boxes[i][0], boxes[i][1], color=color_val[len(id_feature)- 1], thickness=3)
                    cv2.putText(img_box, str(len(id_feature) - 1), boxes[i][0], cv2.FONT_HERSHEY_COMPLEX_SMALL, fontScale=2, color=color_val[len(id_feature) - 1], thickness=2)

                else:
                    ### Get feature
                    feature = get_vector("closed_set/temp/test.jpg")

                    ### Get pose
                    #keypoints.astype(float)
                    #keypoints = keypoints[0][:, :-1]
                    #tensor = torch.from_numpy(keypoints)
                    #tensor = tensor.type(torch.FloatTensor)

                    temp_feature_list = list()
                    #temp_pose_list = list()
                    ### Compare
                    for ct, img in enumerate(id_feature):
                        cos_sim_feature = cos(feature, img)
                        temp_feature_list.append(cos_sim_feature.item())

                        #cos_sim_pose = cos(tensor, id_pose[ct])
                        #temp_pose_list.append(cos_sim_pose[0].item())

                    max_index_feature = temp_feature_list.index(max(temp_feature_list))
                    #max_index_pose = temp_pose_list.index(max(temp_pose_list))
                    print(temp_feature_list)
                    #if (max(temp_feature_list) > 0.8) & (max(temp_pose_list) > 0.8) & (max_index_feature == max_index_pose):
                    if max(temp_feature_list) > 0.8:
                        ### Update the feature & pose
                        id_feature[max_index_feature] = feature
                        #id_pose[max_index_pose] = tensor

                        ### Draw Rectangles
                        cv2.rectangle(img_box, boxes[i][0], boxes[i][1], color=color_val[max_index_feature], thickness=3)
                        cv2.putText(img_box, str(max_index_feature), boxes[i][0], cv2.FONT_HERSHEY_COMPLEX_SMALL, fontScale=2, color=color_val[max_index_feature], thickness=2)
                        check_gal[max_index_feature] = True

                        ### Update the attribute
                        #cur_pic = load_image("closed_set/temp/test.jpg")
                        #cur_pic = cur_pic.cuda()
                        #cur_pic_att = model3.forward(cur_pic)
                        id_att[max_index_feature] = id_att[max_index_feature].add(cur_pic_att)
                        id_att[max_index_feature] = id_att[max_index_feature].div(2)
                    else:           ### New person?
                        #cur_pic = load_image("closed_set/temp/test.jpg")
                        #cur_pic = cur_pic.cuda()
                        #cur_pic_att = model3.forward(cur_pic)

                        for ct, att in enumerate(id_att):
                            if check_gal[ct] is True:
                                cos_sim_att = cos(cur_pic_att, att)
                                if cos_sim_att.item() > 0.8:         ### Re-ID
                                    ### Update Attribute
                                    id_att[ct] = id_att[ct].add(cur_pic_att)
                                    id_att[ct] = id_att[ct].div(2)
                                    check_gal[ct]= True
                                    cv2.rectangle(img_box, boxes[i][0], boxes[i][1], color=color_val[ct], thickness=3)
                                    cv2.putText(img_box, str(ct), boxes[i][0], cv2.FONT_HERSHEY_COMPLEX_SMALL, fontScale=2, color=color_val[ct], thickness=2)
                                else:        ### New person
                                    ### Save feature
                                    feature = get_vector("closed_set/temp/test.jpg")
                                    id_feature.append(feature)

                                    ### Save pose
                                    #id_pose.append(tensor)

                                    ### Save attribute
                                    #cur_pic = load_image("closed_set/temp/test.jpg")
                                    #cur_pic = cur_pic.cuda()
                                    #cur_pic_att = model3.forward(cur_pic)
                                    id_att.append(cur_pic_att)  ## Will save attribute vector

                                    cv2.rectangle(img_box, boxes[i][0], boxes[i][1], color=color_val[len(id_feature)], thickness=3)
                                    cv2.putText(img_box, str(len(id_feature)), boxes[i][0], cv2.FONT_HERSHEY_COMPLEX_SMALL, fontScale=2, color=color_val[len(id_feature)], thickness=2)
    first_time = 0

    cv2.imwrite("closed_set/result/frame_%04d.jpg" % ct_frame, img_box)
    ct_frame += 1
    print("%d th frame done" % ct_frame)