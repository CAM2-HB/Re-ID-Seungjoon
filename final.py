import cv2
from skimage.measure import compare_ssim
import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision
from torchvision import transforms as T
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import os
import argparse
import random
import sys
sys.path.append('../')
from openpose.body.estimator import BodyPoseEstimator
from net import *

### Get args
parser = argparse.ArgumentParser()
parser.add_argument('data_path', help='Path to the first image')
args = parser.parse_args()

estimator = BodyPoseEstimator(pretrained=True)

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
    'resnet18': ResNet18_nFC,
    'resnet34': ResNet34_nFC,
    'resnet50': ResNet50_nFC,
    'densenet': DenseNet121_nFC,
    'resnet50_softmax': ResNet50_nFC_softmax,
}
num_cls_dict = {'market': 30, 'duke': 23}
num_ids_dict = {'market': 751, 'duke': 702}


def load_network(network):
    save_path = os.path.join('./checkpoints', "new", 'resnet50_400x300.pth')
    #save_path = os.path.join('./checkpoints', "duke", "resnet50", 'net_last.pth')
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
model3.cuda()
model3.eval()


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
    pred_boxes = pred_boxes[:pred_t + 1]
    pred_class = pred_class[:pred_t + 1]
    pred_score = pred_score[:pred_t + 1]

    return pred_boxes, pred_class, pred_score


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


def crop_image(img, coord, i):
    x = coord[i][0][0]
    w = coord[i][1][0] - coord[i][0][0]
    y = coord[i][0][1]
    h = coord[i][1][1] - coord[i][0][1]

    cropped = img[int(y):int(y + h), int(x):int(x + w)]

    return cropped


### Function to save the captured image into our gallery
def save_gallery(pic, grayA, x11, x12, y11, y12, x21, x22, y21, y22, threshold, num_ppl, check, att_list, feat_list,
                 iou_list):
    im2 = cv2.imread("test_images/" + pic)
    imCrop3 = im2[int(y11):int(y12), int(x11):int(x12)]

    grayB = cv2.cvtColor(imCrop3, cv2.COLOR_BGR2GRAY)

    (score, diff) = compare_ssim(grayA, grayB, full=True)

    if score > float(threshold):
        im3 = cv2.imread("test_images/" + pic)
        final_im = im3[int(y21):int(y22), int(x21):int(x22)]
        cv2.imwrite("closed_set/temp/temp.jpg", final_im)
        boxes, pred_cls, pred_score = get_prediction("closed_set/temp/temp.jpg", 0.5)
        for i in range(len(boxes)):
            if pred_cls[i] == "person":
                cropped = crop_image(final_im, boxes, i)
                os.mkdir("closed_set/gallery/%04d" % num_ppl)
                cv2.imwrite("closed_set/gallery/%04d/result.jpg" % num_ppl, cropped)

                get_pic = load_image("closed_set/gallery/%04d/result.jpg" % num_ppl)
                get_pic = get_pic.cuda()
                get_pic_att = model3.forward(get_pic)
                att_list.append(get_pic_att)

                temp = (x21, y21)
                boxes[i][0] = tuple(map(sum, zip(boxes[i][0], temp)))
                boxes[i][1] = tuple(map(sum, zip(boxes[i][1], temp)))

                iou_list.append(boxes[i])

                num_ppl += 1
        check = 0

    return num_ppl, check, att_list, feat_list


### Function to check if the door is opened
def check_trigger(pic, grayA, x, x2, y, y2, check):
    im2 = cv2.imread("test_images/" + pic)
    imCrop3 = im2[int(y):int(y2), int(x):int(x2)]

    grayB = cv2.cvtColor(imCrop3, cv2.COLOR_BGR2GRAY)

    (score, diff) = compare_ssim(grayA, grayB, full=True)

    if score < 0.27:
        check = 1

    return check


def iou(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0][0], boxB[0][0])
    yA = max(boxA[0][1], boxB[0][1])
    xB = min(boxA[1][0], boxB[1][0])
    yB = min(boxA[1][1], boxB[1][1])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[1][0] - boxA[0][0] + 1) * (boxA[1][1] - boxA[0][1] + 1)
    boxBArea = (boxB[1][0] - boxB[0][0] + 1) * (boxB[1][1] - boxB[0][1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou

if __name__ == '__main__':
    im = cv2.imread(args.data_path)
    imCrop = im[int(67):int(117), int(1471):int(1487)]
    grayA1 = cv2.cvtColor(imCrop, cv2.COLOR_BGR2GRAY)

    imCrop2 = im[int(70):int(110), int(354):int(375)]
    grayA2 = cv2.cvtColor(imCrop2, cv2.COLOR_BGR2GRAY)

    check_door_1 = 0
    check_door_2 = 0
    num_ppl = 0
    ct_frame = 0

    att_list = list()
    feat_list = list()
    color_val = list()
    iou_list = list()

    for _ in range(0, 100): color_val.append((random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))

    for pic in os.listdir("test_images/"):
        att_rank = list()
        box_list = list()
        person_list = list()

        ct_ppl = 0

        if check_door_1 == 1:
            num_ppl, check_door_1, att_list, feat_list = save_gallery(pic, grayA1, 1471, 1487, 67, 117, 1348, 1640, 72, 671, 0.87, num_ppl, check_door_1, att_list,
                                                                      feat_list, iou_list)

        elif check_door_1 == 0:
            check_door_1 = check_trigger(pic, grayA1, 1471, 1487, 67, 117, check_door_1)

        if check_door_2 == 1:
            num_ppl, check_door_2, att_list, feat_list = save_gallery(pic, grayA2, 354, 375, 70, 110, 144, 600, 64, 722, 0.84, num_ppl, check_door_2, att_list, feat_list, iou_list)

        elif check_door_2 == 0:
            check_door_2 = check_trigger(pic, grayA2, 354, 375, 70, 110, check_door_2)

        if num_ppl > 0:
            boxes, pred_cls, pred_score = get_prediction("test_images/" + pic, 0.5)
            img_crop = cv2.imread("test_images/" + pic)  # Image for cropping
            img_box = cv2.imread("test_images/" + pic)  # Image for boundary boxes

            check_gal = [False for i in range(100)]
            check_person = [False for i in range(100)]
            att_rank = list()
            not_assigned_coord = list()
            ct_ppl = 0
            ct_not_assigned = 0
            check_ppl = 0

            for i in range(len(boxes)):  ### Every detected object
                if pred_cls[i] == "person":  ### If it's person
                    cropped = crop_image(img_crop, boxes, i)  ### Crop it
                    cv2.imwrite("closed_set/temp/test.jpg", cropped)  ### Save it into temp folder
                    image_src = cv2.imread("closed_set/temp/test.jpg")  ### Get the number of joints
                    keypoints = estimator(image_src)
                    count = 0  ### Number of joints
                    if len(keypoints) != 0:
                        for ct in keypoints[0]:
                            if ct[0] + ct[1] + ct[2] != 0:
                                count += 1
                    else:
                        break

                    if (count > 10) & (len(keypoints) < 2):  ## If valid object
                        for ct, j in enumerate(iou_list):
                            iou_val = iou(boxes[i], j)
                            if iou_val > 0.5:
                                if check_gal[ct] is False:
                                    cv2.rectangle(img_box, boxes[i][0], boxes[i][1], color=color_val[ct], thickness=3)
                                    cv2.putText(img_box, str(ct), boxes[i][0], cv2.FONT_HERSHEY_COMPLEX_SMALL, fontScale=2, color=color_val[ct], thickness=2)
                                    iou_list[ct] = boxes[i]
                                    check_gal[ct] = True
                                    ct_ppl += 1
                                    break
                        else:
                            if (ct_ppl < len(iou_list)) & (ct_ppl < num_ppl):
                                cv2.imwrite("closed_set/notassigned/ppl_%04d.jpg" % ct_not_assigned, cropped)  ### Save it into temp folder
                                not_assigned_coord.append(boxes[i])
                                ct_not_assigned += 1
                                check_ppl = 1

            if (check_ppl == 1) & (ct_ppl < len(iou_list)) & (ct_ppl < num_ppl):
                for img in os.listdir("closed_set/notassigned/"):
                    cur_pic = load_image("closed_set/temp/test.jpg")
                    cur_pic = cur_pic.cuda()
                    cur_pic_att = model3.forward(cur_pic)
                    for ct, img in enumerate(att_list):
                        cos_sim_att = cos(cur_pic_att, img)
                        cos_sim_final = cos_sim_att.item()
                        att_rank.append(cos_sim_final)
                temp_not_assigned = 0
                while (temp_not_assigned != ct_not_assigned) & (temp_not_assigned < len(att_list)):
                    max_index_att = att_rank.index(max(att_rank))
                    index_person = int(max_index_att / len(att_list))
                    index_gal = max_index_att % len(att_list)
                    if (check_gal[index_gal] is False) & (check_person[index_person] is False):
                        cv2.rectangle(img_box, not_assigned_coord[index_person][0], not_assigned_coord[index_person][1], color=color_val[index_gal], thickness=3)
                        cv2.putText(img_box, str(index_gal), not_assigned_coord[index_person][0], cv2.FONT_HERSHEY_COMPLEX_SMALL, fontScale=2, color=color_val[index_gal], thickness=2)
                        check_gal[index_gal] = True
                        check_person[index_person] = True
                        iou_list[index_gal] = not_assigned_coord[index_person]
                        temp_not_assigned += 1
                    att_rank[max_index_att] = 0

            cv2.imwrite("closed_set/result/frame_%04d.jpg" % ct_frame, img_box)
            ct_frame += 1
            path = "closed_set/notassigned/"
            if len(os.listdir(path)) != 0:
                for root, dirs, files in os.walk(path):
                    for file in files:
                        os.remove(os.path.join(root, file))