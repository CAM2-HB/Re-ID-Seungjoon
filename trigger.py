import cv2
from skimage.measure import compare_ssim
import os
import torchvision
from PIL import Image
from torchvision import transforms as T

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


if os.path.exists("temp2") == False:
    os.mkdir("temp2")

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


def crop_image(img, coord):
    x = coord[i][0][0]
    w = coord[i][1][0] - coord[i][0][0]
    y = coord[i][0][1]
    h = coord[i][1][1] - coord[i][0][1]
    cropped = img[int(y):int(y + h), int(x):int(x + w)]

    return cropped


if __name__ == '__main__' :
    im = cv2.imread("test_images/frame139.jpg")
    r = cv2.selectROI(im, False, False)

    imCrop = im[int(r[1]):int(r[1]) + int(r[3]), int(r[0]):int(r[0]) + int(r[2])]

    grayA = cv2.cvtColor(imCrop, cv2.COLOR_BGR2GRAY)

    check = 0
    ct = 0
    for pic in os.listdir("test_images/"):
        if check == 0:
            im2 = cv2.imread("test_images/" + pic)
            imCrop2 = im2[int(r[1]):int(r[1]) + int(r[3]), int(r[0]):int(r[0]) + int(r[2])]

            grayB = cv2.cvtColor(imCrop2, cv2.COLOR_BGR2GRAY)

            (score, diff) = compare_ssim(grayA, grayB, full=True)
            diff = (diff * 255).astype("uint8")
            print(pic)
            print("SSIM: {}".format(score))
            print("-------------------------------")

            if score < 0.9:
                check = 1
        elif ct != 28:            ## Door is opened
            ct += 1
        else:
            im3 = cv2.imread("test_images/" + pic)
            r2 = cv2.selectROI(im3, False, False)
            final_im = im3[int(r2[1]):int(r2[1]) + int(r2[3]), int(r2[0]):int(r2[0]) + int(r2[2])]
            cv2.imwrite("temp2/temp.jpg", final_im)
            boxes, pred_cls, pred_score = get_prediction("temp2/temp.jpg", 0.5)
            for i in range(len(boxes)):
                if pred_cls[i] == "person":
                    cropped = crop_image(final_im, boxes)
                    cv2.imwrite("temp2/result.jpg", cropped)
                    break
            break

