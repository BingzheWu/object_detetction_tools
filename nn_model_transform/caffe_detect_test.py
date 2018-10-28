import caffe
import sys
import cv2
import numpy as np
from darknet_utils import *
import torch
def load_image(image_file):
    img = cv2.imread(image_file)
    img = cv2.resize(img, (416, 416))
    img = img/255.0
    img = np.expand_dims(img, 0)
    img = img.transpose((0,3,1,2))
    return img
def test_yolov3_detector(prototxt_file, weights_file, image_file):
    print("Load Net")
    net = caffe.Net(prototxt_file, weights_file, caffe.TEST)
    print("Load net sucess")
    img = load_image(image_file)
    net.blobs['data'].data[...] = img
    net.forward()
    print(net.blobs['layer107-yolo'].data)
    return net.blobs['layer107-yolo'].data, net.blobs['layer95-yolo'].data, net.blobs['layer83-yolo'].data
def vis_detections(prototxt_file, weights_file, image_file, num_classes, masks, thresh = 0.5, anchors = [10,13,  16,30,  33,23,  30,61,  62,45,  59,119,  116,90,  156,198,  373,326]):
    features = test_yolov3_detector(prototxt_file, weights_file, image_file)
    dets = []
    for idx, feature in enumerate(features):
        feature = torch.Tensor(feature)
        print(feature.size())
        masked_anchors = []
        for m in masks[idx]:
            masked_anchors += anchors[m*2:(m+1)*2]
        masked_anchors = [anchor/32.0 for anchor in masked_anchors]
        boxes = get_region_boxes(feature, thresh, num_classes, masked_anchors, 3)
        dets.append(boxes)
    list_boxes = dets
    boxes = list_boxes[0][0] + list_boxes[1][0] + list_boxes[2][0]
    boxes = nms(boxes, 0.4)
    img = cv2.imread(image_file)
    class_names = load_class_names('data/coco.names')
    plot_boxes_cv2(img, boxes, savename='predict.jpg', class_names=class_names)

if __name__ == '__main__':
    prototxt_file = sys.argv[1]
    weights_file = sys.argv[2]
    image_file = sys.argv[3]
    masks = [[0,1,2], [3,4,5], [6,7,8]]
    num_classes = 80
    print("start test")
    vis_detections(prototxt_file, weights_file, image_file, num_classes, masks)