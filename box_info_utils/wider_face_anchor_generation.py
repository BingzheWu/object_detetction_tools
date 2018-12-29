import numpy as np
import os
import glob
import sys
sys.path.append('.')
from box_info_utils.anchor_kmeans import kmeans, traslate_boxes

def parse_yolo_label_file(file_name):
    f = open(file_name, 'r').readlines()
    boxes = []
    for label in f:
        label = label.strip().split(' ')
        w,h = float(label[-2]), float(label[-1])
        if w>0 and h>0.0:
            boxes.append([w,h])
        else:
            continue
    return np.array(boxes)

def load_boxes(data_dir):
    label_dir = os.path.join(data_dir, 'labels')
    labels = glob.glob(label_dir+"/*.txt")
    boxes_array = np.empty([0,2])
    for i, label_file in enumerate(labels):
        tmp = parse_yolo_label_file(label_file)
        #print(tmp.shape)
        if tmp.shape[0]>=1:
            boxes_array = np.concatenate((boxes_array, tmp))
        #if i > 3:
        #    break
    print(boxes_array)
    return boxes_array
def generate_anchors(boxes):
    """
    input:
        boxes: [xmin, ymin, xmax, ymax]
    """
    #boxes = traslate_boxes(boxes)
    out = kmeans(boxes, 9)

def main():
    k = 9
    import sys
    data_dir = sys.argv[1]
    boxes = load_boxes(data_dir)
    out = kmeans(boxes, 9)
    print(out)
if __name__ == '__main__':
    main()