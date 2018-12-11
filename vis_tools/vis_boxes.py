import cv2
import numpy as np
import os
import glob
__extensions__ = ['jpg', 'png']
def read_label_file(label_file):
    boxes = open(label_file, 'r').readlines()
    rects = []
    for box in boxes:
        box = box.split(',')
        print(box)
        x1, x2 = float(box[1]), float(box[3])
        y1, y2 = float(box[5]), float(box[7])
        rects.append([x1,x2,y1,y2])
    return rects
def draw_rects(img, rects, save_path):
    h,w,_ = img.shape
    print(rects)
    for x1, x2, y1, y2 in rects:
        x1,x2 = int(x1*w), int(x2*w)
        y1, y2 = int(y1*h), int(y2*h)
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 00), 2)
    cv2.imwrite(save_path, img)
def vis_test_dir(image_dir, box_file_dir, dst_dir):
    image_files = os.listdir(image_dir)
    if not os.path.exists(dst_dir):
        os.mkdir(dst_dir)
    for image_file in image_files:
        prefix, img_type = image_file.split('.')
        if img_type in __extensions__:
            image_file = os.path.join(image_dir, prefix+'.'+img_type)
            image = cv2.imread(image_file)
            label_file = os.path.join(box_file_dir, prefix+'_lp.txt')
            if os.path.isfile(label_file):
                rects = read_label_file(label_file)
            save_path = os.path.join(dst_dir, prefix+'.jpg')
            draw_rects(image, rects, save_path)

def test_vis_test_dir():
    import sys
    image_dir = sys.argv[1]
    box_file_dir = sys.argv[2]
    dst_dir = sys.argv[3]
    vis_test_dir(image_dir, box_file_dir, dst_dir)
if __name__ == '__main__':
    test_vis_test_dir()
            
        