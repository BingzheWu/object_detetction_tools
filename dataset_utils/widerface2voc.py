"""
data_conversion.py 是将 wider_faced的标注数据集转成yolov3所需要的labels
- 每个图片转成对应名称的label
- 由于yolov3需要的是归一化之后的中心坐标和w，h所以在convert方法中进行了归一化和数据转换
@author: XP
"""
import cv2
import os
import shutil
def convert(size, box):                 #size是图片尺寸 box是坐标
    dw = 1./size[0]
    dh = 1./size[1]
    if box[0] < 0:
        box[0] = 1
    if box[1] >= size[0]:
        box[1] = size[0]-1
    if box[2] <= 0:
        box[2] = 1
    if box[3] >= size[1]:
        print(box[3])
        box[3] = size[1]-1
    #box = [0 if box[i] < 0  else: box[i] for i in range(4)]
    x = (box[0] + box[1])/2.0
    y = (box[2] + box[3])/2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)

def readfile(filename, prefix = 'train'):
    fo = open(filename, "r")  # 读入标注结果集
    prefix_lists = open(os.path.join('/data/widerface/%s.txt'%(prefix)), 'w')
    while True:
        key = next(fo, -1)
        if key == -1:
            break
        key = key.replace('\n', '')
        key1 = key.split("/")
        key1 = key1[1].split(".")
        key1 = key1[0]         #获取图片名称
        print(key)
        save_dir = os.path.join('/data/widerface/WIDER_%s/labels/'%(prefix))
        print(save_dir)
        list_file = open(os.path.join(save_dir,'%s.txt' % (key1)), 'w')                                             #新建对应图片的label，存放在My_labels文件夹下
        value = []
        image_dst = key.split('/')[-1]
        image_dst = os.path.join('/data/widerface/WIDER_%s/images/%s'%(prefix,image_dst))
        key = "/data/widerface/WIDER_train/images/%s"%(key) 	#该图片位置
        shutil.copyfile(key, image_dst)
        image = cv2.imread(key)																						#用opencv读取该图片
        image_size = []																							
        # print(image.shape[0],image.shape[1])
        image_size = [image.shape[1],image.shape[0]]											#得到图片尺寸，为了后面进行归一化
        
        prefix_lists.write(image_dst+'\n')
        num = next(fo, -1)
        for i in range(int(num)):																					#遍历每一张标注的脸
            value = next(fo, -1).split(' ')
            box = [int(value[0]),int(value[0])+int(value[2]),int(value[1]),int(value[1])+int(value[3])]
            x, y, w, h = convert(image_size,box)
            # print(x, y, w, h)
            list_file.write('0 %s %s %s %s\n' % (x,y,w,h))									#将转换后的坐标写入label
    fo.close()



if __name__ == '__main__':
    import sys
    prefix = sys.argv[1]
    filename = "/data/widerface/wider_face_split/wider_face_train_bbx_gt.txt"    #标注文件位置
    readfile(filename, prefix)