from ctypes import *
import math
import random
import cv2
def sample(probs):
    s = sum(probs)
    probs = [a/s for a in probs]
    r = random.uniform(0, 1)
    for i in range(len(probs)):
        r = r - probs[i]
        if r <= 0:
            return i
    return len(probs)-1

def c_array(ctype, values):
    arr = (ctype*len(values))()
    arr[:] = values
    return arr

class BOX(Structure):
    _fields_ = [("x", c_float),
                ("y", c_float),
                ("w", c_float),
                ("h", c_float)]

class DETECTION(Structure):
    _fields_ = [("bbox", BOX),
                ("classes", c_int),
                ("prob", POINTER(c_float)),
                ("mask", POINTER(c_float)),
                ("objectness", c_float),
                ("sort_class", c_int)]


class IMAGE(Structure):
    _fields_ = [("w", c_int),
                ("h", c_int),
                ("c", c_int),
                ("data", POINTER(c_float))]

class METADATA(Structure):
    _fields_ = [("classes", c_int),
                ("names", POINTER(c_char_p))]

    

lib = CDLL("/home/bingzhe/project/face_detection/darknet-original/libdarknet.so", RTLD_GLOBAL)
#lib = CDLL("libdarknet.so", RTLD_GLOBAL)
lib.network_width.argtypes = [c_void_p]
lib.network_width.restype = c_int
lib.network_height.argtypes = [c_void_p]
lib.network_height.restype = c_int

predict = lib.network_predict
predict.argtypes = [c_void_p, POINTER(c_float)]
predict.restype = POINTER(c_float)

set_gpu = lib.cuda_set_device
set_gpu.argtypes = [c_int]

make_image = lib.make_image
make_image.argtypes = [c_int, c_int, c_int]
make_image.restype = IMAGE

get_network_boxes = lib.get_network_boxes
get_network_boxes.argtypes = [c_void_p, c_int, c_int, c_float, c_float, POINTER(c_int), c_int, POINTER(c_int)]
get_network_boxes.restype = POINTER(DETECTION)

make_network_boxes = lib.make_network_boxes
make_network_boxes.argtypes = [c_void_p]
make_network_boxes.restype = POINTER(DETECTION)

free_detections = lib.free_detections
free_detections.argtypes = [POINTER(DETECTION), c_int]

free_ptrs = lib.free_ptrs
free_ptrs.argtypes = [POINTER(c_void_p), c_int]

network_predict = lib.network_predict
network_predict.argtypes = [c_void_p, POINTER(c_float)]

reset_rnn = lib.reset_rnn
reset_rnn.argtypes = [c_void_p]

load_net = lib.load_network
load_net.argtypes = [c_char_p, c_char_p, c_int]
load_net.restype = c_void_p

do_nms_obj = lib.do_nms_obj
do_nms_obj.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

do_nms_sort = lib.do_nms_sort
do_nms_sort.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

free_image = lib.free_image
free_image.argtypes = [IMAGE]

letterbox_image = lib.letterbox_image
letterbox_image.argtypes = [IMAGE, c_int, c_int]
letterbox_image.restype = IMAGE

load_meta = lib.get_metadata
lib.get_metadata.argtypes = [c_char_p]
lib.get_metadata.restype = METADATA

load_image = lib.load_image_color
load_image.argtypes = [c_char_p, c_int, c_int]
load_image.restype = IMAGE

rgbgr_image = lib.rgbgr_image
rgbgr_image.argtypes = [IMAGE]

predict_image = lib.network_predict_image
predict_image.argtypes = [c_void_p, IMAGE]
predict_image.restype = POINTER(c_float)

def classify(net, meta, im):
    out = predict_image(net, im)
    res = []
    for i in range(meta.classes):
        res.append((meta.names[i], out[i]))
    res = sorted(res, key=lambda x: -x[1])
    return res

def detect(net, meta, image, thresh=.5, hier_thresh=.5, nms=.45):
    """
    dets: [num, prob, (x, y, w, h)]
    """
    im = load_image(image, 0, 0)
    num = c_int(0)
    pnum = pointer(num)
    predict_image(net, im)
    dets = get_network_boxes(net, im.w, im.h, thresh, hier_thresh, None, 0, pnum)
    num = pnum[0]
    if (nms): do_nms_obj(dets, num, meta.classes, nms);

    res = []
    for j in range(num):
        for i in range(meta.classes):
            if dets[j].prob[i] > 0:
                b = dets[j].bbox
                res.append((meta.names[i], dets[j].prob[i], (b.x, b.y, b.w, b.h)))
    res = sorted(res, key=lambda x: -x[1])
    free_image(im)
    free_detections(dets, num)
    return res
def main():
    import os
    #from tqdm import tqdm
    import cv2
    import sys
    cfg_file = sys.argv[1]
    weight_file = sys.argv[2]
    meta_file = sys.argv[3]
    test_dir = sys.argv[4]
    mode = sys.argv[5]
    net = load_net(cfg_file, weight_file, 0)
    meta = load_meta(meta_file)
    print('loaded_ok')
    save_dir = os.path.join(test_dir, 'results')
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    '''
    if 'otureo' in test_dir:
        SHOW = 0
        SAVE = 1
        os.system('rm {}/*.jpg'.format(save_dir))
        outfile = open('/tmp/rst.txt', 'w')
        # os.system('rm {}/*.mp4'.format(save_dir))
    else:
        outfile=open('results/wider/rst.txt','w')
        SHOW=1
        SAVE=0
    '''
    SAVE  = 1
    SHOW = 0
    cnt = 1
    if mode == 'wider':
        sys.path.append('/home/bingzhe/project/object_detection_tools/test_scripts')
        from widerface_eval import darknet2widerface, save_widerface_bboxes
        for sub_dir in os.listdir(test_dir):
            if sub_dir =='results':
                continue
            sub_path = os.path.join(test_dir, sub_dir)
            for image_file in os.listdir(sub_path):
                image_file_path = os.path.join(sub_path, image_file)
                r = detect(net, meta, image_file_path, thresh=0.2, hier_thresh=0.1)
                r_ = darknet2widerface(r)
                save_widerface_bboxes(image_file_path, r_, save_dir )
    if mode == 'vis':
        for img_file in sorted(os.listdir(test_dir)):
            if '.jpg' not in img_file and '.jpeg' not in img_file and '.png' not in img_file and 'JPG' not in img_file:
                continue
            im_file=os.path.join(test_dir, img_file)
            r = detect(net, meta, im_file, thresh=0.2,hier_thresh=0.1)
            if SHOW or SAVE: im=cv2.imread(im_file)
            print(r)
            for i in r:
                score=i[1]
                c=list(i[2])
                c[0]-=c[2]/2
                c[1]-=c[3]/2
                #outfile.write('{} {:.5f} {:.2f} {:.2f} {:.2f} {:.2f}\n'.format(filename,score,c[0],c[1],c[2],c[3]))
                if SHOW or SAVE:
                    c = [int(j) for j in c]
                    if score>0.3:
                        cv2.rectangle(im, (c[0], c[1]), (c[0] + c[2], c[1] + c[3]), (0, 0, 255),4)
                    else:
                        cv2.rectangle(im, (c[0], c[1]), (c[0] + c[2], c[1] + c[3]), (255, 255, 0),3)
                    #cv2.putText(im,'{:.2f}'.format(score),(c[0],c[1]),1,2,(0,100,255),2)
            if len(r)!=0 and SHOW:
                cv2.imshow('x',im)
                cv2.waitKey()
            if SAVE:
                cv2.imwrite(os.path.join(save_dir,'%d.jpg'%cnt),im)
            cnt+=1
        #outfile.close()
        if SAVE:
            # to video
            # os.system('ffmpeg -f image2 -i {}/%d.jpg -vf scale=-1:360 -r 8 {}/{}.mp4'.format(save_dir,save_dir,test_dir[-1]))
            os.system('ffmpeg -f image2 -i {}/%d.jpg -r 8 {}/{}.mp4'.format(save_dir,save_dir,test_dir[-1]))

if __name__ == "__main__":
    main()
    
