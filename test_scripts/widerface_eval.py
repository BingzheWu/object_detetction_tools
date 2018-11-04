import os
import time
import numpy as np
import argparse
import functools
def darknet2widerface(bboxes):
    '''
    bboxes: [class_name, conf, (x, y, w, h)]
    '''
    res = []
    for bbox in bboxes:
        score = bbox[1]
        x, y, w, h = bbox[2]
        xmin = x - w/2.
        ymin = y - h/2.
        xmax = x + w/2.
        ymax = y + h/2. 
        res.append([xmin, ymin, xmax, ymax, score])
    res = np.array(res)
    return res

def save_widerface_bboxes(image_path, bboxes_scores, output_dir):
    """
    bboxes: layout is (xmin, ymin, xmax, ymax, score)
    """
    image_name = image_path.split('/')[-1]
    image_class = image_path.split('/')[-2]
    odir = os.path.join(output_dir, image_class)
    if not os.path.exists(odir):
        os.makedirs(odir)
    ofname = os.path.join(odir, '%s.txt' % (image_name[:-4]))
    f = open(ofname, 'w')
    f.write('{:s}\n'.format(image_class + '/' + image_name))
    f.write('{:d}\n'.format(bboxes_scores.shape[0]))
    for box_score in bboxes_scores:
        xmin, ymin, xmax, ymax, score = box_score
        f.write('{:.1f} {:.1f} {:.1f} {:.1f} {:.3f}\n'.format(xmin, ymin, (
            xmax - xmin + 1), (ymax - ymin + 1), score))
    f.close()
    print("The predicted result is saved as {}".format(ofname))