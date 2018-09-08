import time
import torch
def get_region_boxes(output, conf_thresh, num_classes, anchors, num_anchors, only_objectness=1, validation=False):
    anchor_step = len(anchors)/num_anchors
    if output.dim() == 3:
        output = output.unsqueeze(0)
    batch = output.size(0)
    assert(output.size(1) == (5+num_classes)*num_anchors)
    h = output.size(2)
    w = output.size(3)

    t0 = time.time()
    all_boxes = []
    output = output.view(batch*num_anchors, 5+num_classes, h*w).transpose(0,1).contiguous().view(5+num_classes, batch*num_anchors*h*w)

    grid_x = torch.linspace(0, w-1, w).repeat(h,1).repeat(batch*num_anchors, 1, 1).view(batch*num_anchors*h*w).type_as(output) #cuda()
    grid_y = torch.linspace(0, h-1, h).repeat(w,1).t().repeat(batch*num_anchors, 1, 1).view(batch*num_anchors*h*w).type_as(output) #cuda()
    xs = torch.sigmoid(output[0]) + grid_x
    ys = torch.sigmoid(output[1]) + grid_y

    anchor_w = torch.Tensor(anchors).view(num_anchors, anchor_step).index_select(1, torch.LongTensor([0]))
    anchor_h = torch.Tensor(anchors).view(num_anchors, anchor_step).index_select(1, torch.LongTensor([1]))
    anchor_w = anchor_w.repeat(batch, 1).repeat(1, 1, h*w).view(batch*num_anchors*h*w).type_as(output) #cuda()
    anchor_h = anchor_h.repeat(batch, 1).repeat(1, 1, h*w).view(batch*num_anchors*h*w).type_as(output) #cuda()
    ws = torch.exp(output[2]) * anchor_w
    hs = torch.exp(output[3]) * anchor_h

    det_confs = torch.sigmoid(output[4])

    cls_confs = torch.nn.Softmax()(output[5:5+num_classes].transpose(0,1)).data
    cls_max_confs, cls_max_ids = torch.max(cls_confs, 1)
    cls_max_confs = cls_max_confs.view(-1)
    cls_max_ids = cls_max_ids.view(-1)
    t1 = time.time()
    
    sz_hw = h*w
    sz_hwa = sz_hw*num_anchors
    if validation:
        cls_confs = cls_confs.view(-1, num_classes)
    t2 = time.time()
    for b in range(batch):
        boxes = []
        for cy in range(h):
            for cx in range(w):
                for i in range(num_anchors):
                    ind = b*sz_hwa + i*sz_hw + cy*w + cx
                    det_conf =  det_confs[ind]
                    if only_objectness:
                        conf =  det_confs[ind]
                    else:
                        conf = det_confs[ind] * cls_max_confs[ind]
    
                    if conf > conf_thresh:
                        bcx = xs[ind]
                        bcy = ys[ind]
                        bw = ws[ind]
                        bh = hs[ind]
                        cls_max_conf = cls_max_confs[ind]
                        cls_max_id = cls_max_ids[ind]
                        box = [bcx/w, bcy/h, bw/w, bh/h, det_conf, cls_max_conf, cls_max_id]
                        if (not only_objectness) and validation:
                            for c in range(num_classes):
                                tmp_conf = cls_confs[ind][c]
                                if c != cls_max_id and det_confs[ind]*tmp_conf > conf_thresh:
                                    box.append(tmp_conf)
                                    box.append(c)
                        boxes.append(box)
        all_boxes.append(boxes)
    t3 = time.time()
    if False:
        print('---------------------------------')
        print('matrix computation : %f' % (t1-t0))
        print('        gpu to cpu : %f' % (t2-t1))
        print('      boxes filter : %f' % (t3-t2))
        print('---------------------------------')
    return all_boxes