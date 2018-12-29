import numpy as np


def box_cluster_iou(box, cluster):
    """
    calculate the iou between a given box and the boxes in the cluster
    input:
        box: tuple or array (w,h)
        cluster: numpy array with a shape of [k, 2]
    return:
        a numpy array with a shape of [k, 0]
    """
    x = np.minimum(cluster[:, 0], box[0])
    y = np.minimum(cluster[:, 1], box[1])
    if np.count_nonzero(x == 0) > 0 or np.count_nonzero(y==0) > 0:
        raise ValueError("Box has area with zero")
    intersection = x * y
    box_area = box[0] * box[1]
    cluster_area = cluster[:, 0] * cluster[:, 1]
    ious = intersection / (box_area + cluster_area - intersection)
    return ious
def avg_iou(boxes, cluster):
    return np.mean([np.max(box_cluster_iou(boxes[i], cluster)) for i in range(boxes.shape[0])])

def traslate_boxes(boxes):
    """
    translate all the boxes to the tuple comprises width and height
    the boxes is a numpy array with the format [x0, y0, x3, y3]
    """
    new_boxes = boxes.copy()
    for row in range(new_boxes.shape[0]):
        new_boxes[row][2] = np.abs(new_boxes[row][2]-new_boxes[row][0])
        new_boxes[row][3] = np.abs(new_boxes[row][3]-new_boxes[row][1])
    return np.delete(new_boxes, [0,1], axis = 1)
def kmeans(boxes, k, dist = np.median):
    
    rows = boxes.shape[0]
    distances = np.empty((rows, k))
    last_cluster = np.zeros((rows, ))
    np.random.seed()
    cluster = boxes[np.random.choice(rows, k, replace = False)]

    while True:
        for row in range(rows):
            distances[row] = 1 - box_cluster_iou(boxes[row], cluster)
        nearest_cluster = np.argmin(distances, axis = 1)
        if (last_cluster == nearest_cluster).all():
            break
        for i in range(k):
            cluster[i] = dist(boxes[nearest_cluster == i], axis = 0)
        last_cluster = nearest_cluster
    return cluster