# object_detetction_tools
This repository has collected some useful scripts for object detection, including
model transfomer for various framework (e.g darknet2caffe), metric calculator (e.g map).


## nn_model_transformer

This module supports the transformation of the detection models trained within different framework.
Currently, we have implemented the following features:
* yolov1-caffe
* yolov2-caffe
* yolov3-caffe

#### yolo-caffe
Now our tool can be used for the transformation from arbitrary Yolo version to Caffemodel.
Note that you can directly use the **yolov1(v2)-caffe** with the original Caffe version. If you want to
use **yolov3-caffe**, you must add the upsample-layer into the original Caffe and recompile it.
You can find the source code of the upsample-layer in the 
[extra_caffe_layers](https://github.com/BingzheWu/object_detetction_tools/tree/master/extra_caffe_layers) dir.
You can get more information in [add new layers](https://github.com/BVLC/caffe/wiki/Development).
