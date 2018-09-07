# object_detetction_tools
This repository has collected some useful scripts for object detection, including
model transfomer for various framework (e.g darknet2caffe), metric calculator (e.g map).


## nn_model_transformer

This module supports the transformation of the detection models trained within different framework.
Currently, we have implemented the following features:
* Yolov1-Caffe
* Yolov2-Caffe
* Yolov3-caffe

### Yolo-Caffe
Now our tool can be used for the transformation from arbitrary Yolo version to Caffemodel.
Note that you can use directly the $Yolov1(v2)-Caffe$ with the original Caffe version. If you want to
use $Yolov3-Caffe$, you must add the upsample-layer into the original Caffe and recompile it.
You can found the source code of the upsample-layer in the [extra_caffe_layers] dir.  