# 2020 AI Hackathon
This repository uses MaskRCNN to detect car damage.

## MaskRCNN Training
* If you have any questions for training MaskRCNN models, please click [this](https://github.com/matterport/Mask_RCNN).

## OpenCV Library
* This repository uses [OpenCV 4.4.0](https://opencv.org/opencv-4-4-0/) and [Microsoft Visual Studio 2019](https://visualstudio.microsoft.com/).
* To enable OpenCL for OpenCV, please enable ```WITH_OPENCL``` when you configure cmake files.

## (Optional) OpenVino Enviroment
* You can install it [here](https://software.intel.com/content/www/us/en/develop/tools/openvino-toolkit.html).
* To use OpenCV backend ```DNN_BACKEND_INFERENCE_ENGINE```, please use [cmake](https://cmake.org/) and download the entire [OpenCV](https://opencv.org/opencv-4-4-0/) source code to enable ```WITH_INF_ENGINE```.

## Notes
### 0_Dataset
* Car damage dataset.
* Each image with one JSON file will record the object's segmentation.
### 1_Maskrcnn_Segmentation
#### Program arugments
```
{help | | Print help message. }
{weight | | Path to a binary file of model contains trained weights. 
    It could be a file with extensions .caffemodel (Caffe), .pb (TensorFlow), .t7 or .net (Torch), .weights (Darknet), .bin (OpenVINO). }
{graph | | Path to a text file of model contains network configuration, 
    It could be a file with extensions .prototxt (Caffe), .pbtxt (TensorFlow), .cfg (Darknet), .xml (OpenVINO). }
{classes | | Path to a text file with names of classes to label detected objects. }
{colors | | Path to a text file of indication that model works with RGB input images instead BGR ones (this is for segmenttation). }
{scale | 1.0 | (OpenCV and OpenVino Platform) Preprocess input image by multiplying on a scale factor. }
{image |<none>| Path to a image file for input data. }
{video |<none>| Path to a video file for input data. }
{camera |<none>| Use camera's frame for input data. }
{conf | 0.7 | Confidence threshold. }
{mask | 0.3 | Non-maximum suppression threshold. }
{output_file | | Path to output file name. }
{backend | OpenCV | Can be set OpenCV or OpenVino. }
{target | CPU | (Optional) Choose one of target computation devices: 
    CPU: CPU target (by default), 
    GPU: GPU, 
    GPU16: GPU using half-float precision. }
{outlayer_names | | (Optional) Force to set output layer's name (use ',' to seperate). }
```
#### Pre-trained models
* Tensorflow format (frozen_inference_graph.pb, maskrcnn_inception_20201008.pbtxt).
* Inference Model Optimizer FP16 (CAR_16.xml, CAR_16.bin)
* Inference Model Optimizer FP32 (CAR_32.xml, CAR_32.bin

### 2_WebApiServer
#### Method
* CarDamageDetection
```
POST /api/CarDamageDetection
HOST: {your host name}
Content-Type: application/json

ImageData={any image format (ex. *.png, *.jpg, *.bmp, ...) from base64 encode}
```
Output:

|ImageData|X|Y|W|H|Label|Score|
|-|-|-|-|-|-|-|
|Output image (.jpg format)|Top-Left X|Top-Left Y|Object rect width|Object rect height|Label name|The probability of classification|
|```String```|```int```|```int```|```int```|```int```|```string```|```double```|

Data structure in C#:
```C
public class ObjectDetectionDetail
{
    public int X { get; set; }
    public int Y { get; set; }
    public int W { get; set; }
    public int H { get; set; }
    public string Label { get; set; }
    public double Score { get; set; }
} // end class

public class OutputObjectDetection
{
    public List<ObjectDetectionDetail> ObjectDetail { get; set; }
    public string ResultImageData;
} // end class
```