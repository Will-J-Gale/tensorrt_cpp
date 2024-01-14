# TensorRT CPP
Simple engine builder and tensorrt inference programs

## Setup
* `build.sh`
* `build/tensorrt_build <PATH_TO_ONNX_FILE>`
* `build/tensorrt_inference_image <PATH_TO_ENGINE_FILE> <PATH_TO_IMAGE>`
* `build/tensorrt_inference_video <PATH_TO_ENGINE_FILE> <VIDEO_SOURCE>`

## tensorrt_build
* This program turns an `.onnx` model into an `.engine`

## tensorrt_inference_image
* Runs inference on an image, shows the inference time, and draws boxes for detections.

## tensorrt_inference_video
* Runs inference on an video, shows the inference time, and draws boxes for detections.
* VIDEO_SOURCE can be a video path or camera index
