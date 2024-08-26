# TensorRT CPP
Simple engine builder and tensorrt inference programs

## Setup
* `build.sh`

## tensorrt_build
* This program converts an `.onnx` model into an `.engine`
* `build/tensorrt_build <PATH_TO_ONNX_FILE>`

## tensorrt_inference_image
* Runs inference on an image, shows the inference time, and draws boxes for detections.
* `build/tensorrt_inference_image <PATH_TO_ENGINE_FILE> <PATH_TO_IMAGE>`

## tensorrt_inference_video
* Runs inference on an video, shows the inference time, and draws boxes for detections.
* `build/tensorrt_inference_video <PATH_TO_ENGINE_FILE> <VIDEO_SOURCE>`
    * VIDEO_SOURCE can be a video path or camera index
