#! /bin/sh
python yolo_main.py -m prototxt/yolo_tiny_deploy.prototxt -w yolo_tiny.caffemodel -i images/dog.jpg
