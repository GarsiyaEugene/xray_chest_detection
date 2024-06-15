import yaml
import comet_ml
from ultralytics import YOLO
from ultralytics import settings
from PIL import Image
import cv2
import os
import shutil
import json
import numpy as np

imgsz = 800
model_options = {'multiclass': {'checkpoint': 'checkpoints/binary.pt', 'conf':, 'iou'}, 'binary': , 'cardiomegaly':, 'mass'}
def main():
    for model_name in model_options:
        model = YOLO(model_options[model_name]['checkpoint'])
        results = model(images_for_inference, imgsz=imgsz, conf=0.03, iou=0.1, half=True, device='cpu')