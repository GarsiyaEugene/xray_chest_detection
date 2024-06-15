import pydicom as dicom
import cv2
import os
from tqdm import tqdm


image_path = 'kaggle_data/train'
for dicom_im in tqdm(os.listdir(image_path)):
    ds = dicom.dcmread(f"{image_path}/{dicom_im}")
    pixel_array_numpy = ds.pixel_array
    cv2.imwrite(f"kaggle_data/images/{dicom_im.split('.')[0]}.png", pixel_array_numpy)