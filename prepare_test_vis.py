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


ind_to_class = {0: 'Atelectasis',
                1: 'Cardiomegaly',
                2: 'Effusion',
                3: 'Infiltrate',
                4: 'Mass/Nodule',
                5: 'Pneumonia',
                6: 'Pneumothorax'}

# images_for_inference = []
# with open(f'united_dataset/test_list.txt', 'r') as f: #united_dataset/test_list.txt
#     for line in f:
#         line = line[:-1]
#         lines = []
#         annot_name = line.split('/')[-1].replace('.png', '.txt')

#         if annot_name in os.listdir('united_dataset/labels'):
#             with open(f'united_dataset/labels/{annot_name}', 'r') as f2:
#                 for line2 in f2:
#                     # print(line2[0])
#                     if str(line2[0]) == '1': #['6', '3', '5', '2']:
#                         lines.append(line2)

#             # if len(lines)>5:
#             #     images_for_inference.append(line)

#             if len(lines)!=0:
#                 images_for_inference.append(line)

#             # if '6 ' in

# print(len(images_for_inference))
# print(images_for_inference)

images_for_inference = ['united_dataset/images/5714aea8b5a2d9b030196646842a6d47.png', 'united_dataset/images/e988f84d11f5903f930dd67fa55ce628.png',
                        'united_dataset/images/f90442f2620175fa21b6afeda865df11.png',
                        'united_dataset/images/8862ef75274d0c13bba19d7b5f2147c6.png', 'united_dataset/images/144b76b191aa1e02065903ee1cc3d578.png',
                        'united_dataset/images/c5ad7caaee32f3d5647a993ab299b1c1.png', 'united_dataset/images/11b3a0fe7f25bbe7643c60bcb14c35f5.png',
                        'united_dataset/images/c440f25216153cfc2bcf4af70c3d59c4.png', 'united_dataset/images/a537060564b5e08c80f46362deb565e8.png', # class 6
                        'united_dataset/images/eb384402d0dee9dc9818a0b65681bf67.png', 'united_dataset/images/5714aea8b5a2d9b030196646842a6d47.png', #'united_dataset/images/00020482_032.png', # class 3
                        'united_dataset/images/efc7bc78ce88e95191fdab525f974c24.png', #'united_dataset/images/00029464_015.png', # class 2
                        'united_dataset/images/00020986_000.png', 'united_dataset/images/a2511b7d5a4657b9b161d7c3e69587ae.png', # class 1
                        ]

if os.path.exists('test_visuals'):
    shutil.rmtree('test_visuals')
os.makedirs('test_visuals', exist_ok = True)

imgsz = 800

def main():
    project = f"yolov8s_base"

    inference_models = []
    for model_option in ['corrected_multi_800', 'corrected_binary_800', 'corrected_binary_class_1_800', 'corrected_binary_class_4_800']:
        exp_config = yaml.safe_load(open(f'{project}/{model_option}/args.yaml'))
        model = YOLO(f"{project}/{model_option}/weights/best.pt")
        inference_models.append(model)


    for inf_i, image_name in enumerate(images_for_inference):

        results_multi = inference_models[0](image_name, save=True, show=False, imgsz=imgsz, conf=0.2, iou=0.6, half=True, device='4', agnostic_nms=True)
        results_binary = inference_models[1](image_name, save=True, show=False, imgsz=imgsz, conf=0.2, iou=0.5, half=True, device='4')
        results_1 = inference_models[2](image_name, save=True, show=False, imgsz=imgsz, conf=0.2, iou=0.5, half=True, device='4')
        results_4 = inference_models[3](image_name, save=True, show=False, imgsz=imgsz, conf=0.03, iou=0.5, half=True, device='4')


        im_path = results_multi[0].path
        annot_name = im_path.split('/')[-1].replace('.png', '.txt')
        img = results_multi[0].orig_img.copy()
        height, width, channels = img.shape

        with open(f'united_dataset/labels/{annot_name}', 'r') as f:
            for line in f:
                coords = line[:-1].split(' ')
                class_name, coords = ind_to_class[int(coords[0])], coords[1:]
                coords = [float(c) for c in coords]
                x1,y1,x2,y2 = int(coords[0]*width - (coords[2]*width)/2), int(coords[1]*height - (coords[3]*height)/2), \
                            int(coords[0]*width + (coords[2]*width)/2), int(coords[1]*height+(coords[3]*height)/2)
                img = cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 7)

                (w, h), _ = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 3, 7)

                img = cv2.rectangle(img, (x1, y1 - h), (x1+w, y1), (255, 0, 0), -1)
                img = cv2.putText(img, class_name, (x1, y1 - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 7)

        (w, h), _ = cv2.getTextSize('Ground Truth', cv2.FONT_HERSHEY_SIMPLEX, 4, 7)
        img = cv2.rectangle(img, (int(img.shape[1]/2-w/2), int(h/2)), (int(img.shape[1]/2+w/2), int(1.5*h)), (255, 0, 0), -1)
        img = cv2.putText(img, 'Ground Truth', (int(img.shape[1]/2-w/2), int(1.5*h) - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 4, (255, 255, 255), 7)

        r_i_to_title = {0: 'Multi',1: 'Binary',2: 'Cardiomegaly only',3: 'Mass/Nodule only'}

        for r_i, r in enumerate([results_multi[0], results_binary[0], results_1[0], results_4[0]]):
            plotted_pred = r.plot()

            (w, h), _ = cv2.getTextSize(r_i_to_title[r_i], cv2.FONT_HERSHEY_SIMPLEX, 4, 7)
            plotted_pred = cv2.rectangle(plotted_pred, (int(plotted_pred.shape[1]/2-w/2), int(h/2)), (int(plotted_pred.shape[1]/2+w/2), int(1.5*h)), (255, 0, 0), -1)
            plotted_pred = cv2.putText(plotted_pred, r_i_to_title[r_i], (int(plotted_pred.shape[1]/2-w/2), int(1.5*h) - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 4, (255, 255, 255), 7)

            img = cv2.hconcat((img, plotted_pred))

        cv2.imwrite(f"test_visuals/results_{inf_i}.png", img)


if __name__=='__main__':
    main()