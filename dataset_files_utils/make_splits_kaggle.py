import pandas as pd
import os
import numpy as np
import cv2
from tqdm import tqdm
import pydicom as dicom
from PIL import Image
import matplotlib.pyplot as plt
from tqdm.contrib.concurrent import process_map
import json
from itertools import combinations


def get_iou(bb1, bb2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Parameters
    ----------
    bb1 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x1, y1) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
    bb2 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x, y) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner

    Returns
    -------
    float
        in [0, 1]
    """
    assert bb1['x1'] < bb1['x2']
    assert bb1['y1'] < bb1['y2']
    assert bb2['x1'] < bb2['x2']
    assert bb2['y1'] < bb2['y2']

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
    bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou


annot = pd.read_csv('xray_data/annotation/kaggle_annot.csv')
print(len(annot.index), annot['image_id'].nunique())

# 1 - Atelectasis
# 3 - Cardiomegaly
# 10 - Pleural effusion
# 6 - Infiltration
# 8 - Nodule/Mass
# 12 - Pneumothorax

# 0: Atelectasis
# 1: Cardiomegaly
# 2: Effusion
# 3: Infiltrate
# 4: Mass/Nodule
# 5: Pneumonia
# 6: Pneumothorax

image_path = 'kaggle_data/train'
os.makedirs('kaggle_data/images', exist_ok=True)
old_to_new_class_mapping = {1:0, 3:1, 10:2, 6: 3, 8:4, 12:6}
labels_dict = {}

labels_save_fold = 'labels_cleaned'

os.makedirs(f'kaggle_data/{labels_save_fold}', exist_ok=True)
annot = annot[annot['class_id'].isin([1,3,6,8,10,12,])] #14

def parse(row):
    class_id = int(row['class_id'])
    image_id = row['image_id']

    # img_dicom = dicom.dcmread(f"{image_path}/{image_id}.dicom")
    # img = img_dicom.pixel_array
    # if img.max() != img.min():
    #     img = (((img - img.min()) / (img.max() - img.min())) * np.iinfo(np.uint16).max).astype(np.uint16)
    # height, width = img.shape
    # img = Image.fromarray(img)
    # img.save(f"kaggle_data/images/{image_id}.png")

    if class_id in [1,3,6,8,10,12]:
        img = cv2.imread(f"kaggle_data/images/{image_id}.png")

        rad_id = row['rad_id']
        x, y, w, h = row['x_min'], row['y_min'], row['x_max']-row['x_min'], row['y_max']-row['y_min']

        # img_dicom = dicom.dcmread(f"{image_path}/{image_id}.dicom")
        # img = img_dicom.pixel_array
        # if img.max() != img.min():
        #     img = (((img - img.min()) / (img.max() - img.min())) * np.iinfo(np.uint16).max).astype(np.uint16)
        # img = Image.fromarray(img)
        # img.save(f"kaggle_data/images/{image_id}.png")

        # img = cv2.imread(f"kaggle_data/images/{image_id}.png")
        # img = cv2.rectangle(img, (int(x), int(y)), (int(x+w), int(y+h)), (255, 0, 0), 2)
        # img = cv2.putText(img, 'init', (1000,1000), cv2.FONT_HERSHEY_SIMPLEX,
        #            5, (255, 0, 0), 2, cv2.LINE_AA)
        height, width, chan = img.shape
        # print(img.shape)
        x_c, y_c, w_norm, h_norm = (x+w/2)/width , (y+h/2)/height, w/width, h/height
        # img = cv2.rectangle(img, (int((x_c-w_norm/2)*width), int((y_c-h_norm/2)*height)),
        #                         (int((x_c+w_norm/2)*width), int((y_c+h_norm/2)*height)), (0, 255, 0), 2)
        # img = cv2.putText(img, 'norm', (1000,1000), cv2.FONT_HERSHEY_SIMPLEX,
        #            5, (0, 255, 0), 2, cv2.LINE_AA)
        # cv2.imwrite('kaggle_val.png', img)
        # br

        # if image_id not in labels_dict:
        #     labels_dict[image_id] = []

        # labels_dict[image_id].append(f"{old_to_new_class_mapping[class_id]} {x_c} {y_c} {w_norm} {h_norm}")
        # label_mapped =
        # if int(old_to_new_class_mapping[class_id]) in [5,6,7]:

        return (image_id, rad_id, f"{old_to_new_class_mapping[class_id]} {x_c} {y_c} {w_norm} {h_norm}", f"{old_to_new_class_mapping[class_id]} {x} {y} {w} {h}")
    elif class_id == 14:
        return (image_id, None, 14, 14)
        # labels_dict[image_id] = 14

# rows_to_process = []
# for _, row in tqdm(annot.iterrows(), total=len(annot.index)):
#     rows_to_process.append(row)

# results = process_map(parse, rows_to_process, max_workers=20, chunksize=40)

# for res in results:
#     image_id, rad_id, str_row, init_coords = res

#     if image_id not in labels_dict:
#         labels_dict[image_id] = []

#     if str_row != 14:
#         labels_dict[image_id].append([rad_id, str_row, init_coords])

# with open('kaggle_labels_dict.json', 'w', encoding='utf-8') as f:
#     json.dump(labels_dict, f, ensure_ascii=False, indent=4)

labels_dict = json.load(open('kaggle_labels_dict.json'))

labels_dict_new = {}

for image_id in tqdm(labels_dict):

    labels_dict_new[image_id] = []
    # print(image_id)
    labels_presented = {}

    # img = cv2.imread(f"kaggle_data/images/{image_id}.png")
    # img_2 = img.copy()
    # height, width, chan = img.shape

    # print(labels_dict[image_id])
    for id_row, (rad_id, annot_row, init_coords_row) in enumerate(labels_dict[image_id]):
        label, x,y,w,h = init_coords_row.split(' ')
        x,y,w,h = float(x), float(y), float(w), float(h)

        if label not in labels_presented:
            labels_presented[label] = []

        labels_presented[label].append([id_row, rad_id, [x,y,w,h], annot_row])

        # img = cv2.rectangle(img, (int(x), int(y)), (int(x+w), int(y+h)), (255, 0, 0), 2)
        # img = cv2.putText(img, f"{label}_{rad_id}_{id_row}", (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX,
        #         3, (255, 0, 0), 2, cv2.LINE_AA)

    filtered_rows = []
    for label in labels_presented:
        boxes_to_average = []
        boxes_ok = []

        unique, counts = np.unique([item[1] for item in labels_presented[label]], return_counts=True)
        chosen_rad = unique[np.argmax(counts)]

        # print(dict(zip(unique, counts)))
        # print(chosen_rad)

        # print(label, set([item[1] for item in labels_presented[label]]))
        # print(chosen_rad)
        # print(len(labels_presented[label]))

        chosen_dets = [item for item in labels_presented[label] if item[1]==chosen_rad]

        chosen_dets = [item[-1] for item in labels_presented[label] if item[1]==chosen_rad]

        filtered_rows+=chosen_dets
        # print(len(chosen_dets))


    labels_dict_new[image_id] = filtered_rows
    if len(labels_dict[image_id])>5:
        print(image_id)
        print(len(labels_dict[image_id]))
        print(len(labels_dict_new[image_id]))
        # for combo in combinations(labels_presented[label], 2):
        #     id_row_first, rad_id_first, coords_first, annot_row_first = combo[0]
        #     id_row_second, rad_id_second, coords_second, annot_row_second = combo[1]

        #     box_first = {'x1': coords_first[0], 'x2': coords_first[0]+coords_first[2], 'y1':coords_first[1], 'y2':coords_first[1]+coords_first[3]}
        #     box_second = {'x1': coords_second[0], 'x2': coords_second[0]+coords_second[2], 'y1':coords_second[1], 'y2':coords_second[1]+coords_second[3]}
        #     iou_v = get_iou(box_first, box_second)

        #     print(iou_v, id_row_first, id_row_second)

        #     if iou_v>0.3:
        #         boxes_to_average.append(id_row_first)
        #         boxes_to_average.append(id_row_second)
        #     else:
        #         boxes_ok.append(id_row_first)
        #         boxes_ok.append(id_row_second)

        # if len(boxes_ok)>1:
        #     print(set(boxes_to_average))
        #     print(set(boxes_ok))

        # boxes_to_average = set(boxes_to_average)
        # boxes_ok = set(boxes_ok)

        # if len(boxes_to_average)!=0:
        #     coords_per_box = [info[-1].split(' ')[1:] for info in labels_presented[label] if info[0] in boxes_to_average]
        #     x_c_list = [float(c[0]) for c in coords_per_box]
        #     y_c_list = [float(c[1]) for c in coords_per_box]
        #     w_list = [float(c[2]) for c in coords_per_box]
        #     h_list = [float(c[3]) for c in coords_per_box]

        #     x_c_new = float(np.mean(x_c_list))
        #     y_c_new = float(np.mean(y_c_list))
        #     w_new = float(np.mean(w_list))
        #     h_new = float(np.mean(h_list))

        #     img = cv2.rectangle(img, (int((x_c_new-w_new/2)*width), int((y_c_new-h_new/2)*height)),
        #                             (int((x_c_new+w_new/2)*width), int((y_c_new+h_new/2)*height)), (0, 255, 0), 2)
    # print(filtered_rows)
    # len(chosen_dets)
    # if len(labels_dict[image_id])>10:
    #     for chosen in filtered_rows:
    #         id_row, rad_id = chosen[0], chosen[1]
    #         label = chosen[-1][0]
    #         x,y,w,h = chosen[2]
    #         img_2 = cv2.rectangle(img_2, (int(x), int(y)), (int(x+w), int(y+h)), (255, 0, 0), 2)
    #         img_2 = cv2.putText(img_2, f"{label}_{rad_id}_{id_row}", (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX,
    #                 3, (255, 0, 0), 2, cv2.LINE_AA)
    #     cv2.imwrite('kaggle_val.png', img)
    #     cv2.imwrite('kaggle_val_2.png', img_2)
    #     br






for image_id in labels_dict_new:
    if labels_dict_new[image_id] != 14:
        with open(f'kaggle_data/{labels_save_fold}/{image_id}.txt', 'w') as f:
            for line in labels_dict_new[image_id]:
                f.write(f"{line}\n")


# patients = list(labels_dict.keys())
# train_patients = patients.copy()
# val_patients = list(np.random.choice(patients, size=int(0.1*len(patients)), replace=False))

# for pat in val_patients:
#     train_patients.remove(pat)

# test_patients = list(np.random.choice(train_patients, size=int(0.1*len(patients)), replace=False))

# for pat in test_patients:
#     train_patients.remove(pat)

# print(len(train_patients), len(val_patients), len(test_patients), len(patients))


# with open('kaggle_data/train_list.txt', 'w') as f:
#     for line in train_patients:
#         f.write(f"kaggle_data/images/{line}.png\n")

# with open('kaggle_data/val_list.txt', 'w') as f:
#     for line in val_patients:
#         f.write(f"kaggle_data/images/{line}.png\n")

# with open('kaggle_data/test_list.txt', 'w') as f:
#     for line in test_patients:
#         f.write(f"kaggle_data/images/{line}.png\n")