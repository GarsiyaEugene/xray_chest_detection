import json
import pandas as pd
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
import os
import matplotlib.pyplot as plt
import cv2


annot = pd.read_csv('xray_data/annotation/initial_annot.csv')
info = pd.read_csv('xray_data/annotation/initial_info.csv')
categories_list = []
# categories_map = {}
# for label_i, label in enumerate(annot['Finding Label'].unique()):
#     categories_list.append({"supercategory": label,
#                             "id": label_i,
#                             "name": label}
#                                 )
#     categories_map[label] = label_i

categories_map = {'Atelectasis': 0,
                 'Cardiomegaly': 1,
                  'Effusion': 2,
                 'Infiltrate': 3,
                'Mass': 4,
                'Nodule': 4,
                'Pneumonia': 5,
                'Pneumothorax': 6}
print(categories_map)

def parse_line(line):
    img = line[:-1] if '\n' in line else line
    img_name = img.split('/')[-1]
    img_annot = annot[annot['Image Index']==img_name]
    img_info = info[info['Image Index']==img_name]

    img_finding = img_info['Finding Labels'].values[0]

    images_dict = {}
    annotations_dict = []

    if len(img_annot.index)>=1 or img_finding == 'No Finding':
        images_dict = {
                    "file_name": img_name,
                    "height": int(img_info['Height]'].values[0]),
                    "width": int(img_info['OriginalImage[Width'].values[0]),
                    "id": None
                            }

        for _, row in img_annot.iterrows():
            annotations_dict.append(
                            {
                            "image_id": None,
                            "bbox": [float(row['Bbox [x']), float(row['y']), float(row['w']), float(row['h]'])],
                            "category_id": int(categories_map[row['Finding Label']]),
                            "id": None
                                })

    return images_dict, annotations_dict


if __name__ == '__main__':

    os.makedirs('xray_data/labels', exist_ok = True)

    for mode in ['test', 'train', 'val']:

        with open(f'xray_data/{mode}_list.txt', 'r') as f:
            lines = [line for line in f]

        print(len(set([line[:-1] for line in lines]).intersection(annot['Image Index'].unique())))
        print(len(lines), len(annot['Image Index'].unique()))

        results = process_map(parse_line,lines, max_workers=8, chunksize=100)

        n_empty = 0
        n_label_with_no_box = 0
        non_empty_images = []
        for res in tqdm(results):
            images_dict, annot_list = res[0], res[1]
            if len(images_dict)!=0:
                if len(annot_list)!=0:
                    non_empty_images.append(images_dict['file_name'])
                    img = cv2.imread(f"xray_data/images/{images_dict['file_name']}")
                    height, width, channels = img.shape
                    with open(f"xray_data/labels/{images_dict['file_name'].replace('.png','')}.txt", 'w') as f:
                        for annot_item in annot_list:
                            box_str = f"{(annot_item['bbox'][0]+annot_item['bbox'][2]/2)/width} {(annot_item['bbox'][1]+annot_item['bbox'][3]/2)/height} {annot_item['bbox'][2]/width} {annot_item['bbox'][3]/height}"
                            f.write(f"{annot_item['category_id']} {box_str}\n")
                            print(f"{annot_item['category_id']} {box_str}\n")
                            # coords = [float(item) for item in box_str.split(' ')]
                            # print(coords)
                            # print(images_dict['width'], images_dict['height'])
                            # print(images_dict['file_name'])
                            # img = cv2.rectangle(img, (int(annot_item['bbox'][0]), int(annot_item['bbox'][1])), (int(annot_item['bbox'][0]+annot_item['bbox'][2]), int(annot_item['bbox'][1]+annot_item['bbox'][3])), (255, 0, 0), 2)
                            # # img = cv2.rectangle(img, (int(annot_item['bbox'][0]), int(annot_item['bbox'][1])), (int(annot_item['bbox'][0]+annot_item['bbox'][2]), int(annot_item['bbox'][1]+annot_item['bbox'][3])), (0, 255, 0), 2)
                            # img = cv2.rectangle(img, (int(coords[0]*width - (coords[2]*width)/2), int(coords[1]*height - (coords[3]*height)/2)), (int(coords[0]*width + (coords[2]*width)/2), int(coords[1]*height+(coords[3]*height)/2)), (0, 255, 0), 2)
                            # plt.imshow(img)
                            # # plt.plot(int(coords[0]*images_dict['width']), int(coords[1]*images_dict['height']), 'r*')
                            # # plt.plot(int(coords[0]*images_dict['width'] + (coords[2]*images_dict['width'])/2), int(coords[1]*images_dict['height'] + (coords[3]*images_dict['height'])/2), 'r*')
                            # plt.savefig('val.png')
                            # br
                        # print(f)
                        print('________')
                else:
                    n_empty += 1

            else:
                n_label_with_no_box+=1

        with open(f'xray_data/{mode}_list_non_empty.txt', 'w') as f:
            for line in non_empty_images:
                f.write(f"xray_data/images/{line}\n")

        print(n_empty, n_label_with_no_box)