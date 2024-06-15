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


def main():
    train_config = yaml.safe_load(open('configs/train_config.yaml'))

    project = f"{train_config['architecture']}_base"

    settings.reset()
    settings.update({"datasets_dir": "/storage_research/garsiya.evgeniy/xray_chest_detection", "tensorboard": False})

    folder = 'corrected_binary_class_4_800' #corrected_multi_800
    exp_config = yaml.safe_load(open(f'{project}/{folder}/args.yaml'))
    model = YOLO(f"{project}/{folder}/weights/best.pt")

    # Val
    tuning_res = {}
    for conf_score in [exp_config['conf'], 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:

        for iou_score in [0.1, 0.3, 0.5, 0.6, 0.8]:

            results_val = model.val(data="configs/xray_dataset.yaml", imgsz=exp_config['imgsz'], single_cls=exp_config['single_cls'],
                                project=project, name=f"{folder}_conf_{conf_score}_iou_{iou_score}",
                                conf=conf_score, iou=iou_score, half=True, amp=False, split='val',
                                plots=True, save_period=1, device="4",
                                show=True, save_json=True, save_hybrid = False, batch=exp_config['batch'])

            map50 = results_val.results_dict['metrics/mAP50(B)']
            rec = results_val.results_dict['metrics/recall(B)']
            mean_metric = np.mean((rec, map50))

            tuning_res[f"conf_{conf_score}_iou_{iou_score}"] = {'res': results_val.results_dict, 'mean_r_map': mean_metric}

            with open('tuning_res.json', 'w', encoding='utf-8') as f:
                json.dump(tuning_res, f, ensure_ascii=False, indent=4)

    tuning_res = dict(sorted(tuning_res.items(), key=lambda item: float(item[1]['mean_r_map']), reverse=True))

    print('Best NMS')
    print(list(tuning_res.items())[0])

    with open('tuning_res_sorted.json', 'w', encoding='utf-8') as f:
        json.dump(tuning_res, f, ensure_ascii=False, indent=4)

if __name__=='__main__':
    main()