import yaml
import comet_ml
from ultralytics import YOLO
from ultralytics import settings
import shutil



def main():
    train_config = yaml.safe_load(open('configs/train_config.yaml'))

    project = f"{train_config['architecture']}_base"

    settings.reset()
    settings.update({"datasets_dir": "/storage_research/garsiya.evgeniy/xray_chest_detection", "tensorboard": False})

    comet_ml.init(project_name=project)
    model = YOLO(f"{train_config['architecture']}.pt")

    results = model.train(data="configs/xray_dataset.yaml", epochs=train_config['epochs'], imgsz=train_config['imgsz'],
                          single_cls=train_config['single_cls'], project=project,
                        conf=train_config['conf'], iou=train_config['iou'],
                        half=True, amp=False,
                        plots=True, save_period=1, device="5",
                        cls = 1.5, resume=train_config['resume'],

                        optimizer=train_config['optimizer'], lr0=train_config['lr'], weight_decay=train_config['wd'],
                        cos_lr=train_config['cos_lr'], warmup_epochs=train_config['warmup_epochs'],

                        show=True, save_json=True, save_hybrid=train_config['save_hybrid'], batch=train_config['batch'], mosaic=train_config['mosaic'],
                        mixup=train_config['mixup'], erasing=train_config['erasing'],
                        crop_fraction=train_config['crop_fraction'], copy_paste=train_config['copy_paste'])

if __name__=='__main__':
    main()