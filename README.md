# xray chest abnormalities detection

## Datasets

Two datasets were used for this work:

* [NHC](https://nihcc.app.box.com/v/ChestXray-NIHCC/folder/36938765345) is the initial one provided in the task. Here authors provided about 100k images for multilabel classification, where 1000 of them are annotated with bbox for each of 8 pathologies:

    - Atelectasis
    - Cardiomegaly
    - Effusion
    - Infiltrate
    - Mass
    - Nodule
    - Pneumonia
    - Pneumothorax

* [VinBigData Chest X-ray Abnormalities Detection](https://www.kaggle.com/competitions/vinbigdata-chest-xray-abnormalities-detection/data?select=train) is the additional dataset with similar domain and pathologies which was used to expand the initial data and improve model perfomance.

## Training on initial dataset

As a baseline approach for object detection task development the **YOLO** architecture were chosen. In particular, **YOLOV8** version were used as the state-of-the-art approach from the set of YOLO-based models.

In order to speedup the baseline model development process, the whole training procedure is build using [**ultralytics**](https://github.com/ultralytics/ultralytics/tree/main) library functionality and pretrained weights provided by authors.

### Data samples overview

As was described, initial dataset consists of 1000 images with bbox annotation, other images has only classification annotation, including "No findings" class, which can be used as background samples describing healthy patient cases. It was decided to expand 1000 annotated images with 2000 background images instead of 20k presented in order to save balance between background and other classes and speedup the training procedure because of limited calculation resources.

The **train/val/test** splits were made based on the unique patient ids with ratios of **0.8/0.1/0.1**.

As far as the dataset is dramatically small and presents 8 classes for detection, it was decided to use strong augmentations in order to increase possible data diversity. Here is the list of used augmentations:

- Hue
- Saturation
- Brightness
- Translation
- Rescaling
- Horizontal flips
- Mosaic
- Mixup
- Copy paste
- Erasing

Mosaic augmentation is desabled for last 15 epochs.

![Alt text](readme_images/init_multi_800/augmented_batch_init_data.jpg)

Here you can observe some data analytics. As we can see, the classes distribution is quite balanced but still we have too tiny number of samples per class for strong model development. Also, boxes diversity in terms of aspect ratio, sizes and location is quite high. Some of the classes present big objects, while others are mostly small, tall and even similar to each other.

| Labels | Correlogram |
| --- | --- |
| <img src="readme_images/init_multi_800/labels_distrib_init_data.jpg"> | <img src="readme_images/init_multi_800/labels_correlogram_init_data.jpg"> |


### Trainig parameters

Hyperparameters such as optimiser choice, lr, wd were chosen using the **ultralytics** auto mode. During experiments it was shown that auto mode provides quite good parameters. On each experiment **AdamW** optimiser has been chosen by **ultralytics**. Additionally, cosine lr sheduler were added to control the lr dynamic. 15 epochs warmup was used to adapt the model to work with new data and domain.

Initial images have quite high resolution, while most of the clases presented in the dataset are quite small. To be able to dectect small objects we need to use big image size.

The image size of **800** were chosen while batch size was **64**.

The model was trained for **100 epochs**.

**NMS** confidence score were chosen to be low (**0.03**) as far as we expect the model to be quite unsure in predictions while iou score were chosen as **0.8** to filter extra objects.

### Results

For better understanding, refer to [cometml experiment](https://www.comet.com/garsiyaeugene/yolov8s-base/1c29a7c2d275455f9722e3c5048377e5?compareXAxis=step&experiment-tab=panels&showOutliers=true&smoothing=0&xAxis=step)

![Alt text](readme_images/init_multi_800/results.png)

As was assumed, the model trained on such a small dataset for multiclass detection has a very low metrics level. Metrics are very oscillating which shows that it is quite difficult for the model to generalize the information and make a correct prediction. Also, considering the loss dynamic, we can see that val loss becomes growing up in the middle of the training, while train loss continue decreasing which could possibly mean the overfitting. Would be better to train more epochs to become sure if we have an overfitting or not but it is already clear that current task definition is too difficult for such small dataset.

| Matrix | F1 |
| --- | --- |
| <img src="readme_images/init_multi_800/confusion_matrix.png"> | <img src="readme_images/init_multi_800/F1_curve.png"> |

### Binary definition

To simplify the task for the model, we can try to switch from multiclass to binary definition and unite all labels under one class. We assume it must help network to perform better, but we must take into account that classes differ from each other quite much, so still it can be difficult for the network to generalise the information.

For better understanding, refer to [cometml experiment](https://www.comet.com/garsiyaeugene/yolov8s-base/d05e2cc64ecb47739a4e12873b66ca23?compareXAxis=step&experiment-tab=panels&showOutliers=true&smoothing=0&xAxis=step)

![Alt text](readme_images/init_binary_800/results.png)

We see that metrics and their dynamic are quite the same. Seems like corresponding data is not enough for the model to build strong correlations and provide accurate predictions.

| Matrix | F1 |
| --- | --- |
| <img src="readme_images/init_binary_800/confusion_matrix.png"> | <img src="readme_images/init_binary_800/F1_curve.png"> |


## Training on expanded dataset

### Data samples overview

In order to build more accurate model we need to have much more data. Using [VinBigData Chest X-ray Abnormalities Detection](https://www.kaggle.com/competitions/vinbigdata-chest-xray-abnormalities-detection/data?select=train) data we can expand our dataset for all classes except Pneumonia and also we can unite Mass and Nodule into one class (as they are presented united in new data). The new data also presents a lot of "No findings" images which can be used as background class but it was decided not to use them because of limited calculation resources.

The key feature of this dataset is the fact that authors provide annotation from different radiologists for each image. After brief analysis of annotations I decided that it will take some time to build a comprehensive and accurate solution to filter or unite annotations from several radiologists, thus, as a simple baseline solution the next logic were used: for each class, presented on the image, I've left annotation boxes only for the most frequent radiologist from all, presented on the current image. Training a model on full annotations without any cleaning has no sense as far as the annotation becomes a total mess that could confuse the model even stronger.

Here is a new list of classes:

- Atelectasis
- Cardiomegaly
- Effusion
- Infiltrate
- Mass/Nodule
- Pneumonia (all labels are from initial small dataset)
- Pneumothorax

![Alt text](readme_images/united_multi_800/train_batch1.jpg)

New data was also splitted to **train/val/test** modes and these splits were united with initial ones.

| Labels | Correlogram |
| --- | --- |
| <img src="readme_images/united_multi_800/labels.jpg"> | <img src="readme_images/united_multi_800/labels_correlogram.jpg"> |

In new expanded dataset we observe that labels became much more unbalanced but the absolute number of instances became much higher. That fact might help model to learn better correlations.

### Results

For better understanding, refer to [cometml experiment](https://www.comet.com/garsiyaeugene/yolov8s-base/54f2d4397bc845d785d349ce8f074fe0?compareXAxis=step&experiment-tab=panels&showOutliers=true&smoothing=0&xAxis=step)

![Alt text](readme_images/united_multi_800/results.png)

We see that metrics and their dynamic became better, moreover, metrics still have the increasing trend which means that we can continue trainig model for further epochs till the moment we face the plane dynamic of metrics which corresponds to overfitting.

| Matrix | F1 |
| --- | --- |
| <img src="readme_images/united_multi_800/confusion_matrix.png"> | <img src="readme_images/united_multi_800/F1_curve.png"> |

Lets also take a practical look at visualisations of predictions and compare them with labels.

Predictions are not ideal but look adequate. We can also note that we fail to find most of the small objects and sometimes find extra objects of major class.

| Labels | Predictions |
| --- | --- |
| <img src="readme_images/united_multi_800/val_batch0_labels.jpg"> | <img src="readme_images/united_multi_800/val_batch0_pred.jpg"> |

| Labels | Predictions |
| --- | --- |
| <img src="readme_images/united_multi_800/val_batch2_labels.jpg"> | <img src="readme_images/united_multi_800/val_batch2_pred.jpg"> |

### Binary definition

Lets also compare the binary definition trained on new dataset.

For better understanding, refer to [cometml experiment](https://www.comet.com/garsiyaeugene/yolov8s-base/58ec518bc6564fa28ab279856d310301?compareXAxis=step&experiment-tab=panels&showOutliers=true&smoothing=0&xAxis=step)

![Alt text](readme_images/united_binary_800/results.png)

Here the metrics become much better comparing to multiclass definition.

| Matrix | F1 |
| --- | --- |
| <img src="readme_images/united_binary_800/confusion_matrix.png"> | <img src="readme_images/united_binary_800/F1_curve.png"> |

Lets also take a practical look at visualisations of predictions and compare them with labels.

Predictions still are not ideal but look adequate. It's clear that model has begun to learn correlations much better comparing to the one trained on initial dataset.

| Labels | Predictions |
| --- | --- |
| <img src="readme_images/united_binary_800/val_batch0_labels.jpg"> | <img src="readme_images/united_binary_800/val_batch0_pred.jpg"> |


## NMS tuning

As was stated in the begining of trainig section, the NMS confidence value of **0.03** was used in assumption that it will be difficult for the model to be sure in predictions. This idea was clear while training on initial dataset. After switching to expanded dataset our model became better in terms of confedence predictions, that can be seen clearly from F1-confidence plots. That means that we can use higher confedence score for NMS which must help us to filter and clean predictions. I will also try to tune the iou value to rid off too overlapping predictions.


## Additional training for each class separately

As was described previously, each class presents a set of boxes with particular aspect ratio and sizes. It's pretty clear that it becomes difficult for the model to learn strong correlations for each class at once but it could be much simplier for the model to be trained only on one class.

Lets train 2 models for 2 major classes presented in the expanded dataset: Cardiomegaly (mostly rectangular boxes of big size) and Mass/Nodule (mostly square small boxes with wide location distribution on the image). We expect the model trained only for one class to show high metrics and quite accurate predictions.

### Cardiomegaly

![Alt text](readme_images/united_binary_Cardiomegaly_800/train_batch0.jpg)


| Labels | Correlogram |
| --- | --- |
| <img src="readme_images/united_binary_Cardiomegaly_800/labels.jpg"> | <img src="readme_images/united_binary_Cardiomegaly_800/labels_correlogram.jpg"> |


![Alt text](readme_images/united_binary_Cardiomegaly_800/results.png)

| Matrix | F1 |
| --- | --- |
| <img src="readme_images/united_binary_Cardiomegaly_800/confusion_matrix.png"> | <img src="readme_images/united_binary_Cardiomegaly_800/F1_curve.png"> |


| Labels | Predictions |
| --- | --- |
| <img src="readme_images/united_binary_Cardiomegaly_800/val_batch0_labels.jpg"> | <img src="readme_images/united_binary_Cardiomegaly_800/val_batch0_pred.jpg"> |


### Nodule/Mass

![Alt text](readme_images/united_binary_Mass_Nodule_800/train_batch1.jpg)

| Labels | Correlogram |
| --- | --- |
| <img src="readme_images/united_binary_Mass_Nodule_800/labels.jpg"> | <img src="readme_images/united_binary_Mass_Nodule_800/labels_correlogram.jpg"> |


![Alt text](readme_images/united_binary_Mass_Nodule_800/results.png)

| Matrix | F1 |
| --- | --- |
| <img src="readme_images/united_binary_Mass_Nodule_800/confusion_matrix.png"> | <img src="readme_images/united_binary_Mass_Nodule_800/F1_curve.png"> |


| Labels | Predictions |
| --- | --- |
| <img src="readme_images/united_binary_Mass_Nodule_800/val_batch2_labels.jpg"> | <img src="readme_images/united_binary_Mass_Nodule_800/val_batch2_pred.jpg"> |

We can conclude that even we switched to predict only one class for the network, we still fail to detect small objects even having a good number of samples. Possibly, chosen image size is too small for accurate detection of small objects for this task definition.

# Visualisations on test images

![Alt text](readme_images/test_vis/results_0.png)

![Alt text](readme_images/test_vis/results_2.png)

![Alt text](readme_images/test_vis/results_3.png)

![Alt text](readme_images/test_vis/results_4.png)

![Alt text](readme_images/test_vis/results_5.png)

![Alt text](readme_images/test_vis/results_6.png)

![Alt text](readme_images/test_vis/results_7.png)

![Alt text](readme_images/test_vis/results_15.png)

# Conclusions

The proof of concept for abnormalities detection on chest xrays were proven.

- It was shown that initial dataset size of 1000 images with bbox annotations is not enough for a strong model development.

- Increase of annotated images number can improve the models' quality quite much, even though, 4000 of annotated images still not enough for accurate model development. It is possible to work around with this data but more complex and sophisticated approaches are required.

- The task definition complexity also has quite a strong affect on training procedure. In our case the classes bounding boxes are very deverse and it becomes difficult for the model to build strong correlations for each class. It is always a nice idea to simplify the task if it is possible. In our case we can unite some similar classes like Mass and Nodule (better to have a medcial consultation on this point) or build a model for each class separately. The benifit of such solution was clearly shown previously

- Considering the new dataset the annotation cleaning and filtration means a lot. A deeper research is required on this point.

- Generally speaking, if pathologies localization is not a strong requirement for the task the switch from object detection to classification seems like a nice idea. Calssification task is simplier as far as there is no need to additionally predict a bbox. Moreover, we can find an access to bigger number of data as far as the classification annotation is much easier and faster.

# Possible improvements

## Easiest ways

- Increase the dataset (prooved)
- Develop model for each class separately (prooved)
- Clean and filter annotation and samples
- Switch from object detection to classification task
- Try to annotate the unlabeled images based on accurate classification model. For example, using GradCam to get an attention heatmap for each class and localize the objects based on heatmap intensity thresholding.
- Train on patches of images instead of initial images to detect small objects better
- Increase image size and batch size and train much more epochs

## Complex approaches (research is required)

- Smart xray preprocessing
- Comples pre and post-processing
- Usage of foundation models
- Detectors if we get enough data
- Few shot learning to workaround with limitted datasets
- Medical detectors and pretrained models (smth similar to MedSam)


# Demo

To launch locally:

```
cd xray_chest_detection
python3 -m venv env
source env/bin/activate
pip3 install -r requirements.txt
python3 demo.py
```

The app will be available locally on http://127.0.0.1:8080

Just upload an X-Ray chest image and the app will show you detected abnormalities based on 4 models described above. Or just use test images from `images_for_inference`.

The app is also available online [here](https://xray-detection-426613.ew.r.appspot.com)

![Alt text](readme_images/demo/home_page.png)

![Alt text](readme_images/demo/detection.png)