import os
import shutil
import numpy as np

os.makedirs('united_dataset/labels_cleaned', exist_ok = True)


for mode in ['train', 'val', 'test']:
    # total_lines = []
    # with open(f'kaggle_data/{mode}_list.txt', 'r') as f:
    #     for line in f:
    #         total_lines.append(line.replace('kaggle_data','united_dataset'))
    #         # labels_presented.append(line.split(' ')[0])

    # with open(f'xray_data/{mode}_list_reduced_empty.txt', 'r') as f:
    #     for line in f:
    #         total_lines.append(line.replace('xray_data','united_dataset'))
    #         # labels_presented.append(line.split(' ')[0])

    # with open(f'united_dataset/{mode}_list.txt', 'w') as f:
    #     for line in total_lines:
    #         f.write(line)


    # set1 = set(os.listdir('kaggle_data/labels'))
    # set2 = set(os.listdir('xray_data/labels'))
    # print(set1.intersection(set2), set2.intersection(set1))

    shutil.copytree('kaggle_data/labels_cleaned/.', 'united_dataset/labels_cleaned',  dirs_exist_ok = True)
    shutil.copytree('xray_data/labels_full/.', 'united_dataset/labels_cleaned',  dirs_exist_ok = True)

labels_presented = []
labels_presented_after = []
for l in os.listdir('united_dataset/labels_cleaned'):
    with open(f'united_dataset/labels_cleaned/{l}', 'r') as f:
        for line in f:
            labels_presented.append(line.split(' ')[0])

    if '_' in l:
        lines = []
        with open(f'united_dataset/labels_cleaned/{l}', 'r') as f:
            for line in f:
                lines.append(line)

        with open(f'united_dataset/labels_cleaned/{l}', 'w') as f:
            for line in lines:
                if int(line.split(' ')[0]) in [5,6,7]:
                    print(line)
                    line = f"{int(line.split(' ')[0]) - 1} {line[2:]}"
                    print(line)
                    print("_______")
                labels_presented_after.append(line.split(' ')[0])
                f.write(line)
    else:
        labels_presented_after.append(line.split(' ')[0])

print(np.unique(labels_presented))
print(np.unique(labels_presented_after))