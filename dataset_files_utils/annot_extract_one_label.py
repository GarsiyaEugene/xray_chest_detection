import os

class_to_extract = 4
split_suffix = '_list'
data_fold = 'united_dataset'
labels_fold ='labels_full'


os.makedirs(f'{data_fold}/labels_{class_to_extract}', exist_ok=True)

for mode in ['train', 'val', 'test']:
    with open(f'{data_fold}/{mode}{split_suffix}.txt', 'r') as f:
        for line in f:
            line = line[:-1].split('/')[-1].replace('.png', '')
            new_lines = []

            if os.path.exists(f'{data_fold}/{labels_fold}/{line}.txt'):
                with open(f'{data_fold}/{labels_fold}/{line}.txt', 'r') as f_annot:
                    for line_annot in f_annot:
                        label = int(line_annot[0])
                        if label==class_to_extract:
                            line_annot = f"{label-class_to_extract}{line_annot[1:]}"
                            new_lines.append(line_annot)

                if len(new_lines)!=0:
                    with open(f'{data_fold}/labels_{class_to_extract}/{line}.txt', 'w') as f_annot_new:
                        for line_new in new_lines:
                            f_annot_new.write(line_new)
