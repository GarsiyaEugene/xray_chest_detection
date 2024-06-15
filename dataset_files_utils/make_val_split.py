import numpy as np

patients = set()
with open(f'xray_data/initial_splits/test_list.txt', 'r') as f: #train_val_list
    for line in f:
        patients.add(str(line.split('_')[0]))

patients = list(patients)
train_patients = patients.copy()
val_patients = list(np.random.choice(patients, size=int(0.1*len(patients)), replace=False))

for pat in val_patients:
    train_patients.remove(pat)

test_patients = list(np.random.choice(train_patients, size=int(0.1*len(patients)), replace=False))

for pat in test_patients:
    train_patients.remove(pat)

# test_patients = set()
# with open(f'xray_data/initial_splits/test_list.txt', 'r') as f:
#     for line in f:
#         test_patients.add(line.split('_')[0])

print(len(train_patients), len(val_patients), len(test_patients), len(patients))

train_file = []
val_file = []
test_file = []
with open(f'xray_data/initial_splits/test_list.txt', 'r') as f: #
    for line in f:
        if str(line.split('_')[0]) in train_patients:
            train_file.append(line)
        elif str(line.split('_')[0]) in val_patients:
            val_file.append(line)
        elif str(line.split('_')[0]) in test_patients:
            test_file.append(line)
        else:
            print('Error', str(line.split('_')[0]))

with open('xray_data/train_list.txt', 'w') as f:
    for line in train_file:
        f.write(f"{line}")

with open('xray_data/val_list.txt', 'w') as f:
    for line in val_file:
        f.write(f"{line}")

with open('xray_data/test_list.txt', 'w') as f:
    for line in test_file:
        f.write(f"{line}")
