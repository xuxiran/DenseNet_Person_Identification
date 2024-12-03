import os
import mne
import json
import numpy as np
from tqdm import tqdm
file_info = '../eremus_dataset/splits_subject_identification.json'

files_info = json.load(open(file_info, 'r'))

train_files_info = files_info["train"] + files_info["val_trial"]
test_files_info = files_info["test_trial"]

# get 'subject_id' and 'id' from train_files_info and test_files_info

all_train = []
all_label = []

for file in train_files_info:
    file_id = file['id']
    subject_id = file['subject_id']

    all_train.append(file_id)
    all_label.append(subject_id)

savedir = './'
np.save(os.path.join(savedir, '../pre_data/all_train.npy'), all_train)
np.save(os.path.join(savedir, '../pre_data/all_label.npy'), all_label)