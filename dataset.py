import torch
import config as cfg
from torch.utils.data import Dataset
import os
import numpy as np

class AADataset(Dataset):
    def __init__(self, dirs, files):
        all_train = np.load('../pre_data/all_train.npy')
        all_label = np.load('../pre_data/all_label.npy')
        self.len = cfg.eeglen
        trnum = len(files)
        self.seg = np.zeros((trnum,1))
        for index in range(trnum):
            eeg = np.load(dirs + str(files[index]) + '.npy').astype(np.float32)
            if np.size(eeg,1) < self.len:
                self.seg[index] = 0
            else:
                self.seg[index] = np.size(eeg,1)//self.len

        self.alleeg = []
        self.alllabel = []
        self.alltr = []

        for index in range(trnum):
            eeg = np.load(dirs + str(files[index]) + '.npy').astype(np.float32)
            # according to the all_train.npy and all_label.npy get index and label

            # modify value>100 to 0
            eeg[eeg>100] = 0

            index_index = np.where(all_train == files[index])[0]
            label = all_label[index_index]
            eeg = np.transpose(eeg)
            eeg_mean = np.mean(eeg, axis=1, keepdims=True)
            eeg_std = np.std(eeg, axis=1, keepdims=True) + 1e-6
            eeg = (eeg - eeg_mean) / eeg_std

            self.alleeg.append(eeg)
            self.alllabel.append(label)
            self.alltr.append(index)


    def __len__(self):
        return int(np.sum(self.seg))

    def find_index(self, target):
        acc = 0
        for i, num in enumerate(self.seg):
            acc += num
            if acc > target:
                return i, int(target - acc + num)

    def __getitem__(self, index):
        nparry_index, time_index = self.find_index(index)

        eeg = self.alleeg[nparry_index]
        label = self.alllabel[nparry_index][0]
        tr = self.alltr[nparry_index]

        eeg_seg = eeg[self.len*time_index:self.len*time_index+self.len,:]
        label_seg = label
        tr_seg = tr

        eeg_seg = torch.tensor(eeg_seg, dtype=torch.float32)

        return eeg_seg,label_seg

class AADataset_test(Dataset):
    def __init__(self, dirs, files,withlabel):
        self.dirs = dirs
        self.files = files
        self.len = cfg.eeglen
        self.withlabel = withlabel

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        eeg = np.load(self.dirs + str(self.files[index]) + '.npy').astype(np.float32)
        eeg[eeg>100] = 0
        eeg = np.transpose(eeg)
        eeg_mean = np.mean(eeg, axis=1, keepdims=True)
        eeg_std = np.std(eeg, axis=1, keepdims=True) + 1e-6
        eeg = (eeg - eeg_mean) / eeg_std
        eeg = torch.tensor(eeg, dtype=torch.float32)

        # further segment and normalization
        if self.withlabel:
            all_train = np.load('../pre_data/all_train.npy')
            all_label = np.load('../pre_data/all_label.npy')
            index_index = np.where(all_train == self.files[index])[0]
            label = all_label[index_index]
            return eeg,label

        else:
            return eeg


