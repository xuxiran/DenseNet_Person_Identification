import os
import mne
import json
import numpy as np
from tqdm import tqdm

if __name__ == "__main__":

    pre_way = 'pruned' # 'raw' , '32ch' or 'pruned'
    # raw 78 channels
    # 32ch 32 channels
    # pruned 32 channels

    save_traindir = '../pre_data/train' + pre_way + '/'
    save_testdir = '../pre_data/test' + pre_way + '/'

    if not os.path.exists(save_traindir):
        os.makedirs(save_traindir)
    if not os.path.exists(save_testdir):
        os.makedirs(save_testdir)

    # read the data
    if pre_way == 'raw' or pre_way == '32ch':
        traindir = '../eremus_dataset/raw/train/'
        testdir = '../eremus_dataset/raw/test_trial/'
    else:
        traindir = '../eremus_dataset/pruned/train/'
        testdir = '../eremus_dataset/pruned/test_trial/'
    trainfiles = []
    # read files in datadir
    trainfiles.extend([os.path.join(traindir, f) for f in os.listdir(traindir) if f.endswith(".fif")])
    testfiles = []
    testfiles.extend([os.path.join(testdir, f) for f in os.listdir(testdir) if f.endswith(".fif")])



    nandata = 0
    minlen = 1e9
    for file in testfiles:
        filename = file.split('/')[-1]
        filename = filename.replace('_eeg.fif', '')
        raw = mne.io.read_raw_fif(file, preload=True)
        data = raw.get_data()
        if pre_way == '32ch':
            data = data[4:36,:]

        # label = file["subject_id"]

        print(data.shape)
        # print(label)

        # detect is there any  NaNs in data
        if np.isnan(data).any():
            print(f"NaNs in {filename}")
            nandata += 1
            # replace nan with 0
            data = np.nan_to_num(data)

        if data.shape[1] < minlen:
            minlen = data.shape[1]

        # save data and label
        savedir1 = os.path.join(save_testdir, filename)
        np.save(savedir1, data)
        # savedir2 = os.path.join(save_testdir, f"{file['id']}_label.npy")
        # np.save(savedir2, label)


    nandata = 0
    minlen = 1e9
    # get eegdata and label
    for file in trainfiles:
        filename = file.split('/')[-1]
        filename = filename.replace('_eeg.fif', '')
        raw = mne.io.read_raw_fif(file, preload=True)
        data = raw.get_data()
        if pre_way == '32ch':
            data = data[4:36,:]

        #
        # # investigate whether data in some channel is all 0
        # data_channel = np.sum(np.abs(data), axis=1)

        print(data.shape)

        # detect is there any  NaNs in data
        if np.isnan(data).any():
            print(f"NaNs in {filename}")
            nandata += 1
            # replace nan with 0
            data = np.nan_to_num(data)

        if data.shape[1] < minlen:
            minlen = data.shape[1]

        # save data and label
        savedir1 = os.path.join(save_traindir, filename)
        np.save(savedir1, data)
        # savedir2 = os.path.join(save_traindir, f"{file['id']}_label.npy")
        # np.save(savedir2, label)




